#include "laser_aim/app/laser_vision_node.hpp"

#include <wust_vl/common/utils/logger.hpp>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>

namespace laser_aim::app {

namespace {

constexpr const char* kLogNode = "laser.node";
constexpr double kEmaAlpha = 0.12;

struct DebugReloadTag {};

void buildCameraInfo(
    const YAML::Node& camera_config,
    cv::Mat* k_out,
    cv::Mat* d_out
) {
    if (k_out == nullptr || d_out == nullptr) {
        return;
    }

    const std::string camera_info_path =
        utils::expandEnv(camera_config["camera_info_path"].as<std::string>(""));
    if (camera_info_path.empty()) {
        *k_out = cv::Mat::eye(3, 3, CV_64F);
        *d_out = cv::Mat::zeros(1, 5, CV_64F);
        return;
    }

    YAML::Node info;
    try {
        info = YAML::LoadFile(camera_info_path);
    } catch (...) {
        *k_out = cv::Mat::eye(3, 3, CV_64F);
        *d_out = cv::Mat::zeros(1, 5, CV_64F);
        return;
    }

    const auto k_data = info["camera_matrix"]["data"].as<std::vector<double>>();
    const auto d_data = info["distortion_coefficients"]["data"].as<std::vector<double>>();

    if (k_data.size() == 9) {
        cv::Mat k(3, 3, CV_64F);
        std::memcpy(k.data, k_data.data(), 9 * sizeof(double));
        *k_out = k.clone();
    } else {
        *k_out = cv::Mat::eye(3, 3, CV_64F);
    }

    if (!d_data.empty()) {
        cv::Mat d(1, static_cast<int>(d_data.size()), CV_64F);
        std::memcpy(d.data, d_data.data(), d_data.size() * sizeof(double));
        *d_out = d.clone();
    } else {
        *d_out = cv::Mat::zeros(1, 5, CV_64F);
    }
}

nlohmann::json buildCmdLog(
    const PipelineDebugFrame& dbg,
    const LaserVisionNode::PerfMetricsSnapshot& perf
) {
    nlohmann::json j;
    j["frame_id"] = dbg.frame.frame_id;
    j["aim_x"] = dbg.output.aim_center_px.x;
    j["aim_y"] = dbg.output.aim_center_px.y;
    j["aim_pred_x"] = dbg.output.predicted_aim_center_px.x;
    j["aim_pred_y"] = dbg.output.predicted_aim_center_px.y;
    j["yaw_cmd"] = dbg.output.yaw_cmd;
    j["pitch_cmd"] = dbg.output.pitch_cmd;
    j["fire_enable"] = dbg.output.fire_enable;
    j["stage"] = toString(dbg.output.lock_stage);
    j["route_mode"] = toString(dbg.output.route_mode);
    j["stage_ref_raw"] = dbg.policy.stage_ref_raw;
    j["enemy_prob"] = dbg.output.track.enemy_prob_avg;
    j["laser_prob"] = dbg.output.track.laser_prob_avg;
    j["confirm_frames"] = dbg.output.track.continuous_confirm_frames;
    j["gate_reason"] = dbg.output.gate.reason;
    j["predicted_window_ok"] = dbg.output.gate.predicted_window_ok;
    j["track_stable_ok"] = dbg.output.gate.track_stable_ok;
    j["total_arrived_frames"] = perf.total_arrived;
    j["dropped_busy_frames"] = perf.dropped_busy;
    j["dropped_enqueue_frames"] = perf.dropped_enqueue;
    j["dropped_total_frames"] = perf.dropped_total;
    j["processed_frames"] = perf.processed;
    j["drop_rate"] = perf.drop_rate;
    j["capture_fps"] = perf.capture_fps;
    j["jitter_ms"] = perf.jitter_ms;
    j["e2e_latency_ms"] = perf.e2e_latency_ms;
    j["e2e_latency_max_ms"] = perf.e2e_latency_max_ms;
    j["fps_ok"] = perf.fps_ok;
    j["drop_ok"] = perf.drop_ok;
    j["jitter_ok"] = perf.jitter_ok;
    j["latency_ok"] = perf.latency_ok;
    j["perf_in_target"] = perf.in_target;
    return j;
}

} // namespace

LaserVisionNode::LaserVisionNode(
    std::string common_cfg,
    std::string camera_cfg,
    std::string pipeline_cfg,
    std::string model_cfg
):
    common_cfg_(std::move(common_cfg)),
    camera_cfg_(std::move(camera_cfg)),
    pipeline_cfg_(std::move(pipeline_cfg)),
    model_cfg_(std::move(model_cfg)) {}

LaserVisionNode::~LaserVisionNode() {
    stop();
}

bool LaserVisionNode::init(bool debug_mode) {
    debug_mode_ = debug_mode;

    if (!param_hub_.init(common_cfg_, pipeline_cfg_, model_cfg_)) {
        return false;
    }

    YAML::Node camera_cfg = YAML::LoadFile(camera_cfg_);
    camera_ = std::make_shared<wust_vl::video::Camera>();
    if (!camera_->init(camera_cfg)) {
        WUST_ERROR(kLogNode) << "camera init failed";
        return false;
    }

    buildCameraInfo(camera_cfg, &camera_k_, &camera_d_);

    if (!pipeline_.init(&param_hub_, camera_k_, camera_d_)) {
        WUST_ERROR(kLogNode) << "pipeline init failed";
        return false;
    }

    camera_->setFrameCallback(
        std::bind(&LaserVisionNode::frameCallback, this, std::placeholders::_1)
    );

    const int workers = std::max(1, param_hub_.runtime().max_infer_running);
    thread_pool_ = std::make_unique<wust_vl::common::concurrency::ThreadPool>(
        static_cast<std::size_t>(workers)
    );

    WUST_MAIN(kLogNode) << "node initialized, workers=" << workers;
    return true;
}

void LaserVisionNode::start() {
    if (run_flag_) {
        return;
    }

    run_flag_ = true;
    if (camera_) {
        camera_->start();
    }

    if (debug_mode_) {
        debug_thread_ = std::thread([this]() { this->debugLoop(); });
    }
}

void LaserVisionNode::stop() {
    if (!run_flag_) {
        return;
    }

    run_flag_ = false;
    if (camera_) {
        camera_->stop();
    }

    if (debug_thread_.joinable()) {
        debug_thread_.join();
    }
}

void LaserVisionNode::frameCallback(wust_vl::video::ImageFrame& img_frame) {
    if (!run_flag_) {
        return;
    }

    if (img_frame.src_img.empty()) {
        return;
    }

    const auto receive_ts = std::chrono::steady_clock::now();
    recordFrameArrival(receive_ts);

    const int max_running = std::max(1, param_hub_.runtime().max_infer_running);
    if (infer_running_count_.load() >= max_running) {
        recordFrameDroppedBusy();
        return;
    }

    if (!thread_pool_) {
        recordFrameDroppedEnqueue();
        return;
    }

    infer_running_count_++;
    const auto ok = thread_pool_->enqueue([this, img_frame = std::move(img_frame), receive_ts]() mutable {
        this->processFrame(std::move(img_frame), receive_ts);
        infer_running_count_--;
    });

    if (!ok) {
        recordFrameDroppedEnqueue();
        infer_running_count_--;
    }
}

void LaserVisionNode::processFrame(
    wust_vl::video::ImageFrame img_frame,
    std::chrono::steady_clock::time_point receive_ts
) {
    FrameContext ctx;
    ctx.frame_id = frame_id_gen_++;
    ctx.receive_ts = receive_ts;
    img_frame.timestamp = receive_ts; // Normalize all downstream timing to steady_clock.
    ctx.img_frame = std::move(img_frame);

    PipelineDebugFrame dbg;
    const AimOutput out = pipeline_.process(ctx, &dbg);
    const auto done_ts = std::chrono::steady_clock::now();
    recordFrameProcessed(receive_ts, done_ts);

    (void)out;
    {
        std::lock_guard<std::mutex> lock(dbg_mutex_);
        dbg_frame_ = std::move(dbg);
    }
}

void LaserVisionNode::debugLoop() {
    ShmWriter shm_writer("/debug_frame");

    while (run_flag_) {
        const auto loop_start = std::chrono::steady_clock::now();

        PipelineDebugFrame dbg;
        PerfMetricsSnapshot perf;
        {
            std::lock_guard<std::mutex> lock(dbg_mutex_);
            dbg = dbg_frame_;
        }
        perf = snapshotPerfMetrics();

        drawDebugOverlayImpl(
            dbg,
            false,
            [](cv::Mat& img, const PipelineDebugFrame& d) {
                modules::perception::drawPipelineDebug(img, d);
            },
            shm_writer
        );

        writeJsonAtomically("/dev/shm/cmd_log.json", buildCmdLog(dbg, perf));

        const auto now = std::chrono::steady_clock::now();
        const int perf_log_interval_ms = std::max(200, param_hub_.runtime().perf_log_interval_ms);
        const auto perf_interval = std::chrono::milliseconds(perf_log_interval_ms);
        if (last_perf_log_ts_.time_since_epoch().count() == 0 || now - last_perf_log_ts_ >= perf_interval) {
            last_perf_log_ts_ = now;
            WUST_MAIN(kLogNode) << "perf fps=" << perf.capture_fps << " drop_rate=" << perf.drop_rate
                                << " jitter_ms=" << perf.jitter_ms << " e2e_ms=" << perf.e2e_latency_ms
                                << " e2e_max_ms=" << perf.e2e_latency_max_ms
                                << " in_target=" << (perf.in_target ? "true" : "false");
        }

        utils::xSecOnce<DebugReloadTag>(
            [this]() {
                param_hub_.reloadAll();
            },
            1.0
        );

        const int fps = std::max(1, param_hub_.runtime().debug_fps);
        const auto interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / fps));
        const auto elapsed = std::chrono::steady_clock::now() - loop_start;
        if (elapsed < interval) {
            std::this_thread::sleep_for(interval - elapsed);
        }
    }
}

void LaserVisionNode::recordFrameArrival(std::chrono::steady_clock::time_point ts) {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    total_arrived_frames_ += 1;
    if (last_arrival_ts_.time_since_epoch().count() > 0) {
        const double interval_ms = std::chrono::duration<double, std::milli>(ts - last_arrival_ts_).count();
        if (interval_ms > 0.0) {
            if (ema_arrival_period_ms_ <= 1e-6) {
                ema_arrival_period_ms_ = interval_ms;
            } else {
                const double prev = ema_arrival_period_ms_;
                ema_arrival_period_ms_ = (1.0 - kEmaAlpha) * ema_arrival_period_ms_ + kEmaAlpha * interval_ms;
                const double jitter = std::abs(interval_ms - prev);
                ema_arrival_jitter_ms_ = (1.0 - kEmaAlpha) * ema_arrival_jitter_ms_ + kEmaAlpha * jitter;
            }
        }
    }
    last_arrival_ts_ = ts;
}

void LaserVisionNode::recordFrameDroppedBusy() {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    dropped_busy_frames_ += 1;
}

void LaserVisionNode::recordFrameDroppedEnqueue() {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    dropped_enqueue_frames_ += 1;
}

void LaserVisionNode::recordFrameProcessed(
    std::chrono::steady_clock::time_point receive_ts,
    std::chrono::steady_clock::time_point done_ts
) {
    const double latency_ms = std::chrono::duration<double, std::milli>(done_ts - receive_ts).count();
    std::lock_guard<std::mutex> lock(perf_mutex_);
    processed_frames_ += 1;
    if (ema_e2e_latency_ms_ <= 1e-6) {
        ema_e2e_latency_ms_ = latency_ms;
    } else {
        ema_e2e_latency_ms_ = (1.0 - kEmaAlpha) * ema_e2e_latency_ms_ + kEmaAlpha * latency_ms;
    }
    max_e2e_latency_ms_ = std::max(max_e2e_latency_ms_, latency_ms);
}

LaserVisionNode::PerfMetricsSnapshot LaserVisionNode::snapshotPerfMetrics() {
    PerfMetricsSnapshot s;
    const auto& rt_cfg = param_hub_.runtime();
    std::lock_guard<std::mutex> lock(perf_mutex_);
    s.total_arrived = total_arrived_frames_;
    s.dropped_busy = dropped_busy_frames_;
    s.dropped_enqueue = dropped_enqueue_frames_;
    s.dropped_total = dropped_busy_frames_ + dropped_enqueue_frames_;
    s.processed = processed_frames_;
    s.drop_rate = (s.total_arrived > 0)
        ? static_cast<double>(s.dropped_total) / static_cast<double>(s.total_arrived)
        : 0.0;
    s.capture_fps = (ema_arrival_period_ms_ > 1e-6) ? 1000.0 / ema_arrival_period_ms_ : 0.0;
    s.jitter_ms = ema_arrival_jitter_ms_;
    s.e2e_latency_ms = ema_e2e_latency_ms_;
    s.e2e_latency_max_ms = max_e2e_latency_ms_;

    s.fps_ok = s.capture_fps >= static_cast<double>(rt_cfg.target_capture_fps_min)
        && s.capture_fps <= static_cast<double>(rt_cfg.target_capture_fps_max);
    s.drop_ok = s.drop_rate <= rt_cfg.max_drop_rate;
    s.jitter_ok = s.jitter_ms <= rt_cfg.max_jitter_ms;
    s.latency_ok = s.e2e_latency_ms <= rt_cfg.max_e2e_latency_ms;
    s.in_target = s.fps_ok && s.drop_ok && s.jitter_ok && s.latency_ok;
    return s;
}

} // namespace laser_aim::app
