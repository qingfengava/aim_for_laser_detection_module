#include "laser_aim/app/laser_vision_node.hpp"

#include <wust_vl/common/utils/logger.hpp>

#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>

namespace laser_aim::app {

namespace {

constexpr const char* kLogNode = "laser.node";
constexpr double kEmaAlpha = 0.12;

struct DebugReloadTag {};
constexpr const char* kCmdLogPath = "/dev/shm/cmd_log.json";
constexpr const char* kCmdLogJsonlPath = "/dev/shm/cmd_log.jsonl";
constexpr int kGateReasonCount = 8;

double safeDivide(double numer, double denom) {
    return denom > 1e-12 ? numer / denom : 0.0;
}

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

struct DebugRunStats {
    int last_frame_id { -1 };
    uint64_t frames_counted { 0 };
    uint64_t valid_output_frames { 0 };
    uint64_t fire_enable_frames { 0 };
    uint64_t center_in_window_ok_frames { 0 };
    uint64_t predicted_window_ok_frames { 0 };
    uint64_t hit_window_both_ok_frames { 0 };
    uint64_t pnp_error_samples { 0 };
    double pnp_error_sum { 0.0 };
    double pnp_error_max { 0.0 };
    std::array<uint64_t, kGateReasonCount> gate_reason_counts {};
    std::array<uint64_t, kGateReasonCount> gate_fail_reason_counts {};
    uint64_t recapture_frames { 0 };
    uint64_t recapture_enter_count { 0 };
    uint64_t recapture_event_count { 0 };
    bool in_recapture { false };
    int recapture_len_frames { 0 };
    uint64_t relock_success_count { 0 };
    uint64_t relock_total_frames { 0 };
    int relock_max_frames { 0 };
    bool in_loss { false };
    int loss_len_frames { 0 };
    uint64_t loss_event_count { 0 };
    uint64_t loss_total_frames { 0 };
    int loss_max_frames { 0 };
    uint64_t backup_recovery_frames { 0 };
};

struct SelectedCandidateInfo {
    int index { -1 };
    std::optional<double> laser_reproj_error;
    bool recovered_from_backup { false };
};

SelectedCandidateInfo findSelectedCandidateInfo(const PipelineDebugFrame& dbg) {
    SelectedCandidateInfo info;
    if (!dbg.output.valid || dbg.candidates.empty()) {
        return info;
    }

    double best_dist = std::numeric_limits<double>::infinity();
    int best_idx = -1;
    for (std::size_t i = 0; i < dbg.candidates.size(); ++i) {
        const double dist = cv::norm(dbg.candidates[i].aim_center_px - dbg.output.aim_center_px);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<int>(i);
        }
    }
    if (best_idx < 0) {
        return info;
    }

    info.index = best_idx;
    const auto& c = dbg.candidates[static_cast<std::size_t>(best_idx)];
    info.recovered_from_backup = c.recovered_from_backup;
    if (std::isfinite(c.laser_reproj_error)) {
        info.laser_reproj_error = c.laser_reproj_error;
    }
    return info;
}

void updateDebugRunStats(const PipelineDebugFrame& dbg, DebugRunStats* stats) {
    if (stats == nullptr) {
        return;
    }
    if (dbg.frame.img_frame.src_img.empty()) {
        return;
    }
    if (dbg.frame.frame_id <= stats->last_frame_id) {
        return;
    }
    stats->last_frame_id = dbg.frame.frame_id;
    stats->frames_counted += 1;

    const auto& out = dbg.output;
    if (out.valid) {
        stats->valid_output_frames += 1;
    }
    if (out.fire_enable) {
        stats->fire_enable_frames += 1;
    }
    if (out.gate.center_in_window_ok) {
        stats->center_in_window_ok_frames += 1;
    }
    if (out.gate.predicted_window_ok) {
        stats->predicted_window_ok_frames += 1;
    }
    if (out.gate.center_in_window_ok && out.gate.predicted_window_ok) {
        stats->hit_window_both_ok_frames += 1;
    }

    const int reason_idx = static_cast<int>(out.gate.code);
    if (reason_idx >= 0 && reason_idx < kGateReasonCount) {
        stats->gate_reason_counts[static_cast<std::size_t>(reason_idx)] += 1;
        if (!out.fire_enable) {
            stats->gate_fail_reason_counts[static_cast<std::size_t>(reason_idx)] += 1;
        }
    }

    if (out.route_mode == RouteMode::CLASSIC_RECAPTURE) {
        stats->recapture_frames += 1;
        if (!stats->in_recapture) {
            stats->in_recapture = true;
            stats->recapture_len_frames = 1;
            stats->recapture_enter_count += 1;
        } else {
            stats->recapture_len_frames += 1;
        }
    } else if (stats->in_recapture) {
        stats->in_recapture = false;
        stats->recapture_event_count += 1;
        if (out.valid) {
            stats->relock_success_count += 1;
            stats->relock_total_frames += static_cast<uint64_t>(stats->recapture_len_frames);
            stats->relock_max_frames = std::max(stats->relock_max_frames, stats->recapture_len_frames);
        }
        stats->recapture_len_frames = 0;
    }

    if (!out.valid) {
        if (!stats->in_loss) {
            stats->in_loss = true;
            stats->loss_len_frames = 0;
        }
        stats->loss_len_frames += 1;
    } else if (stats->in_loss) {
        stats->in_loss = false;
        stats->loss_event_count += 1;
        stats->loss_total_frames += static_cast<uint64_t>(stats->loss_len_frames);
        stats->loss_max_frames = std::max(stats->loss_max_frames, stats->loss_len_frames);
        stats->loss_len_frames = 0;
    }

    const SelectedCandidateInfo selected = findSelectedCandidateInfo(dbg);
    if (selected.recovered_from_backup) {
        stats->backup_recovery_frames += 1;
    }
    if (selected.laser_reproj_error.has_value()) {
        stats->pnp_error_samples += 1;
        stats->pnp_error_sum += *selected.laser_reproj_error;
        stats->pnp_error_max = std::max(stats->pnp_error_max, *selected.laser_reproj_error);
    }
}

nlohmann::json gateReasonCountsToJson(const std::array<uint64_t, kGateReasonCount>& counts) {
    nlohmann::json out = nlohmann::json::object();
    for (int i = 0; i < kGateReasonCount; ++i) {
        out[toString(static_cast<GateReasonCode>(i))] = counts[static_cast<std::size_t>(i)];
    }
    return out;
}

nlohmann::json buildCmdLog(
    const PipelineDebugFrame& dbg,
    const LaserVisionNode::PerfMetricsSnapshot& perf,
    const DebugRunStats& stats
) {
    const SelectedCandidateInfo selected = findSelectedCandidateInfo(dbg);
    const double selected_pnp_err = selected.laser_reproj_error.value_or(-1.0);
    const double hit_window_rate = safeDivide(
        static_cast<double>(stats.hit_window_both_ok_frames),
        static_cast<double>(stats.frames_counted)
    );
    const double pnp_error_mean = safeDivide(
        stats.pnp_error_sum,
        static_cast<double>(stats.pnp_error_samples)
    );
    const double loss_avg_frames = safeDivide(
        static_cast<double>(stats.loss_total_frames),
        static_cast<double>(stats.loss_event_count)
    );
    const double relock_avg_frames = safeDivide(
        static_cast<double>(stats.relock_total_frames),
        static_cast<double>(stats.relock_success_count)
    );

    nlohmann::json cand_scores = nlohmann::json::array();
    for (const auto& c : dbg.candidates) {
        const double laser_err = std::isfinite(c.laser_reproj_error) ? c.laser_reproj_error : -1.0;
        const double armor_err = std::isfinite(c.armor_reproj_error) ? c.armor_reproj_error : -1.0;
        nlohmann::json item;
        item["id"] = c.candidate_id;
        item["class"] = toString(c.raw_class);
        item["conf"] = c.detector_confidence;
        item["geom_score"] = c.geom_consistency_score;
        item["color_score"] = c.color.color_score;
        item["color_confident"] = c.color.confident;
        item["color_decided"] = toString(c.color.decided);
        item["aim_from_pnp"] = c.aim_center_from_pnp;
        item["aim_x"] = c.aim_center_px.x;
        item["aim_y"] = c.aim_center_px.y;
        item["geom_x"] = c.geometric_center_px.x;
        item["geom_y"] = c.geometric_center_px.y;
        item["laser_err"] = laser_err;
        item["armor_err"] = armor_err;
        item["laser_template_valid"] = c.laser_template_valid;
        item["armor_template_valid"] = c.armor_template_valid;
        item["pnp_ambiguous"] = c.pnp_ambiguous;
        item["pnp_margin_delta"] = c.pnp_margin_delta;
        item["pnp_margin_ratio"] = c.pnp_margin_ratio;
        item["from_classic"] = c.from_classic;
        item["from_yolo"] = c.from_yolo;
        item["direction_ratio"] = c.direction_ratio;
        item["direction_confidence"] = c.direction_confidence;
        item["direction_hint"] = toString(c.direction_hint);
        item["pair_mode"] = toString(c.pair_mode);
        cand_scores.push_back(std::move(item));
    }

    nlohmann::json j;
    j["schema_version"] = "2.0";
    j["source"] = "laser_aim";
    j["paths"] = {
        { "snapshot_json", kCmdLogPath },
        { "replay_jsonl", kCmdLogJsonlPath }
    };
    j["frame"] = {
        { "frame_id", dbg.frame.frame_id },
        { "timestamp_ns", dbg.output.timestamp_ns }
    };
    j["policy"] = {
        { "team_color", toString(dbg.policy.team_color) },
        { "enemy_color", toString(dbg.policy.enemy_color) },
        { "stage_ref_raw", dbg.policy.stage_ref_raw },
        { "safe_track_only", dbg.policy.isSafeTrackOnly() }
    };
    j["output"] = {
        { "valid", dbg.output.valid },
        { "aim_center_px", { dbg.output.aim_center_px.x, dbg.output.aim_center_px.y } },
        { "predicted_aim_center_px", { dbg.output.predicted_aim_center_px.x, dbg.output.predicted_aim_center_px.y } },
        { "yaw_cmd", dbg.output.yaw_cmd },
        { "pitch_cmd", dbg.output.pitch_cmd },
        { "fire_enable", dbg.output.fire_enable },
        { "lock_stage", toString(dbg.output.lock_stage) },
        { "route_mode", toString(dbg.output.route_mode) },
        { "selected_candidate_index", selected.index },
        { "selected_pnp_laser_reproj_error", selected_pnp_err },
        { "selected_from_backup", selected.recovered_from_backup }
    };
    j["track"] = {
        { "enemy_prob_avg", dbg.output.track.enemy_prob_avg },
        { "laser_prob_avg", dbg.output.track.laser_prob_avg },
        { "confirm_frames", dbg.output.track.continuous_confirm_frames },
        { "innovation_norm_m", dbg.output.track.innovation_norm_m },
        { "innovation_rejected", dbg.output.track.innovation_rejected }
    };
    j["gate"] = {
        { "reason", dbg.output.gate_reason },
        { "reason_code", static_cast<int>(dbg.output.gate.code) },
        { "reason_detail", dbg.output.gate.reason },
        { "enemy_conf_ok", dbg.output.gate.enemy_conf_ok },
        { "laser_conf_ok", dbg.output.gate.laser_conf_ok },
        { "pnp_error_ok", dbg.output.gate.pnp_error_ok },
        { "track_stable_ok", dbg.output.gate.track_stable_ok },
        { "center_in_window_ok", dbg.output.gate.center_in_window_ok },
        { "predicted_window_ok", dbg.output.gate.predicted_window_ok },
        { "hit_window_ok", dbg.output.gate.center_in_window_ok && dbg.output.gate.predicted_window_ok },
        { "safe_mode_ok", dbg.output.gate.safe_mode_ok }
    };
    j["pnp"] = {
        { "selected_laser_reproj_error", selected_pnp_err },
        { "frame_valid_samples", dbg.pnp_calib.frame_valid_samples },
        { "frame_pred_laser", dbg.pnp_calib.frame_pred_laser },
        { "frame_pred_armor", dbg.pnp_calib.frame_pred_armor },
        { "frame_pred_unknown", dbg.pnp_calib.frame_pred_unknown },
        { "frame_ambiguous", dbg.pnp_calib.frame_ambiguous },
        { "frame_mean_margin_delta", dbg.pnp_calib.frame_mean_margin_delta },
        { "frame_mean_margin_ratio", dbg.pnp_calib.frame_mean_margin_ratio },
        { "frame_laser_as_armor_ratio", dbg.pnp_calib.frame_laser_as_armor_ratio },
        { "total_valid_samples", dbg.pnp_calib.total_valid_samples },
        { "total_pred_laser", dbg.pnp_calib.total_pred_laser },
        { "total_pred_armor", dbg.pnp_calib.total_pred_armor },
        { "total_pred_unknown", dbg.pnp_calib.total_pred_unknown },
        { "total_ambiguous", dbg.pnp_calib.total_ambiguous },
        { "total_expected_laser_samples", dbg.pnp_calib.total_expected_laser_samples },
        { "total_laser_as_armor", dbg.pnp_calib.total_laser_as_armor },
        { "total_laser_as_armor_ratio", dbg.pnp_calib.total_laser_as_armor_ratio }
    };
    j["perf"] = {
        { "total_arrived_frames", perf.total_arrived },
        { "dropped_busy_frames", perf.dropped_busy },
        { "dropped_enqueue_frames", perf.dropped_enqueue },
        { "dropped_total_frames", perf.dropped_total },
        { "processed_frames", perf.processed },
        { "drop_rate", perf.drop_rate },
        { "capture_fps", perf.capture_fps },
        { "jitter_ms", perf.jitter_ms },
        { "e2e_latency_ms", perf.e2e_latency_ms },
        { "e2e_latency_max_ms", perf.e2e_latency_max_ms },
        { "fps_ok", perf.fps_ok },
        { "drop_ok", perf.drop_ok },
        { "jitter_ok", perf.jitter_ok },
        { "latency_ok", perf.latency_ok },
        { "in_target", perf.in_target }
    };
    j["stats"] = {
        { "frames_counted", stats.frames_counted },
        { "valid_output_frames", stats.valid_output_frames },
        { "fire_enable_frames", stats.fire_enable_frames },
        { "center_in_window_ok_frames", stats.center_in_window_ok_frames },
        { "predicted_window_ok_frames", stats.predicted_window_ok_frames },
        { "hit_window_both_ok_frames", stats.hit_window_both_ok_frames },
        { "hit_window_rate", hit_window_rate },
        { "pnp_error_samples", stats.pnp_error_samples },
        { "pnp_error_mean", pnp_error_mean },
        { "pnp_error_max", stats.pnp_error_max },
        { "gate_reason_counts", gateReasonCountsToJson(stats.gate_reason_counts) },
        { "gate_fail_reason_counts", gateReasonCountsToJson(stats.gate_fail_reason_counts) },
        { "recapture_frames", stats.recapture_frames },
        { "recapture_enter_count", stats.recapture_enter_count },
        { "recapture_event_count", stats.recapture_event_count },
        { "relock_success_count", stats.relock_success_count },
        { "relock_avg_frames", relock_avg_frames },
        { "relock_max_frames", stats.relock_max_frames },
        { "loss_event_count", stats.loss_event_count },
        { "loss_avg_frames", loss_avg_frames },
        { "loss_max_frames", stats.loss_max_frames },
        { "backup_recovery_frames", stats.backup_recovery_frames }
    };
    j["candidates"] = std::move(cand_scores);

    // Legacy flattened aliases for quick shell parsing.
    j["frame_id"] = dbg.frame.frame_id;
    j["timestamp_ns"] = dbg.output.timestamp_ns;
    j["aim_x"] = dbg.output.aim_center_px.x;
    j["aim_y"] = dbg.output.aim_center_px.y;
    j["aim_pred_x"] = dbg.output.predicted_aim_center_px.x;
    j["aim_pred_y"] = dbg.output.predicted_aim_center_px.y;
    j["yaw_cmd"] = dbg.output.yaw_cmd;
    j["pitch_cmd"] = dbg.output.pitch_cmd;
    j["fire_enable"] = dbg.output.fire_enable;
    j["stage"] = toString(dbg.output.lock_stage);
    j["route_mode"] = toString(dbg.output.route_mode);
    j["gate_reason"] = dbg.output.gate_reason;
    j["selected_pnp_error"] = selected_pnp_err;
    j["hit_window_ok"] = dbg.output.gate.center_in_window_ok && dbg.output.gate.predicted_window_ok;
    j["candidate_scores"] = j["candidates"];
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
    {
        std::ofstream ofs(kCmdLogJsonlPath, std::ios::out | std::ios::trunc);
        (void)ofs;
    }
    DebugRunStats run_stats;

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

        const int prev_logged_frame_id = run_stats.last_frame_id;
        updateDebugRunStats(dbg, &run_stats);
        const nlohmann::json cmd_log = buildCmdLog(dbg, perf, run_stats);
        writeJsonAtomically(kCmdLogPath, cmd_log);
        if (run_stats.last_frame_id > prev_logged_frame_id) {
            appendJsonLine(kCmdLogJsonlPath, cmd_log);
        }

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
