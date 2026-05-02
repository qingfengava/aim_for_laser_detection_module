#pragma once

#include "laser_aim/common/types.hpp"
#include "laser_aim/utils/debug_utils.hpp"
#include "laser_aim/utils/runtime_utils.hpp"
#include "laser_aim/modules/config/parameter_hub.hpp"
#include "laser_aim/modules/perception/laser_pipeline.hpp"

#include <wust_vl/common/concurrency/ThreadPool.h>
#include <wust_vl/video/camera.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <thread>

namespace laser_aim::app {

class LaserVisionNode {
public:
    LaserVisionNode(
        std::string common_cfg,
        std::string camera_cfg,
        std::string pipeline_cfg,
        std::string model_cfg
    );

    ~LaserVisionNode();

    bool init(bool debug_mode);
    void start();
    void stop();

private:
    void frameCallback(wust_vl::video::ImageFrame& img_frame);
    void processFrame(
        wust_vl::video::ImageFrame img_frame,
        std::chrono::steady_clock::time_point receive_ts
    );
    void debugLoop();

public:
    struct PerfMetricsSnapshot {
        uint64_t total_arrived { 0 };
        uint64_t dropped_busy { 0 };
        uint64_t dropped_enqueue { 0 };
        uint64_t dropped_total { 0 };
        uint64_t processed { 0 };
        double drop_rate { 0.0 };
        double capture_fps { 0.0 };
        double jitter_ms { 0.0 };
        double e2e_latency_ms { 0.0 };
        double e2e_latency_max_ms { 0.0 };
        bool fps_ok { false };
        bool drop_ok { false };
        bool jitter_ok { false };
        bool latency_ok { false };
        bool in_target { false };
    };

private:
    void recordFrameArrival(std::chrono::steady_clock::time_point ts);
    void recordFrameDroppedBusy();
    void recordFrameDroppedEnqueue();
    void recordFrameProcessed(
        std::chrono::steady_clock::time_point receive_ts,
        std::chrono::steady_clock::time_point done_ts
    );
    PerfMetricsSnapshot snapshotPerfMetrics();

private:
    std::string common_cfg_;
    std::string camera_cfg_;
    std::string pipeline_cfg_;
    std::string model_cfg_;

    bool debug_mode_ { false };
    std::atomic<bool> run_flag_ { false };

    std::shared_ptr<wust_vl::video::Camera> camera_;
    std::unique_ptr<wust_vl::common::concurrency::ThreadPool> thread_pool_;

    modules::config::ParameterHub param_hub_;
    modules::perception::LaserPipeline pipeline_;

    cv::Mat camera_k_;
    cv::Mat camera_d_;

    std::atomic<int> frame_id_gen_ { 0 };
    std::atomic<int> infer_running_count_ { 0 };

    std::thread debug_thread_;
    std::mutex dbg_mutex_;
    PipelineDebugFrame dbg_frame_;

    mutable std::mutex perf_mutex_;
    uint64_t total_arrived_frames_ { 0 };
    uint64_t dropped_busy_frames_ { 0 };
    uint64_t dropped_enqueue_frames_ { 0 };
    uint64_t processed_frames_ { 0 };
    std::chrono::steady_clock::time_point last_arrival_ts_ {};
    std::chrono::steady_clock::time_point last_perf_log_ts_ {};
    double ema_arrival_period_ms_ { 0.0 };
    double ema_arrival_jitter_ms_ { 0.0 };
    double ema_e2e_latency_ms_ { 0.0 };
    double max_e2e_latency_ms_ { 0.0 };
};

} // namespace laser_aim::app
