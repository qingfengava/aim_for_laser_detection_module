#pragma once

#include "laser_aim/common/types.hpp"

#include <wust_vl/common/utils/logger.hpp>
#include <wust_vl/common/utils/parameter.hpp>

#include <memory>
#include <string>

namespace laser_aim::modules::config {

struct LoggerConfig : wust_vl::common::utils::ParamGroup {
    static constexpr const char* kKey = "logger";
    static constexpr const char* kLogNode = "config.logger";

    static std::shared_ptr<LoggerConfig> create() {
        return std::make_shared<LoggerConfig>();
    }

    const char* key() const override {
        return kKey;
    }

    void loadSelf(const YAML::Node& node) override {
        if (first_load) {
            return;
        }

        const std::string level = node["log_level"].as<std::string>("INFO");
        const std::string path = node["log_path"].as<std::string>("log");
        const bool use_cli = node["use_logcli"].as<bool>(true);
        const bool use_file = node["use_logfile"].as<bool>(false);
        const bool simple = node["use_simplelog"].as<bool>(true);
        wust_vl::initLogger(level, path, use_cli, use_file, simple);
        WUST_MAIN(kLogNode) << "logger initialized";
        first_load = true;
    }

private:
    bool first_load { false };
};

struct SystemConfig : wust_vl::common::utils::SimpleConfigBase<SystemConfig> {
    static constexpr const char* kKey = "system";
    static constexpr const char* kLogNode = "config.system";

    int team_color_raw { -1 };
    bool unknown_color_safe_mode { true };
    int stage_ref_raw { -1 };

    void loadSelf(const YAML::Node& node) override {
        loadOnceOrUpdate(
            node,
            team_color_raw,
            [](const YAML::Node& n, int& v) { v = n["team_color"].as<int>(-1); },
            [](const YAML::Node& n, int& v) {
                const int nv = n["team_color"].as<int>(-1);
                if (nv != v) {
                    v = nv;
                    WUST_DEBUG(kLogNode) << "team_color changed to " << nv;
                }
            }
        );
        unknown_color_safe_mode = node["unknown_color_safe_mode"].as<bool>(true);
        stage_ref_raw = node["stage_ref"].as<int>(-1);
    }

    [[nodiscard]] TeamPolicy toPolicy() const {
        TeamPolicy policy;
        if (team_color_raw == 0) {
            policy.team_color = TeamColor::RED;
        } else if (team_color_raw == 1) {
            policy.team_color = TeamColor::BLUE;
        } else {
            policy.team_color = TeamColor::UNKNOWN;
        }
        policy.enemy_color = opposite(policy.team_color);
        policy.unknown_color_safe_mode = unknown_color_safe_mode;
        policy.stage_ref_raw = stage_ref_raw;
        return policy;
    }
};

struct RuntimeConfig : wust_vl::common::utils::SimpleConfigBase<RuntimeConfig> {
    static constexpr const char* kKey = "runtime";
    static constexpr const char* kLogNode = "config.runtime";

    int debug_fps { 60 };
    int max_infer_running { 1 };
    int debug_reload_interval_ms { 1000 };
    int target_capture_fps_min { 90 };
    int target_capture_fps_max { 120 };
    double max_drop_rate { 0.05 };
    double max_jitter_ms { 2.5 };
    double max_e2e_latency_ms { 35.0 };
    int perf_log_interval_ms { 1000 };

    void loadSelf(const YAML::Node& node) override {
        loadOnceOrUpdate(
            node,
            debug_fps,
            [](const YAML::Node& n, int& v) { v = n["debug_fps"].as<int>(60); },
            [](const YAML::Node& n, int& v) { v = n["debug_fps"].as<int>(60); }
        );
        max_infer_running = node["max_infer_running"].as<int>(1);
        debug_reload_interval_ms = node["debug_reload_interval_ms"].as<int>(1000);
        target_capture_fps_min = node["target_capture_fps_min"].as<int>(90);
        target_capture_fps_max = node["target_capture_fps_max"].as<int>(120);
        max_drop_rate = node["max_drop_rate"].as<double>(0.05);
        max_jitter_ms = node["max_jitter_ms"].as<double>(2.5);
        max_e2e_latency_ms = node["max_e2e_latency_ms"].as<double>(35.0);
        perf_log_interval_ms = node["perf_log_interval_ms"].as<int>(1000);
    }
};

struct PipelineConfig : wust_vl::common::utils::ParamGroup {
    static constexpr const char* kKey = "pipeline";
    static constexpr const char* kLogNode = "config.pipeline";

    const char* key() const override {
        return kKey;
    }

    static std::shared_ptr<PipelineConfig> create() {
        return std::make_shared<PipelineConfig>();
    }

    GEN_PARAM(int, lock_confirm_frames);
    GEN_PARAM(double, enemy_color_score_min);
    GEN_PARAM(double, enemy_prob_min);
    GEN_PARAM(double, laser_prob_min);
    GEN_PARAM(double, laser_reproj_error_max);
    GEN_PARAM(double, armor_laser_error_margin);
    GEN_PARAM(double, gate_center_window_px);
    GEN_PARAM(double, stage2_scale);
    GEN_PARAM(double, stage3_scale);
    GEN_PARAM(double, fusion_iou_min);
    GEN_PARAM(double, fusion_w_model);
    GEN_PARAM(double, fusion_w_geom);
    GEN_PARAM(double, fusion_w_temporal);
    GEN_PARAM(bool, enable_undistort);
    GEN_PARAM(double, undistort_alpha);
    GEN_PARAM(int, undistort_border_px);
    GEN_PARAM(bool, use_classic_only);
    GEN_PARAM(double, stage3_color_weight);
    GEN_PARAM(int, low_conf_recap_frames);
    GEN_PARAM(int, recover_frames);
    GEN_PARAM(double, control_latency_ms);
    GEN_PARAM(double, pid_yaw_kp);
    GEN_PARAM(double, pid_yaw_ki);
    GEN_PARAM(double, pid_yaw_kd);
    GEN_PARAM(double, pid_pitch_kp);
    GEN_PARAM(double, pid_pitch_ki);
    GEN_PARAM(double, pid_pitch_kd);
    GEN_PARAM(double, pid_output_limit);
    GEN_PARAM(double, color_min_s);
    GEN_PARAM(double, color_min_v);
    GEN_PARAM(int, ekf_min_confirm_frames);
    GEN_PARAM(double, ekf_max_speed_mps);

    void loadSelf(const YAML::Node& node) override {
        lock_confirm_frames_param.load(node);
        enemy_color_score_min_param.load(node);
        enemy_prob_min_param.load(node);
        laser_prob_min_param.load(node);
        laser_reproj_error_max_param.load(node);
        armor_laser_error_margin_param.load(node);
        gate_center_window_px_param.load(node);
        stage2_scale_param.load(node);
        stage3_scale_param.load(node);
        fusion_iou_min_param.load(node);
        fusion_w_model_param.load(node);
        fusion_w_geom_param.load(node);
        fusion_w_temporal_param.load(node);
        enable_undistort_param.load(node);
        undistort_alpha_param.load(node);
        undistort_border_px_param.load(node);
        use_classic_only_param.load(node);
        stage3_color_weight_param.load(node);
        low_conf_recap_frames_param.load(node);
        recover_frames_param.load(node);
        control_latency_ms_param.load(node);
        pid_yaw_kp_param.load(node);
        pid_yaw_ki_param.load(node);
        pid_yaw_kd_param.load(node);
        pid_pitch_kp_param.load(node);
        pid_pitch_ki_param.load(node);
        pid_pitch_kd_param.load(node);
        pid_output_limit_param.load(node);
        color_min_s_param.load(node);
        color_min_v_param.load(node);
        ekf_min_confirm_frames_param.load(node);
        ekf_max_speed_mps_param.load(node);
    }
};

struct ModelConfig : wust_vl::common::utils::SimpleConfigBase<ModelConfig> {
    static constexpr const char* kKey = "model";
    static constexpr const char* kLogNode = "config.model";

    std::string yolo_backend = "onnxruntime";
    std::string yolo_model_path;
    bool enable_yolo = true;

    void loadSelf(const YAML::Node& node) override {
        yolo_backend = node["yolo_backend"].as<std::string>("onnxruntime");
        yolo_model_path = node["yolo_model_path"].as<std::string>("");
        enable_yolo = node["enable_yolo"].as<bool>(true);
    }
};

} // namespace laser_aim::modules::config
