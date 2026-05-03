#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <wust_vl/video/icamera.hpp>

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace laser_aim {

enum class TeamColor : int { RED = 0, BLUE = 1, UNKNOWN = -1 };

enum class TargetClass : int { LASER_MODULE = 0, ARMOR = 1, UNKNOWN = 2 };

enum class LockStage : int { STAGE_1 = 0, STAGE_2 = 1, STAGE_3 = 2 };
enum class RouteMode : int { CLASSIC_ONLY = 0, CLASSIC_RECAPTURE = 1, FUSION = 2 };
enum class DirectionHint : int { LASER_HINT = 0, ARMOR_HINT = 1, UNKNOWN = 2 };
enum class ClassicPairMode : int { TOP_BOTTOM = 0, LEFT_RIGHT = 1, UNKNOWN = 2 };
enum class GateReasonCode : int {
    READY = 0,
    TEAM_COLOR_UNKNOWN = 1,
    ENEMY_CONF_LOW = 2,
    LASER_CONF_LOW = 3,
    EKF_UNSTABLE = 4,
    PNP_ERROR_TOO_LARGE = 5,
    CENTER_OUT_OF_WINDOW = 6,
    PREDICTED_OUT_OF_WINDOW = 7,
};

enum class LaserSemanticKp : int {
    KP0_UPPER_LEFT_INNER = 0,
    KP1_UPPER_RIGHT_INNER = 1,
    KP2_LOWER_LEFT_INNER = 2,
    KP3_LOWER_RIGHT_INNER = 3,
    KP4_UPPER_LEFT_OUTER = 4,
    KP5_UPPER_RIGHT_OUTER = 5,
    KP6_LOWER_LEFT_OUTER = 6,
    KP7_LOWER_RIGHT_OUTER = 7,
};

constexpr int kLaserSemanticKpCount = 8;

inline TeamColor opposite(TeamColor c) {
    if (c == TeamColor::RED) {
        return TeamColor::BLUE;
    }
    if (c == TeamColor::BLUE) {
        return TeamColor::RED;
    }
    return TeamColor::UNKNOWN;
}

inline std::string toString(TeamColor c) {
    switch (c) {
        case TeamColor::RED:
            return "RED";
        case TeamColor::BLUE:
            return "BLUE";
        default:
            return "UNKNOWN";
    }
}

inline std::string toString(TargetClass c) {
    switch (c) {
        case TargetClass::LASER_MODULE:
            return "LASER_MODULE";
        case TargetClass::ARMOR:
            return "ARMOR";
        default:
            return "UNKNOWN";
    }
}

inline std::string toString(LockStage stage) {
    switch (stage) {
        case LockStage::STAGE_1:
            return "STAGE_1";
        case LockStage::STAGE_2:
            return "STAGE_2";
        case LockStage::STAGE_3:
            return "STAGE_3";
    }
    return "STAGE_1";
}

inline std::string toString(RouteMode mode) {
    switch (mode) {
        case RouteMode::CLASSIC_ONLY:
            return "CLASSIC_ONLY";
        case RouteMode::CLASSIC_RECAPTURE:
            return "CLASSIC_RECAPTURE";
        case RouteMode::FUSION:
            return "FUSION";
    }
    return "FUSION";
}

inline std::string toString(DirectionHint hint) {
    switch (hint) {
        case DirectionHint::LASER_HINT:
            return "LASER_HINT";
        case DirectionHint::ARMOR_HINT:
            return "ARMOR_HINT";
        default:
            return "UNKNOWN";
    }
}

inline std::string toString(ClassicPairMode mode) {
    switch (mode) {
        case ClassicPairMode::TOP_BOTTOM:
            return "TOP_BOTTOM";
        case ClassicPairMode::LEFT_RIGHT:
            return "LEFT_RIGHT";
        default:
            return "UNKNOWN";
    }
}

inline std::string toString(GateReasonCode code) {
    switch (code) {
        case GateReasonCode::READY:
            return "READY";
        case GateReasonCode::TEAM_COLOR_UNKNOWN:
            return "TEAM_COLOR_UNKNOWN";
        case GateReasonCode::ENEMY_CONF_LOW:
            return "ENEMY_CONF_LOW";
        case GateReasonCode::LASER_CONF_LOW:
            return "LASER_CONF_LOW";
        case GateReasonCode::EKF_UNSTABLE:
            return "EKF_UNSTABLE";
        case GateReasonCode::PNP_ERROR_TOO_LARGE:
            return "PNP_ERROR_TOO_LARGE";
        case GateReasonCode::CENTER_OUT_OF_WINDOW:
            return "CENTER_OUT_OF_WINDOW";
        case GateReasonCode::PREDICTED_OUT_OF_WINDOW:
            return "PREDICTED_OUT_OF_WINDOW";
    }
    return "READY";
}

struct TeamPolicy {
    TeamColor team_color { TeamColor::UNKNOWN };
    TeamColor enemy_color { TeamColor::UNKNOWN };
    bool unknown_color_safe_mode { true };
    int stage_ref_raw { -1 };

    [[nodiscard]] bool isSafeTrackOnly() const {
        return unknown_color_safe_mode && enemy_color == TeamColor::UNKNOWN;
    }

    [[nodiscard]] bool hasStageRef() const {
        return stage_ref_raw >= 0 && stage_ref_raw <= 2;
    }

    [[nodiscard]] LockStage stageFromRefOr(LockStage fallback) const {
        if (!hasStageRef()) {
            return fallback;
        }
        if (stage_ref_raw == 0) {
            return LockStage::STAGE_1;
        }
        if (stage_ref_raw == 1) {
            return LockStage::STAGE_2;
        }
        return LockStage::STAGE_3;
    }
};

struct FrameContext {
    int frame_id { 0 };
    wust_vl::video::ImageFrame img_frame;
    std::chrono::steady_clock::time_point receive_ts;
};

struct ColorEvidence {
    double p_red { 0.0 };
    double p_blue { 0.0 };
    double color_score { 0.0 };
    TeamColor decided { TeamColor::UNKNOWN };
    bool confident { false };
};

struct Candidate {
    int candidate_id { -1 };
    TargetClass raw_class { TargetClass::UNKNOWN };
    cv::Rect2f bbox;
    std::vector<cv::Point2f> keypoints;
    float detector_confidence { 0.0F };

    bool from_yolo { false };
    bool from_classic { false };
    double direction_ratio { 0.0 };
    double direction_confidence { 0.0 };
    DirectionHint direction_hint { DirectionHint::UNKNOWN };
    ClassicPairMode pair_mode { ClassicPairMode::UNKNOWN };

    ColorEvidence color;

    bool topology_ok { false };
    bool pnp_ok { false };
    double laser_reproj_error { std::numeric_limits<double>::infinity() };
    double armor_reproj_error { std::numeric_limits<double>::infinity() };
    bool laser_template_valid { false };
    bool armor_template_valid { false };
    bool pnp_ambiguous { false };
    double pnp_margin_delta { 0.0 };
    double pnp_margin_ratio { 0.0 };

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Point2f aim_center_px { 0.0F, 0.0F };
    bool aim_center_from_pnp { false };
    cv::Point2f geometric_center_px { 0.0F, 0.0F };
    double geom_consistency_score { 0.0 };
    double fused_score { 0.0 };
    bool recovered_from_backup { false };

    bool is_enemy_laser { false };
};

struct PnpCalibMetrics {
    int frame_valid_samples { 0 };
    int frame_pred_laser { 0 };
    int frame_pred_armor { 0 };
    int frame_pred_unknown { 0 };
    int frame_ambiguous { 0 };
    double frame_mean_margin_delta { 0.0 };
    double frame_mean_margin_ratio { 0.0 };
    double frame_laser_as_armor_ratio { -1.0 };

    std::uint64_t total_valid_samples { 0 };
    std::uint64_t total_pred_laser { 0 };
    std::uint64_t total_pred_armor { 0 };
    std::uint64_t total_pred_unknown { 0 };
    std::uint64_t total_ambiguous { 0 };
    std::uint64_t total_expected_laser_samples { 0 };
    std::uint64_t total_laser_as_armor { 0 };
    double total_laser_as_armor_ratio { -1.0 };
};

struct TrackState {
    bool valid { false };
    Eigen::Vector3d pos { Eigen::Vector3d::Zero() };
    Eigen::Vector3d vel { Eigen::Vector3d::Zero() };
    Eigen::Vector3d predicted_pos { Eigen::Vector3d::Zero() };
    double enemy_prob_avg { 0.0 };
    double laser_prob_avg { 0.0 };
    int continuous_confirm_frames { 0 };
    double innovation_norm_m { 0.0 };
    bool innovation_rejected { false };
};

struct GateReport {
    bool enemy_conf_ok { false };
    bool laser_conf_ok { false };
    bool pnp_error_ok { false };
    bool track_stable_ok { false };
    bool center_in_window_ok { false };
    bool predicted_window_ok { false };
    bool safe_mode_ok { false };
    bool fire_enable { false };
    GateReasonCode code { GateReasonCode::READY };
    std::string reason;
};

struct AimOutput {
    bool valid { false };
    std::uint64_t timestamp_ns { 0 };
    bool fire_enable { false };
    cv::Point2f aim_center_px { 0.0F, 0.0F };
    cv::Point2f predicted_aim_center_px { 0.0F, 0.0F };
    cv::Point2f backup_aim_center_px { 0.0F, 0.0F };
    bool has_backup { false };
    double yaw_cmd { 0.0 };
    double pitch_cmd { 0.0 };
    std::string gate_reason { "READY" };
    LockStage lock_stage { LockStage::STAGE_1 };
    RouteMode route_mode { RouteMode::FUSION };
    TrackState track;
    GateReport gate;
};

struct PipelineDebugFrame {
    FrameContext frame;
    TeamPolicy policy;
    std::vector<Candidate> candidates;
    PnpCalibMetrics pnp_calib;
    TrackState track;
    AimOutput output;
};

} // namespace laser_aim
