#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <wust_vl/video/icamera.hpp>

#include <chrono>
#include <limits>
#include <string>
#include <vector>

namespace laser_aim {

enum class TeamColor : int { RED = 0, BLUE = 1, UNKNOWN = -1 };

enum class TargetClass : int { LASER_MODULE = 0, ARMOR = 1, UNKNOWN = 2 };

enum class LockStage : int { STAGE_1 = 0, STAGE_2 = 1, STAGE_3 = 2 };
enum class RouteMode : int { CLASSIC_ONLY = 0, CLASSIC_RECAPTURE = 1, FUSION = 2 };

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

    ColorEvidence color;

    bool topology_ok { false };
    bool pnp_ok { false };
    double laser_reproj_error { std::numeric_limits<double>::infinity() };
    double armor_reproj_error { std::numeric_limits<double>::infinity() };

    cv::Mat rvec;
    cv::Mat tvec;
    cv::Point2f aim_center_px { 0.0F, 0.0F };
    cv::Point2f geometric_center_px { 0.0F, 0.0F };
    double fused_score { 0.0 };
    bool recovered_from_backup { false };

    bool is_enemy_laser { false };
};

struct TrackState {
    bool valid { false };
    Eigen::Vector3d pos { Eigen::Vector3d::Zero() };
    Eigen::Vector3d vel { Eigen::Vector3d::Zero() };
    Eigen::Vector3d predicted_pos { Eigen::Vector3d::Zero() };
    double enemy_prob_avg { 0.0 };
    double laser_prob_avg { 0.0 };
    int continuous_confirm_frames { 0 };
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
    std::string reason;
};

struct AimOutput {
    bool valid { false };
    bool fire_enable { false };
    cv::Point2f aim_center_px { 0.0F, 0.0F };
    cv::Point2f predicted_aim_center_px { 0.0F, 0.0F };
    cv::Point2f backup_aim_center_px { 0.0F, 0.0F };
    bool has_backup { false };
    double yaw_cmd { 0.0 };
    double pitch_cmd { 0.0 };
    LockStage lock_stage { LockStage::STAGE_1 };
    RouteMode route_mode { RouteMode::FUSION };
    TrackState track;
    GateReport gate;
};

struct PipelineDebugFrame {
    FrameContext frame;
    TeamPolicy policy;
    std::vector<Candidate> candidates;
    TrackState track;
    AimOutput output;
};

} // namespace laser_aim
