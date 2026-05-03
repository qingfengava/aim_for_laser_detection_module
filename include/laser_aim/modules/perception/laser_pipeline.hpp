#pragma once

#include "laser_aim/common/types.hpp"
#include "laser_aim/modules/config/parameter_hub.hpp"
#include "laser_aim/modules/tracking/laser_track_filter.hpp"

#include <wust_vl/algorithm/control/pid.hpp>
#include <wust_vl/algorithm/pnp_solver.hpp>

#include <deque>
#include <optional>

namespace laser_aim::modules::perception {

class LaserPipeline {
public:
    LaserPipeline();

    bool init(config::ParameterHub* cfg_hub, const cv::Mat& camera_k, const cv::Mat& camera_d);

    AimOutput process(const FrameContext& frame, PipelineDebugFrame* dbg_out);

private:
    std::vector<Candidate> runDualCandidateGeneration(const cv::Mat& src_img, TeamColor enemy_color);
    std::vector<Candidate> runYoloCandidateGeneration(const cv::Mat& src_img) const;
    std::vector<Candidate> runClassicCandidateGeneration(const cv::Mat& src_img, TeamColor enemy_color);
    std::vector<Candidate> fuseCandidates(
        const std::vector<Candidate>& yolo_candidates,
        const std::vector<Candidate>& classic_candidates
    );
    cv::Mat rectifyFrame(const cv::Mat& src_img);
    void ensureUndistortMaps(const cv::Size& frame_size);
    void preprocessFrame(const cv::Mat& src_img, cv::Mat* enhanced_bgr, cv::Mat* enhanced_gray) const;
    void classifyColor(
        const cv::Mat& src_img,
        std::vector<Candidate>& candidates,
        TeamColor team_color,
        TeamColor enemy_color,
        bool strict_unknown
    ) const;
    void classifyLaserModule(std::vector<Candidate>& candidates);
    void refineCandidatesKeypoints(const cv::Mat& gray_img, std::vector<Candidate>& candidates) const;
    void solveCandidateAimCenters(std::vector<Candidate>& candidates);
    void refineKeypointsSubpix(const cv::Mat& gray_img, Candidate* candidate) const;
    void smoothSelectedColor(std::optional<Candidate>* selected, const TeamPolicy& policy);
    std::optional<Candidate> selectBest(const std::vector<Candidate>& candidates) const;
    std::optional<Candidate> selectBackup(
        const std::vector<Candidate>& candidates,
        int primary_id
    ) const;

    TrackState updateTrack(
        const std::optional<Candidate>& selected,
        const FrameContext& frame,
        const TeamPolicy& policy
    );

    LockStage updateLockStage(
        const TeamPolicy& policy,
        const TrackState& track,
        const std::optional<Candidate>& selected
    );

    GateReport evaluateGate(
        const TeamPolicy& policy,
        const TrackState& track,
        const std::optional<Candidate>& selected,
        LockStage stage,
        const cv::Point2f& predicted_aim_px
    ) const;

    std::optional<cv::Point2f> solveAimCenter(const Candidate& c);
    cv::Point2f predictAimCenterPx(
        const TrackState& track,
        const std::optional<Candidate>& selected
    ) const;
    std::pair<double, double> computeControlCmd(
        const cv::Point2f& predicted_aim_px,
        const cv::Size& img_size,
        std::chrono::steady_clock::time_point now_ts
    );
    PnpCalibMetrics updatePnpCalibMetrics(const std::vector<Candidate>& candidates);
    double candidateScore(const Candidate& c, LockStage stage) const;

    static double currentStageScale(LockStage stage, const config::PipelineConfig& cfg);
    static double computeIoU(const cv::Rect2f& a, const cv::Rect2f& b);

private:
    config::ParameterHub* cfg_hub_ { nullptr };
    cv::Mat raw_camera_k_;
    cv::Mat raw_camera_d_;
    cv::Mat camera_k_;
    cv::Mat camera_d_;
    cv::Mat undistort_map1_;
    cv::Mat undistort_map2_;
    cv::Rect undistort_valid_roi_;
    cv::Size undistort_map_size_ { 0, 0 };
    bool undistort_ready_ { false };

    wust_vl::algorithm::PnPSolver laser_pnp_solver_epnp_;
    wust_vl::algorithm::PnPSolver laser_pnp_solver_ippe_;
    wust_vl::algorithm::PnPSolver armor_pnp_solver_epnp_;
    wust_vl::algorithm::PnPSolver armor_pnp_solver_ippe_;

    tracking::LaserTrackFilter track_filter_;
    wust_vl::algorithm::control::PID<double> yaw_pid_;
    wust_vl::algorithm::control::PID<double> pitch_pid_;

    LockStage stage_ { LockStage::STAGE_1 };
    int lock_hit_count_ { 0 };
    int low_conf_frames_ { 0 };
    int recover_frames_ { 0 };
    bool recapture_mode_ { false };
    std::optional<Candidate> backup_candidate_;
    int backup_candidate_age_frames_ { 0 };
    int backup_switch_cooldown_left_ { 0 };
    std::optional<cv::Point2f> last_selected_center_;
    std::chrono::steady_clock::time_point last_ctrl_ts_ {};
    double last_yaw_cmd_ { 0.0 };
    double last_pitch_cmd_ { 0.0 };
    struct PnpCalibAccumulator {
        std::uint64_t total_valid_samples { 0 };
        std::uint64_t total_pred_laser { 0 };
        std::uint64_t total_pred_armor { 0 };
        std::uint64_t total_pred_unknown { 0 };
        std::uint64_t total_ambiguous { 0 };
        std::uint64_t total_expected_laser_samples { 0 };
        std::uint64_t total_laser_as_armor { 0 };
    };
    PnpCalibAccumulator pnp_calib_acc_;
    int pnp_calib_expected_class_last_ { -999 };
    std::deque<double> color_score_window_;
    std::deque<int> color_valid_window_;
    TeamColor color_smooth_enemy_ref_ { TeamColor::UNKNOWN };
    TeamColor color_stable_decision_ { TeamColor::UNKNOWN };
};

void drawPipelineDebug(cv::Mat& debug_img, const PipelineDebugFrame& dbg);

} // namespace laser_aim::modules::perception
