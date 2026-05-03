#include "laser_aim/modules/perception/laser_pipeline.hpp"

#include <wust_vl/common/utils/logger.hpp>

#include <opencv2/calib3d.hpp>

#include <algorithm>
#include <cmath>

namespace laser_aim::modules::perception {

LaserPipeline::LaserPipeline():
    laser_pnp_solver_epnp_(cv::SOLVEPNP_EPNP),
    laser_pnp_solver_ippe_(cv::SOLVEPNP_IPPE),
    armor_pnp_solver_epnp_(cv::SOLVEPNP_EPNP),
    armor_pnp_solver_ippe_(cv::SOLVEPNP_IPPE) {}

bool LaserPipeline::init(config::ParameterHub* cfg_hub, const cv::Mat& camera_k, const cv::Mat& camera_d) {
    cfg_hub_ = cfg_hub;
    raw_camera_k_ = camera_k.clone();
    raw_camera_d_ = camera_d.clone();
    camera_k_ = raw_camera_k_.clone();
    camera_d_ = raw_camera_d_.clone();

    // Laser template is calibrated from module drawings (unit: meter):
    // visual feature width = 50 mm, total top-to-bottom extent = 72 mm, middle gap = 20 mm.
    constexpr float kLaserHalfWidth = 0.025F;      // 50 / 2
    constexpr float kLaserInnerHalfHeight = 0.010F; // 20 / 2
    constexpr float kLaserOuterHalfHeight = 0.036F; // 72 / 2

    const std::vector<cv::Point3f> laser_quad = {
        { -kLaserHalfWidth, -kLaserOuterHalfHeight, 0.0F },
        { kLaserHalfWidth, -kLaserOuterHalfHeight, 0.0F },
        { kLaserHalfWidth, kLaserOuterHalfHeight, 0.0F },
        { -kLaserHalfWidth, kLaserOuterHalfHeight, 0.0F },
    };
    const std::vector<cv::Point3f> laser_oct = {
        { -kLaserHalfWidth, -kLaserInnerHalfHeight, 0.0F },
        { kLaserHalfWidth, -kLaserInnerHalfHeight, 0.0F },
        { -kLaserHalfWidth, kLaserInnerHalfHeight, 0.0F },
        { kLaserHalfWidth, kLaserInnerHalfHeight, 0.0F },
        { -kLaserHalfWidth, -kLaserOuterHalfHeight, 0.0F },
        { kLaserHalfWidth, -kLaserOuterHalfHeight, 0.0F },
        { -kLaserHalfWidth, kLaserOuterHalfHeight, 0.0F },
        { kLaserHalfWidth, kLaserOuterHalfHeight, 0.0F },
    };
    const std::vector<cv::Point3f> laser_center = { { 0.0F, 0.0F, 0.0F } };

    // Armor templates are calibrated from drawings:
    // small armor: 135 mm (width) x 125 mm (height), big armor: 230 mm x 127 mm.
    constexpr float kArmorSmallHalfWidth = 0.0675F;
    constexpr float kArmorSmallOuterHalfHeight = 0.0625F;
    constexpr float kArmorSmallInnerHalfHeight = 0.03125F;
    constexpr float kArmorBigHalfWidth = 0.115F;
    constexpr float kArmorBigOuterHalfHeight = 0.0635F;
    constexpr float kArmorBigInnerHalfHeight = 0.03175F;

    const std::vector<cv::Point3f> armor_small_quad = {
        { -kArmorSmallHalfWidth, -kArmorSmallOuterHalfHeight, 0.0F },
        { kArmorSmallHalfWidth, -kArmorSmallOuterHalfHeight, 0.0F },
        { kArmorSmallHalfWidth, kArmorSmallOuterHalfHeight, 0.0F },
        { -kArmorSmallHalfWidth, kArmorSmallOuterHalfHeight, 0.0F },
    };
    const std::vector<cv::Point3f> armor_small_oct = {
        { -kArmorSmallHalfWidth, -kArmorSmallInnerHalfHeight, 0.0F },
        { kArmorSmallHalfWidth, -kArmorSmallInnerHalfHeight, 0.0F },
        { -kArmorSmallHalfWidth, kArmorSmallInnerHalfHeight, 0.0F },
        { kArmorSmallHalfWidth, kArmorSmallInnerHalfHeight, 0.0F },
        { -kArmorSmallHalfWidth, -kArmorSmallOuterHalfHeight, 0.0F },
        { kArmorSmallHalfWidth, -kArmorSmallOuterHalfHeight, 0.0F },
        { -kArmorSmallHalfWidth, kArmorSmallOuterHalfHeight, 0.0F },
        { kArmorSmallHalfWidth, kArmorSmallOuterHalfHeight, 0.0F },
    };
    const std::vector<cv::Point3f> armor_big_quad = {
        { -kArmorBigHalfWidth, -kArmorBigOuterHalfHeight, 0.0F },
        { kArmorBigHalfWidth, -kArmorBigOuterHalfHeight, 0.0F },
        { kArmorBigHalfWidth, kArmorBigOuterHalfHeight, 0.0F },
        { -kArmorBigHalfWidth, kArmorBigOuterHalfHeight, 0.0F },
    };
    const std::vector<cv::Point3f> armor_big_oct = {
        { -kArmorBigHalfWidth, -kArmorBigInnerHalfHeight, 0.0F },
        { kArmorBigHalfWidth, -kArmorBigInnerHalfHeight, 0.0F },
        { -kArmorBigHalfWidth, kArmorBigInnerHalfHeight, 0.0F },
        { kArmorBigHalfWidth, kArmorBigInnerHalfHeight, 0.0F },
        { -kArmorBigHalfWidth, -kArmorBigOuterHalfHeight, 0.0F },
        { kArmorBigHalfWidth, -kArmorBigOuterHalfHeight, 0.0F },
        { -kArmorBigHalfWidth, kArmorBigOuterHalfHeight, 0.0F },
        { kArmorBigHalfWidth, kArmorBigOuterHalfHeight, 0.0F },
    };
    const std::vector<cv::Point3f> armor_center = { { 0.0F, 0.0F, 0.0F } };

    laser_pnp_solver_epnp_.setObjectPoints("laser_quad", laser_quad);
    laser_pnp_solver_epnp_.setObjectPoints("laser_oct", laser_oct);
    laser_pnp_solver_epnp_.setObjectPoints("laser_center", laser_center);
    laser_pnp_solver_ippe_.setObjectPoints("laser_quad", laser_quad);
    laser_pnp_solver_ippe_.setObjectPoints("laser_oct", laser_oct);
    laser_pnp_solver_ippe_.setObjectPoints("laser_center", laser_center);

    armor_pnp_solver_epnp_.setObjectPoints("armor_small_quad", armor_small_quad);
    armor_pnp_solver_epnp_.setObjectPoints("armor_small_oct", armor_small_oct);
    armor_pnp_solver_epnp_.setObjectPoints("armor_big_quad", armor_big_quad);
    armor_pnp_solver_epnp_.setObjectPoints("armor_big_oct", armor_big_oct);
    armor_pnp_solver_epnp_.setObjectPoints("armor_center", armor_center);
    armor_pnp_solver_ippe_.setObjectPoints("armor_small_quad", armor_small_quad);
    armor_pnp_solver_ippe_.setObjectPoints("armor_small_oct", armor_small_oct);
    armor_pnp_solver_ippe_.setObjectPoints("armor_big_quad", armor_big_quad);
    armor_pnp_solver_ippe_.setObjectPoints("armor_big_oct", armor_big_oct);
    armor_pnp_solver_ippe_.setObjectPoints("armor_center", armor_center);

    if (cfg_hub_ != nullptr) {
        auto& p_cfg = cfg_hub_->pipeline();
        yaw_pid_.setGains(
            p_cfg.pid_yaw_kp_param.get(),
            p_cfg.pid_yaw_ki_param.get(),
            p_cfg.pid_yaw_kd_param.get()
        );
        pitch_pid_.setGains(
            p_cfg.pid_pitch_kp_param.get(),
            p_cfg.pid_pitch_ki_param.get(),
            p_cfg.pid_pitch_kd_param.get()
        );
        const double deriv_tau = std::max(0.0, p_cfg.pid_derivative_tau_param.get());
        yaw_pid_.setDerivativeFilterTau(deriv_tau);
        pitch_pid_.setDerivativeFilterTau(deriv_tau);
        const double anti_windup_gain = std::max(0.0, p_cfg.pid_anti_windup_gain_param.get());
        yaw_pid_.setAntiWindupGain(anti_windup_gain);
        pitch_pid_.setAntiWindupGain(anti_windup_gain);
        const double lim = std::max(1e-6, p_cfg.pid_output_limit_param.get());
        yaw_pid_.setOutputLimits(-lim, lim);
        pitch_pid_.setOutputLimits(-lim, lim);
        yaw_pid_.setIntegratorLimit(lim);
        pitch_pid_.setIntegratorLimit(lim);
    }
    return true;
}

AimOutput LaserPipeline::process(const FrameContext& frame, PipelineDebugFrame* dbg_out) {
    AimOutput output;
    if (cfg_hub_ == nullptr || frame.img_frame.src_img.empty()) {
        return output;
    }

    const TeamPolicy policy = cfg_hub_->teamPolicy();
    const bool use_classic_only = cfg_hub_->pipeline().use_classic_only_param.get();
    const LockStage stage_for_scoring = policy.hasStageRef() ? policy.stageFromRefOr(stage_) : stage_;
    stage_ = stage_for_scoring;
    const cv::Mat rectified_img = rectifyFrame(frame.img_frame.src_img);
    cv::Mat enhanced_bgr;
    cv::Mat enhanced_gray;
    preprocessFrame(rectified_img, &enhanced_bgr, &enhanced_gray);
    auto candidates = runDualCandidateGeneration(rectified_img, policy.enemy_color);

    refineCandidatesKeypoints(enhanced_gray, candidates);
    classifyColor(
        rectified_img,
        candidates,
        policy.team_color,
        policy.enemy_color,
        policy.unknown_color_safe_mode
    );
    classifyLaserModule(candidates);
    solveCandidateAimCenters(candidates);
    const PnpCalibMetrics pnp_calib = updatePnpCalibMetrics(candidates);

    for (auto& c : candidates) {
        c.is_enemy_laser = c.raw_class == TargetClass::LASER_MODULE && c.pnp_ok
            && c.color.confident && c.color.decided == policy.enemy_color;
        c.fused_score = candidateScore(c, stage_for_scoring);
    }

    auto selected = selectBest(candidates);
    auto frame_backup = selected.has_value()
        ? selectBackup(candidates, selected->candidate_id)
        : std::optional<Candidate> {};

    if (backup_switch_cooldown_left_ > 0) {
        backup_switch_cooldown_left_ -= 1;
    }

    if (frame_backup.has_value()) {
        backup_candidate_ = frame_backup;
        backup_candidate_age_frames_ = 0;
    } else if (backup_candidate_.has_value()) {
        backup_candidate_age_frames_ += 1;
    } else {
        backup_candidate_age_frames_ = 0;
    }

    const auto& p_cfg = cfg_hub_->pipeline();
    const double recapture_scale = std::max(0.1, p_cfg.recapture_trigger_scale_param.get());
    const double recapture_trigger = std::max(0.05, p_cfg.enemy_prob_min_param.get() * recapture_scale);
    const auto isReliableCandidate = [recapture_trigger](const Candidate& c) {
        return c.fused_score >= recapture_trigger && c.raw_class == TargetClass::LASER_MODULE && c.pnp_ok;
    };

    bool selected_reliable = selected.has_value() && isReliableCandidate(*selected);
    if (!selected_reliable) {
        low_conf_frames_ += 1;
        recover_frames_ = 0;
    } else {
        low_conf_frames_ = 0;
        recover_frames_ += 1;
    }

    const int low_conf_n = std::max(1, p_cfg.low_conf_recap_frames_param.get());
    const int recover_m = std::max(1, p_cfg.recover_frames_param.get());
    if (low_conf_frames_ >= low_conf_n) {
        recapture_mode_ = true;
    }
    if (recapture_mode_ && recover_frames_ >= recover_m) {
        recapture_mode_ = false;
        low_conf_frames_ = 0;
    }

    if (use_classic_only) {
        output.route_mode = RouteMode::CLASSIC_ONLY;
    } else if (recapture_mode_) {
        output.route_mode = RouteMode::CLASSIC_RECAPTURE;
    } else {
        output.route_mode = RouteMode::FUSION;
    }

    if (recapture_mode_ && !selected_reliable && backup_candidate_.has_value()) {
        const int backup_max_age = std::max(0, p_cfg.backup_max_age_frames_param.get());
        if (backup_candidate_age_frames_ <= backup_max_age) {
            Candidate backup_try = *backup_candidate_;
            backup_try.fused_score = candidateScore(backup_try, stage_for_scoring);

            const double backup_min_score = recapture_trigger * std::max(0.0, p_cfg.backup_min_score_ratio_param.get());
            const double switch_margin = std::max(0.0, p_cfg.backup_switch_score_margin_param.get());
            const bool score_ok = backup_try.fused_score >= backup_min_score;
            const bool cooldown_ok = backup_switch_cooldown_left_ <= 0;
            const bool better_than_primary = !selected.has_value()
                || (backup_try.fused_score >= selected->fused_score + switch_margin);

            bool dist_ok = true;
            const double max_dist_px = std::max(0.0, p_cfg.backup_switch_max_dist_px_param.get());
            if (max_dist_px > 1e-6) {
                if (selected.has_value()) {
                    dist_ok = cv::norm(backup_try.aim_center_px - selected->aim_center_px) <= max_dist_px;
                } else if (last_selected_center_.has_value()) {
                    dist_ok = cv::norm(backup_try.aim_center_px - *last_selected_center_) <= max_dist_px;
                }
            }

            if (score_ok && cooldown_ok && better_than_primary && dist_ok) {
                selected = backup_try;
                selected->recovered_from_backup = true;
                selected_reliable = isReliableCandidate(*selected);
                backup_switch_cooldown_left_ = std::max(0, p_cfg.backup_switch_cooldown_frames_param.get());
                low_conf_frames_ = 0;
                recover_frames_ = selected_reliable ? std::max(1, recover_frames_ + 1) : 0;
            }
        }
    }
    smoothSelectedColor(&selected, policy);

    const TrackState track = updateTrack(selected, frame, policy);
    const LockStage stage = updateLockStage(policy, track, selected);
    const cv::Point2f predicted_aim_px = predictAimCenterPx(track, selected);
    const bool control_active = selected.has_value() || track.valid;
    double yaw_cmd = 0.0;
    double pitch_cmd = 0.0;
    if (control_active) {
        const auto cmd = computeControlCmd(
            predicted_aim_px,
            enhanced_bgr.size(),
            frame.img_frame.timestamp
        );
        yaw_cmd = cmd.first;
        pitch_cmd = cmd.second;
    } else {
        yaw_pid_.reset();
        pitch_pid_.reset();
        last_ctrl_ts_ = {};
        last_yaw_cmd_ = 0.0;
        last_pitch_cmd_ = 0.0;
    }

    GateReport gate = evaluateGate(policy, track, selected, stage, predicted_aim_px);

    if (selected.has_value()) {
        output.valid = true;
        output.aim_center_px = selected->aim_center_px;
        std::optional<Candidate> backup_for_output = frame_backup;
        const int backup_max_age = std::max(0, p_cfg.backup_max_age_frames_param.get());
        if (!backup_for_output.has_value() && backup_candidate_.has_value() && backup_candidate_age_frames_ <= backup_max_age) {
            backup_for_output = backup_candidate_;
        }
        if (backup_for_output.has_value()) {
            output.backup_aim_center_px = backup_for_output->aim_center_px;
            output.has_backup = true;
        }
    }
    output.predicted_aim_center_px = predicted_aim_px;
    output.yaw_cmd = yaw_cmd;
    output.pitch_cmd = pitch_cmd;
    output.fire_enable = gate.fire_enable;
    output.timestamp_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(frame.img_frame.timestamp.time_since_epoch()).count()
    );
    output.gate_reason = toString(gate.code);
    output.lock_stage = stage;
    output.track = track;
    output.gate = gate;

    if (dbg_out != nullptr) {
        dbg_out->frame = frame;
        dbg_out->policy = policy;
        dbg_out->candidates = std::move(candidates);
        dbg_out->pnp_calib = pnp_calib;
        dbg_out->track = track;
        dbg_out->output = output;
    }
    return output;
}

} // namespace laser_aim::modules::perception
