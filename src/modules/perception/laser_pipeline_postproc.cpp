#include "laser_aim/modules/perception/laser_pipeline.hpp"
#include "pipeline_internals.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace laser_aim::modules::perception {

using internal::clamp01;
using internal::centerFromSemantic8;

namespace {

struct StageWindowModel {
    cv::Point2f center { 0.0F, 0.0F };
    cv::Point2f axis_unit { 0.0F, 1.0F };
    float half_height { 1.0F };
    bool valid { false };
};

double pointDistanceAlongAxis(
    const cv::Point2f& point,
    const cv::Point2f& center,
    const cv::Point2f& axis_unit
) {
    const cv::Point2f d = point - center;
    return std::abs(static_cast<double>(d.x * axis_unit.x + d.y * axis_unit.y));
}

StageWindowModel buildStageWindowModel(const Candidate& c) {
    StageWindowModel model;
    model.center = c.geometric_center_px;
    model.axis_unit = cv::Point2f(0.0F, 1.0F);
    float full_height = std::max(1.0F, c.bbox.height);

    if (c.keypoints.size() >= static_cast<std::size_t>(kLaserSemanticKpCount)) {
        const cv::Point2f ct = (
            c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)]
            + c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)]
        ) * 0.5F;
        const cv::Point2f cb = (
            c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)]
            + c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)]
        ) * 0.5F;

        model.center = (ct + cb) * 0.5F;
        const cv::Point2f axis = ct - cb;
        const float axis_norm = std::sqrt(axis.x * axis.x + axis.y * axis.y);
        if (axis_norm > 1e-3F) {
            model.axis_unit = cv::Point2f(axis.x / axis_norm, axis.y / axis_norm);
            full_height = axis_norm;
        }
    }

    model.half_height = std::max(1.0F, 0.5F * full_height);
    model.valid = true;
    return model;
}

Eigen::Vector3d fallbackPosFromImage(const Candidate& c, const cv::Size& size) {
    if (size.width <= 0 || size.height <= 0) {
        return Eigen::Vector3d::Zero();
    }
    const double nx = (static_cast<double>(c.aim_center_px.x) / static_cast<double>(size.width) - 0.5) * 2.0;
    const double ny = (static_cast<double>(c.aim_center_px.y) / static_cast<double>(size.height) - 0.5) * 2.0;
    return Eigen::Vector3d(nx, ny, 1.0);
}

} // namespace

TrackState LaserPipeline::updateTrack(
    const std::optional<Candidate>& selected,
    const FrameContext& frame,
    const TeamPolicy& policy
) {
    TrackState out;
    auto& p_cfg = cfg_hub_->pipeline();
    tracking::LaserTrackFilter::Tuning tuning;
    tuning.q_pos = p_cfg.ekf_q_pos_param.get();
    tuning.q_vel = p_cfg.ekf_q_vel_param.get();
    tuning.r_pos = p_cfg.ekf_r_pos_param.get();
    tuning.dt_min_s = p_cfg.ekf_dt_min_s_param.get();
    tuning.dt_max_s = p_cfg.ekf_dt_max_s_param.get();
    tuning.max_obs_jump_m = p_cfg.ekf_max_obs_jump_m_param.get();
    tuning.prob_smooth = p_cfg.ekf_prob_smooth_param.get();
    track_filter_.setTuning(tuning);

    const double horizon_scale = std::max(0.0, p_cfg.ekf_predict_horizon_scale_param.get());
    const double base_horizon_s = std::max(0.0, p_cfg.control_latency_ms_param.get() * 1e-3) * horizon_scale;
    double horizon_min_s = std::max(0.0, p_cfg.ekf_predict_horizon_min_s_param.get());
    double horizon_max_s = std::max(0.0, p_cfg.ekf_predict_horizon_max_s_param.get());
    if (horizon_min_s > horizon_max_s) {
        std::swap(horizon_min_s, horizon_max_s);
    }
    const double predict_horizon_s = std::clamp(base_horizon_s, horizon_min_s, horizon_max_s);

    if (!selected.has_value()) {
        track_filter_.predictTo(frame.img_frame.timestamp);
        out.valid = track_filter_.valid();
        out.pos = track_filter_.pos();
        out.vel = track_filter_.vel();
        out.predicted_pos = out.pos + out.vel * predict_horizon_s;
        out.enemy_prob_avg = track_filter_.enemyProbAvg();
        out.laser_prob_avg = track_filter_.laserProbAvg();
        out.continuous_confirm_frames = track_filter_.continuousConfirmFrames();
        out.innovation_norm_m = track_filter_.lastInnovationNorm();
        out.innovation_rejected = track_filter_.lastUpdateRejected();
        return out;
    }

    const auto& c = *selected;
    double enemy_prob = 0.0;
    if (policy.enemy_color != TeamColor::UNKNOWN) {
        if (c.color.confident && c.color.decided != TeamColor::UNKNOWN) {
            enemy_prob = clamp01((c.color.color_score + 1.0) * 0.5);
        } else {
            enemy_prob = 0.5;
        }
    }
    const double laser_prob = c.raw_class == TargetClass::LASER_MODULE ? 1.0 : 0.2;

    Eigen::Vector3d pos;
    if (c.pnp_ok && !c.tvec.empty()) {
        pos = Eigen::Vector3d(c.tvec.at<double>(0), c.tvec.at<double>(1), c.tvec.at<double>(2));
    } else {
        pos = fallbackPosFromImage(c, frame.img_frame.src_img.size());
    }

    tracking::LaserTrackFilter::Observation obs;
    obs.pos = pos;
    obs.enemy_prob = enemy_prob;
    obs.laser_prob = laser_prob;
    obs.timestamp = frame.img_frame.timestamp;
    track_filter_.update(obs);

    out.valid = track_filter_.valid();
    out.pos = track_filter_.pos();
    out.vel = track_filter_.vel();
    out.predicted_pos = out.pos + out.vel * predict_horizon_s;
    out.enemy_prob_avg = track_filter_.enemyProbAvg();
    out.laser_prob_avg = track_filter_.laserProbAvg();
    out.continuous_confirm_frames = track_filter_.continuousConfirmFrames();
    out.innovation_norm_m = track_filter_.lastInnovationNorm();
    out.innovation_rejected = track_filter_.lastUpdateRejected();
    last_selected_center_ = c.aim_center_px;

    return out;
}

LockStage LaserPipeline::updateLockStage(
    const TeamPolicy& policy,
    const TrackState& track,
    const std::optional<Candidate>& selected
) {
    if (policy.hasStageRef()) {
        stage_ = policy.stageFromRefOr(stage_);
        return stage_;
    }

    if (!selected.has_value() || !track.valid) {
        lock_hit_count_ = 0;
        stage_ = LockStage::STAGE_1;
        return stage_;
    }

    auto& p_cfg = cfg_hub_->pipeline();
    const int confirm_frames = std::max(1, p_cfg.lock_confirm_frames_param.get());

    if (track.continuous_confirm_frames > 0) {
        lock_hit_count_ += 1;
    } else {
        lock_hit_count_ = 0;
    }

    if (lock_hit_count_ > confirm_frames * 2) {
        stage_ = LockStage::STAGE_3;
    } else if (lock_hit_count_ > confirm_frames) {
        stage_ = LockStage::STAGE_2;
    } else {
        stage_ = LockStage::STAGE_1;
    }
    return stage_;
}

GateReport LaserPipeline::evaluateGate(
    const TeamPolicy& policy,
    const TrackState& track,
    const std::optional<Candidate>& selected,
    LockStage stage,
    const cv::Point2f& predicted_aim_px
) const {
    GateReport gate;
    gate.code = GateReasonCode::READY;
    gate.reason = "ready";

    const auto& p_cfg = cfg_hub_->pipeline();
    gate.safe_mode_ok = !policy.isSafeTrackOnly();
    gate.enemy_conf_ok = track.enemy_prob_avg >= p_cfg.enemy_prob_min_param.get();
    gate.laser_conf_ok = track.laser_prob_avg >= p_cfg.laser_prob_min_param.get();
    const int min_confirm = std::max(1, p_cfg.ekf_min_confirm_frames_param.get());
    const double max_speed = std::max(0.1, p_cfg.ekf_max_speed_mps_param.get());
    gate.track_stable_ok = track.valid && track.continuous_confirm_frames >= min_confirm
        && track.vel.norm() <= max_speed;

    if (selected.has_value()) {
        gate.pnp_error_ok = selected->laser_reproj_error <= p_cfg.laser_reproj_error_max_param.get();

        const double scale = currentStageScale(stage, p_cfg);
        const auto win = buildStageWindowModel(*selected);
        const double min_half_h = std::max(3.0, p_cfg.gate_center_window_px_param.get() * 0.5 * scale);
        const double stage_half_h = std::max(win.half_height * scale, min_half_h);
        const double dist = pointDistanceAlongAxis(selected->aim_center_px, win.center, win.axis_unit);
        const double pred_dist = pointDistanceAlongAxis(predicted_aim_px, win.center, win.axis_unit);
        gate.center_in_window_ok = dist <= stage_half_h;
        gate.predicted_window_ok = pred_dist <= stage_half_h;
    } else {
        gate.pnp_error_ok = false;
        gate.center_in_window_ok = false;
        gate.predicted_window_ok = false;
    }

    gate.fire_enable = gate.safe_mode_ok && gate.enemy_conf_ok && gate.laser_conf_ok && gate.track_stable_ok
        && gate.pnp_error_ok
        && gate.center_in_window_ok && gate.predicted_window_ok;

    if (!gate.safe_mode_ok) {
        gate.code = GateReasonCode::TEAM_COLOR_UNKNOWN;
        gate.reason = "team color unknown: track only";
    } else if (!gate.enemy_conf_ok) {
        gate.code = GateReasonCode::ENEMY_CONF_LOW;
        gate.reason = "enemy confidence low";
    } else if (!gate.laser_conf_ok) {
        gate.code = GateReasonCode::LASER_CONF_LOW;
        gate.reason = "laser confidence low";
    } else if (!gate.track_stable_ok) {
        gate.code = GateReasonCode::EKF_UNSTABLE;
        gate.reason = "ekf not stable";
    } else if (!gate.pnp_error_ok) {
        gate.code = GateReasonCode::PNP_ERROR_TOO_LARGE;
        gate.reason = "pnp error too large";
    } else if (!gate.center_in_window_ok) {
        gate.code = GateReasonCode::CENTER_OUT_OF_WINDOW;
        gate.reason = "center not in effective window";
    } else if (!gate.predicted_window_ok) {
        gate.code = GateReasonCode::PREDICTED_OUT_OF_WINDOW;
        gate.reason = "predicted center not in effective window";
    }

    return gate;
}

cv::Point2f LaserPipeline::predictAimCenterPx(
    const TrackState& track,
    const std::optional<Candidate>& selected
) const {
    if (track.valid) {
        const auto& p = track.predicted_pos;
        if (p.z() > 1e-6) {
            const double fx = camera_k_.at<double>(0, 0);
            const double fy = camera_k_.at<double>(1, 1);
            const double cx = camera_k_.at<double>(0, 2);
            const double cy = camera_k_.at<double>(1, 2);
            const float u = static_cast<float>(fx * p.x() / p.z() + cx);
            const float v = static_cast<float>(fy * p.y() / p.z() + cy);
            return cv::Point2f(u, v);
        }
    }
    if (selected.has_value()) {
        return selected->aim_center_px;
    }
    return cv::Point2f(0.0F, 0.0F);
}

std::pair<double, double> LaserPipeline::computeControlCmd(
    const cv::Point2f& predicted_aim_px,
    const cv::Size& img_size,
    std::chrono::steady_clock::time_point now_ts
) {
    if (img_size.width <= 0 || img_size.height <= 0) {
        return { 0.0, 0.0 };
    }

    double dt = 0.01;
    if (last_ctrl_ts_.time_since_epoch().count() > 0) {
        dt = std::chrono::duration<double>(now_ts - last_ctrl_ts_).count();
        dt = std::clamp(dt, 0.001, 0.1);
    }
    last_ctrl_ts_ = now_ts;

    const double cx = static_cast<double>(img_size.width) * 0.5;
    const double cy = static_cast<double>(img_size.height) * 0.5;
    double err_x = (static_cast<double>(predicted_aim_px.x) - cx) / std::max(1.0, static_cast<double>(img_size.width));
    double err_y = (static_cast<double>(predicted_aim_px.y) - cy) / std::max(1.0, static_cast<double>(img_size.height));
    if (cfg_hub_ != nullptr) {
        const double deadband = std::max(0.0, cfg_hub_->pipeline().pid_error_deadband_param.get());
        if (std::abs(err_x) < deadband) {
            err_x = 0.0;
        }
        if (std::abs(err_y) < deadband) {
            err_y = 0.0;
        }
    }

    double yaw_cmd = yaw_pid_.update(0.0, err_x, dt);
    double pitch_cmd = pitch_pid_.update(0.0, err_y, dt);
    if (cfg_hub_ != nullptr) {
        const double slew_limit = std::max(0.0, cfg_hub_->pipeline().pid_cmd_slew_limit_param.get());
        if (slew_limit > 0.0) {
            const double max_delta = slew_limit * dt;
            yaw_cmd = std::clamp(yaw_cmd, last_yaw_cmd_ - max_delta, last_yaw_cmd_ + max_delta);
            pitch_cmd = std::clamp(pitch_cmd, last_pitch_cmd_ - max_delta, last_pitch_cmd_ + max_delta);
        }
    }
    last_yaw_cmd_ = yaw_cmd;
    last_pitch_cmd_ = pitch_cmd;
    return { yaw_cmd, pitch_cmd };
}

PnpCalibMetrics LaserPipeline::updatePnpCalibMetrics(const std::vector<Candidate>& candidates) {
    PnpCalibMetrics out;
    if (cfg_hub_ == nullptr) {
        return out;
    }

    const int expected_cls_raw = cfg_hub_->pipeline().pnp_calib_expected_class_param.get();
    if (pnp_calib_expected_class_last_ != expected_cls_raw) {
        pnp_calib_acc_ = {};
        pnp_calib_expected_class_last_ = expected_cls_raw;
    }
    const TargetClass expected_cls = (expected_cls_raw == 0) ? TargetClass::LASER_MODULE
        : (expected_cls_raw == 1)                       ? TargetClass::ARMOR
                                                         : TargetClass::UNKNOWN;

    double sum_margin_delta = 0.0;
    double sum_margin_ratio = 0.0;
    int margin_samples = 0;
    std::uint64_t frame_expected_laser = 0;
    std::uint64_t frame_laser_as_armor = 0;

    for (const auto& c : candidates) {
        const bool valid_sample = c.laser_template_valid || c.armor_template_valid;
        if (!valid_sample) {
            continue;
        }

        out.frame_valid_samples += 1;
        if (c.raw_class == TargetClass::LASER_MODULE) {
            out.frame_pred_laser += 1;
        } else if (c.raw_class == TargetClass::ARMOR) {
            out.frame_pred_armor += 1;
        } else {
            out.frame_pred_unknown += 1;
        }

        if (c.pnp_ambiguous) {
            out.frame_ambiguous += 1;
        }

        if (std::isfinite(c.pnp_margin_delta) && std::isfinite(c.pnp_margin_ratio)
            && c.laser_template_valid && c.armor_template_valid) {
            sum_margin_delta += c.pnp_margin_delta;
            sum_margin_ratio += c.pnp_margin_ratio;
            margin_samples += 1;
        }

        if (expected_cls == TargetClass::LASER_MODULE) {
            frame_expected_laser += 1;
            if (c.raw_class == TargetClass::ARMOR) {
                frame_laser_as_armor += 1;
            }
        }
    }

    if (margin_samples > 0) {
        out.frame_mean_margin_delta = sum_margin_delta / static_cast<double>(margin_samples);
        out.frame_mean_margin_ratio = sum_margin_ratio / static_cast<double>(margin_samples);
    }

    if (frame_expected_laser > 0) {
        out.frame_laser_as_armor_ratio = static_cast<double>(frame_laser_as_armor)
            / static_cast<double>(frame_expected_laser);
    }

    pnp_calib_acc_.total_valid_samples += static_cast<std::uint64_t>(out.frame_valid_samples);
    pnp_calib_acc_.total_pred_laser += static_cast<std::uint64_t>(out.frame_pred_laser);
    pnp_calib_acc_.total_pred_armor += static_cast<std::uint64_t>(out.frame_pred_armor);
    pnp_calib_acc_.total_pred_unknown += static_cast<std::uint64_t>(out.frame_pred_unknown);
    pnp_calib_acc_.total_ambiguous += static_cast<std::uint64_t>(out.frame_ambiguous);
    pnp_calib_acc_.total_expected_laser_samples += frame_expected_laser;
    pnp_calib_acc_.total_laser_as_armor += frame_laser_as_armor;

    out.total_valid_samples = pnp_calib_acc_.total_valid_samples;
    out.total_pred_laser = pnp_calib_acc_.total_pred_laser;
    out.total_pred_armor = pnp_calib_acc_.total_pred_armor;
    out.total_pred_unknown = pnp_calib_acc_.total_pred_unknown;
    out.total_ambiguous = pnp_calib_acc_.total_ambiguous;
    out.total_expected_laser_samples = pnp_calib_acc_.total_expected_laser_samples;
    out.total_laser_as_armor = pnp_calib_acc_.total_laser_as_armor;
    if (out.total_expected_laser_samples > 0) {
        out.total_laser_as_armor_ratio = static_cast<double>(out.total_laser_as_armor)
            / static_cast<double>(out.total_expected_laser_samples);
    }
    return out;
}

double LaserPipeline::currentStageScale(LockStage stage, const config::PipelineConfig& cfg) {
    switch (stage) {
        case LockStage::STAGE_1:
            return 1.0;
        case LockStage::STAGE_2:
            return cfg.stage2_scale_param.get();
        case LockStage::STAGE_3:
            return cfg.stage3_scale_param.get();
    }
    return 1.0;
}

} // namespace laser_aim::modules::perception
