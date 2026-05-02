#include "laser_aim/modules/perception/laser_pipeline.hpp"

#include <wust_vl/common/utils/logger.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace laser_aim::modules::perception {

namespace {

constexpr const char* kLogNode = "laser.pipeline";

struct LightBar {
    cv::RotatedRect rect;
    std::array<cv::Point2f, 4> corners;
    cv::Point2f center;
    float long_side { 0.0F };
    float short_side { 0.0F };
    float long_angle_deg { 0.0F };
    double contour_area { 0.0 };
};

std::vector<cv::Point2f> rectPoints(const cv::RotatedRect& rr) {
    std::array<cv::Point2f, 4> pts;
    rr.points(pts.data());
    return std::vector<cv::Point2f>(pts.begin(), pts.end());
}

float normalizeAngle90(float deg) {
    float out = deg;
    while (out > 90.0F) {
        out -= 180.0F;
    }
    while (out <= -90.0F) {
        out += 180.0F;
    }
    return out;
}

float angleDiffAbs(float a_deg, float b_deg) {
    const float d = std::abs(normalizeAngle90(a_deg - b_deg));
    return std::min(d, 180.0F - d);
}

double clamp01(double x) {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

cv::Rect roiFromRect2f(const cv::Rect2f& r, const cv::Size& size) {
    const int x = std::max(0, static_cast<int>(std::floor(r.x)));
    const int y = std::max(0, static_cast<int>(std::floor(r.y)));
    const int w = std::min(size.width - x, static_cast<int>(std::ceil(r.width)));
    const int h = std::min(size.height - y, static_cast<int>(std::ceil(r.height)));
    if (w <= 1 || h <= 1) {
        return {};
    }
    return cv::Rect(x, y, w, h);
}

cv::Rect2f bboxFromPoints(const std::vector<cv::Point2f>& pts) {
    if (pts.empty()) {
        return {};
    }
    float min_x = pts.front().x;
    float max_x = pts.front().x;
    float min_y = pts.front().y;
    float max_y = pts.front().y;
    for (const auto& p : pts) {
        min_x = std::min(min_x, p.x);
        max_x = std::max(max_x, p.x);
        min_y = std::min(min_y, p.y);
        max_y = std::max(max_y, p.y);
    }
    return cv::Rect2f(min_x, min_y, std::max(0.0F, max_x - min_x), std::max(0.0F, max_y - min_y));
}

LightBar buildLightBar(const cv::RotatedRect& rr, double contour_area) {
    LightBar bar;
    bar.rect = rr;
    rr.points(bar.corners.data());
    bar.center = rr.center;
    bar.long_side = std::max(rr.size.width, rr.size.height);
    bar.short_side = std::min(rr.size.width, rr.size.height);
    bar.contour_area = contour_area;

    float angle = rr.angle;
    if (rr.size.width < rr.size.height) {
        angle += 90.0F;
    }
    bar.long_angle_deg = normalizeAngle90(angle);
    return bar;
}

bool splitBarSemanticCorners(
    const LightBar& bar,
    const cv::Point2f& left_to_right_unit,
    float inner_sign,
    cv::Point2f* upper_inner,
    cv::Point2f* lower_inner,
    cv::Point2f* upper_outer,
    cv::Point2f* lower_outer
) {
    if (upper_inner == nullptr || lower_inner == nullptr || upper_outer == nullptr || lower_outer == nullptr) {
        return false;
    }

    struct ScoredPt {
        cv::Point2f pt;
        float signed_score { 0.0F };
    };

    std::array<ScoredPt, 4> scored;
    for (int i = 0; i < 4; ++i) {
        const auto& p = bar.corners[static_cast<std::size_t>(i)];
        const float s = (p.x - bar.center.x) * left_to_right_unit.x + (p.y - bar.center.y) * left_to_right_unit.y;
        scored[static_cast<std::size_t>(i)] = { p, s * inner_sign };
    }

    std::sort(
        scored.begin(),
        scored.end(),
        [](const ScoredPt& a, const ScoredPt& b) { return a.signed_score > b.signed_score; }
    );

    std::array<cv::Point2f, 2> inners = { scored[0].pt, scored[1].pt };
    std::array<cv::Point2f, 2> outers = { scored[2].pt, scored[3].pt };

    std::sort(
        inners.begin(),
        inners.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; }
    );
    std::sort(
        outers.begin(),
        outers.end(),
        [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; }
    );

    *upper_inner = inners[0];
    *lower_inner = inners[1];
    *upper_outer = outers[0];
    *lower_outer = outers[1];
    return true;
}

bool buildSemantic8FromBarPair(
    const LightBar& left_bar,
    const LightBar& right_bar,
    std::vector<cv::Point2f>* keypoints
) {
    if (keypoints == nullptr) {
        return false;
    }

    const cv::Point2f lr = right_bar.center - left_bar.center;
    const float lr_norm = std::sqrt(lr.x * lr.x + lr.y * lr.y);
    if (lr_norm < 1e-3F) {
        return false;
    }
    const cv::Point2f lr_unit(lr.x / lr_norm, lr.y / lr_norm);

    cv::Point2f ul_in_l;
    cv::Point2f ll_in_l;
    cv::Point2f ul_out_l;
    cv::Point2f ll_out_l;
    cv::Point2f ur_in_r;
    cv::Point2f lr_in_r;
    cv::Point2f ur_out_r;
    cv::Point2f lr_out_r;

    const bool left_ok = splitBarSemanticCorners(
        left_bar,
        lr_unit,
        +1.0F,
        &ul_in_l,
        &ll_in_l,
        &ul_out_l,
        &ll_out_l
    );
    const bool right_ok = splitBarSemanticCorners(
        right_bar,
        lr_unit,
        -1.0F,
        &ur_in_r,
        &lr_in_r,
        &ur_out_r,
        &lr_out_r
    );

    if (!left_ok || !right_ok) {
        return false;
    }

    const cv::Point2f kp0 = ul_in_l * 0.75F + ll_in_l * 0.25F;
    const cv::Point2f kp2 = ul_in_l * 0.25F + ll_in_l * 0.75F;
    const cv::Point2f kp4 = ul_out_l * 0.75F + ll_out_l * 0.25F;
    const cv::Point2f kp6 = ul_out_l * 0.25F + ll_out_l * 0.75F;

    const cv::Point2f kp1 = ur_in_r * 0.75F + lr_in_r * 0.25F;
    const cv::Point2f kp3 = ur_in_r * 0.25F + lr_in_r * 0.75F;
    const cv::Point2f kp5 = ur_out_r * 0.75F + lr_out_r * 0.25F;
    const cv::Point2f kp7 = ur_out_r * 0.25F + lr_out_r * 0.75F;

    keypoints->assign(static_cast<std::size_t>(kLaserSemanticKpCount), cv::Point2f(0.0F, 0.0F));
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)] = kp0;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)] = kp1;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)] = kp2;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)] = kp3;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP4_UPPER_LEFT_OUTER)] = kp4;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP5_UPPER_RIGHT_OUTER)] = kp5;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP6_LOWER_LEFT_OUTER)] = kp6;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP7_LOWER_RIGHT_OUTER)] = kp7;
    return true;
}

std::optional<cv::Point2f> centerFromSemantic8(const std::vector<cv::Point2f>& keypoints) {
    if (keypoints.size() < static_cast<std::size_t>(kLaserSemanticKpCount)) {
        return std::nullopt;
    }

    const cv::Point2f ct = (
        keypoints[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)]
        + keypoints[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)]
    ) * 0.5F;
    const cv::Point2f cb = (
        keypoints[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)]
        + keypoints[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)]
    ) * 0.5F;
    return (ct + cb) * 0.5F;
}

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

    const std::vector<cv::Point3f> laser_quad = {
        { -0.080F, -0.045F, 0.0F },
        { 0.080F, -0.045F, 0.0F },
        { 0.080F, 0.045F, 0.0F },
        { -0.080F, 0.045F, 0.0F },
    };
    const std::vector<cv::Point3f> laser_oct = {
        { -0.060F, -0.0225F, 0.0F }, // kp0 upper-left-inner center
        { 0.060F, -0.0225F, 0.0F },  // kp1 upper-right-inner center
        { -0.060F, 0.0225F, 0.0F },  // kp2 lower-left-inner center
        { 0.060F, 0.0225F, 0.0F },   // kp3 lower-right-inner center
        { -0.080F, -0.0225F, 0.0F }, // kp4 upper-left-outer center
        { 0.080F, -0.0225F, 0.0F },  // kp5 upper-right-outer center
        { -0.080F, 0.0225F, 0.0F },  // kp6 lower-left-outer center
        { 0.080F, 0.0225F, 0.0F },   // kp7 lower-right-outer center
    };
    const std::vector<cv::Point3f> laser_center = { { 0.0F, 0.0F, 0.0F } };

    const std::vector<cv::Point3f> armor_quad = {
        { -0.115F, -0.027F, 0.0F },
        { 0.115F, -0.027F, 0.0F },
        { 0.115F, 0.027F, 0.0F },
        { -0.115F, 0.027F, 0.0F },
    };
    const std::vector<cv::Point3f> armor_oct = {
        { -0.085F, -0.0135F, 0.0F },
        { 0.085F, -0.0135F, 0.0F },
        { -0.085F, 0.0135F, 0.0F },
        { 0.085F, 0.0135F, 0.0F },
        { -0.115F, -0.0135F, 0.0F },
        { 0.115F, -0.0135F, 0.0F },
        { -0.115F, 0.0135F, 0.0F },
        { 0.115F, 0.0135F, 0.0F },
    };

    laser_pnp_solver_epnp_.setObjectPoints("laser_quad", laser_quad);
    laser_pnp_solver_epnp_.setObjectPoints("laser_oct", laser_oct);
    laser_pnp_solver_epnp_.setObjectPoints("laser_center", laser_center);
    laser_pnp_solver_ippe_.setObjectPoints("laser_quad", laser_quad);
    laser_pnp_solver_ippe_.setObjectPoints("laser_oct", laser_oct);
    laser_pnp_solver_ippe_.setObjectPoints("laser_center", laser_center);

    armor_pnp_solver_epnp_.setObjectPoints("armor_quad", armor_quad);
    armor_pnp_solver_epnp_.setObjectPoints("armor_oct", armor_oct);
    armor_pnp_solver_ippe_.setObjectPoints("armor_quad", armor_quad);
    armor_pnp_solver_ippe_.setObjectPoints("armor_oct", armor_oct);

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
    const cv::Mat rectified_img = rectifyFrame(frame.img_frame.src_img);
    cv::Mat enhanced_bgr;
    cv::Mat enhanced_gray;
    preprocessFrame(rectified_img, &enhanced_bgr, &enhanced_gray);
    auto candidates = runDualCandidateGeneration(enhanced_bgr);

    classifyColor(
        enhanced_bgr,
        candidates,
        policy.team_color,
        policy.enemy_color,
        policy.unknown_color_safe_mode
    );
    classifyLaserModule(candidates);
    refineAndSolvePose(enhanced_gray, candidates);

    for (auto& c : candidates) {
        c.is_enemy_laser = c.raw_class == TargetClass::LASER_MODULE && c.pnp_ok
            && c.color.confident && c.color.decided == policy.enemy_color;
        c.fused_score = candidateScore(c, stage_);
    }

    auto selected = selectBest(candidates);
    auto backup = selected.has_value()
        ? selectBackup(candidates, selected->candidate_id)
        : std::optional<Candidate> {};

    if (backup.has_value()) {
        backup_candidate_ = backup;
    }

    const double recapture_trigger = std::max(0.05, cfg_hub_->pipeline().enemy_prob_min_param.get() * 0.7);
    if (!selected.has_value() || selected->fused_score < recapture_trigger) {
        low_conf_frames_ += 1;
        recover_frames_ = 0;
    } else {
        low_conf_frames_ = 0;
        recover_frames_ += 1;
    }

    if (low_conf_frames_ >= std::max(1, cfg_hub_->pipeline().low_conf_recap_frames_param.get())) {
        recapture_mode_ = true;
    } else if (recover_frames_ >= std::max(1, cfg_hub_->pipeline().recover_frames_param.get())) {
        recapture_mode_ = false;
    }

    if (use_classic_only) {
        output.route_mode = RouteMode::CLASSIC_ONLY;
    } else if (recapture_mode_) {
        output.route_mode = RouteMode::CLASSIC_RECAPTURE;
    } else {
        output.route_mode = RouteMode::FUSION;
    }

    if (recapture_mode_ && (!selected.has_value() || selected->fused_score < recapture_trigger)
        && backup_candidate_.has_value()) {
        selected = backup_candidate_;
        selected->recovered_from_backup = true;
    }

    const TrackState track = updateTrack(selected, frame, policy);
    const LockStage stage = updateLockStage(policy, track, selected);
    const cv::Point2f predicted_aim_px = predictAimCenterPx(track, selected);
    const auto [yaw_cmd, pitch_cmd] = computeControlCmd(
        predicted_aim_px,
        enhanced_bgr.size(),
        frame.img_frame.timestamp
    );

    GateReport gate = evaluateGate(policy, track, selected, stage, predicted_aim_px);

    if (selected.has_value()) {
        output.valid = true;
        output.aim_center_px = selected->aim_center_px;
        if (backup.has_value()) {
            output.backup_aim_center_px = backup->aim_center_px;
            output.has_backup = true;
        }
    }
    output.predicted_aim_center_px = predicted_aim_px;
    output.yaw_cmd = yaw_cmd;
    output.pitch_cmd = pitch_cmd;
    output.fire_enable = gate.fire_enable;
    output.lock_stage = stage;
    output.track = track;
    output.gate = gate;

    if (dbg_out != nullptr) {
        dbg_out->frame = frame;
        dbg_out->policy = policy;
        dbg_out->candidates = std::move(candidates);
        dbg_out->track = track;
        dbg_out->output = output;
    }
    return output;
}

std::vector<Candidate> LaserPipeline::runDualCandidateGeneration(const cv::Mat& src_img) {
    const auto classic_candidates = runClassicCandidateGeneration(src_img);
    if (cfg_hub_ != nullptr && cfg_hub_->pipeline().use_classic_only_param.get()) {
        return classic_candidates;
    }
    if (recapture_mode_) {
        return classic_candidates;
    }
    const auto yolo_candidates = runYoloCandidateGeneration(src_img);
    return fuseCandidates(yolo_candidates, classic_candidates);
}

cv::Mat LaserPipeline::rectifyFrame(const cv::Mat& src_img) {
    if (src_img.empty()) {
        return src_img;
    }
    ensureUndistortMaps(src_img.size());

    const bool enable_undistort = cfg_hub_ != nullptr && cfg_hub_->pipeline().enable_undistort_param.get();
    if (!enable_undistort || !undistort_ready_ || undistort_map1_.empty() || undistort_map2_.empty()) {
        return src_img.clone();
    }

    cv::Mat rectified;
    cv::remap(src_img, rectified, undistort_map1_, undistort_map2_, cv::INTER_LINEAR, cv::BORDER_CONSTANT);

    cv::Rect roi = undistort_valid_roi_;
    if (roi.width <= 0 || roi.height <= 0) {
        roi = cv::Rect(0, 0, rectified.cols, rectified.rows);
    }

    const int border_px = (cfg_hub_ != nullptr) ? std::max(0, cfg_hub_->pipeline().undistort_border_px_param.get()) : 0;
    roi.x = std::min(std::max(0, roi.x + border_px), rectified.cols);
    roi.y = std::min(std::max(0, roi.y + border_px), rectified.rows);
    roi.width = std::max(0, std::min(rectified.cols - roi.x, roi.width - 2 * border_px));
    roi.height = std::max(0, std::min(rectified.rows - roi.y, roi.height - 2 * border_px));

    if (roi.width > 0 && roi.height > 0) {
        if (roi.x > 0) {
            cv::rectangle(rectified, cv::Rect(0, 0, roi.x, rectified.rows), cv::Scalar::all(0), cv::FILLED);
        }
        if (roi.y > 0) {
            cv::rectangle(rectified, cv::Rect(0, 0, rectified.cols, roi.y), cv::Scalar::all(0), cv::FILLED);
        }
        const int right = roi.x + roi.width;
        if (right < rectified.cols) {
            cv::rectangle(
                rectified,
                cv::Rect(right, 0, rectified.cols - right, rectified.rows),
                cv::Scalar::all(0),
                cv::FILLED
            );
        }
        const int bottom = roi.y + roi.height;
        if (bottom < rectified.rows) {
            cv::rectangle(
                rectified,
                cv::Rect(0, bottom, rectified.cols, rectified.rows - bottom),
                cv::Scalar::all(0),
                cv::FILLED
            );
        }
    }
    return rectified;
}

void LaserPipeline::ensureUndistortMaps(const cv::Size& frame_size) {
    if (cfg_hub_ == nullptr || frame_size.width <= 0 || frame_size.height <= 0) {
        return;
    }
    if (raw_camera_k_.empty() || raw_camera_d_.empty()) {
        undistort_ready_ = false;
        camera_k_ = raw_camera_k_.clone();
        camera_d_ = raw_camera_d_.clone();
        return;
    }

    const bool enable_undistort = cfg_hub_->pipeline().enable_undistort_param.get();
    if (!enable_undistort) {
        undistort_ready_ = false;
        camera_k_ = raw_camera_k_.clone();
        camera_d_ = raw_camera_d_.clone();
        return;
    }

    if (undistort_ready_ && undistort_map_size_ == frame_size) {
        return;
    }

    const double alpha = std::clamp(cfg_hub_->pipeline().undistort_alpha_param.get(), 0.0, 1.0);
    cv::Mat new_k = cv::getOptimalNewCameraMatrix(
        raw_camera_k_,
        raw_camera_d_,
        frame_size,
        alpha,
        frame_size,
        &undistort_valid_roi_,
        true
    );

    cv::initUndistortRectifyMap(
        raw_camera_k_,
        raw_camera_d_,
        cv::Mat(),
        new_k,
        frame_size,
        CV_16SC2,
        undistort_map1_,
        undistort_map2_
    );

    if (undistort_map1_.empty() || undistort_map2_.empty()) {
        undistort_ready_ = false;
        camera_k_ = raw_camera_k_.clone();
        camera_d_ = raw_camera_d_.clone();
        WUST_ERROR(kLogNode) << "failed to build undistort maps, fallback to raw image";
        return;
    }

    undistort_map_size_ = frame_size;
    undistort_ready_ = true;
    camera_k_ = new_k;
    const int dist_n = std::max(5, static_cast<int>(raw_camera_d_.total()));
    camera_d_ = cv::Mat::zeros(1, dist_n, CV_64F);

    WUST_MAIN(kLogNode) << "undistort map built, size=" << frame_size.width << "x" << frame_size.height
                        << ", alpha=" << alpha;
}

std::vector<Candidate> LaserPipeline::runYoloCandidateGeneration(const cv::Mat& src_img) const {
    (void)src_img;
    // TODO: integrate yolo11-pose by wust_vl::ml_net. Keep the interface for A/B dual-route fusion.
    return {};
}

std::vector<Candidate> LaserPipeline::fuseCandidates(
    const std::vector<Candidate>& yolo_candidates,
    const std::vector<Candidate>& classic_candidates
) {
    std::vector<Candidate> fused;
    fused.reserve(yolo_candidates.size() + classic_candidates.size());

    const auto& p_cfg = cfg_hub_->pipeline();
    const double iou_min = p_cfg.fusion_iou_min_param.get();

    std::vector<bool> classic_used(classic_candidates.size(), false);
    int next_id = 0;

    for (const auto& y : yolo_candidates) {
        int best_match = -1;
        double best_iou = 0.0;
        for (std::size_t i = 0; i < classic_candidates.size(); ++i) {
            const double iou = computeIoU(y.bbox, classic_candidates[i].bbox);
            if (iou > best_iou) {
                best_iou = iou;
                best_match = static_cast<int>(i);
            }
        }

        Candidate c = y;
        c.candidate_id = next_id++;
        c.from_yolo = true;
        if (best_match >= 0 && best_iou >= iou_min) {
            const auto& cls = classic_candidates[static_cast<std::size_t>(best_match)];
            classic_used[static_cast<std::size_t>(best_match)] = true;
            c.from_classic = true;
            if (cls.keypoints.size() > c.keypoints.size()) {
                c.keypoints = cls.keypoints;
            }
            c.bbox = cls.bbox;
            c.detector_confidence = static_cast<float>(std::max<double>(c.detector_confidence, cls.detector_confidence));
        }
        fused.push_back(std::move(c));
    }

    for (std::size_t i = 0; i < classic_candidates.size(); ++i) {
        if (!classic_used[i]) {
            Candidate c = classic_candidates[i];
            c.candidate_id = next_id++;
            fused.push_back(std::move(c));
        }
    }

    if (fused.empty()) {
        return fused;
    }

    for (auto& c : fused) {
        const double model_term = static_cast<double>(c.detector_confidence);
        const double geom_term = c.keypoints.size() >= static_cast<std::size_t>(kLaserSemanticKpCount) ? 1.0 : 0.45;
        double temporal_term = 0.0;
        if (last_selected_center_.has_value()) {
            const cv::Point2f center = centerFromSemantic8(c.keypoints).value_or(
                cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F)
            );
            const double dist = cv::norm(center - *last_selected_center_);
            temporal_term = clamp01(1.0 - dist / 220.0);
        }
        c.fused_score = p_cfg.fusion_w_model_param.get() * model_term
            + p_cfg.fusion_w_geom_param.get() * geom_term
            + p_cfg.fusion_w_temporal_param.get() * temporal_term;
    }

    return fused;
}

std::vector<Candidate> LaserPipeline::runClassicCandidateGeneration(const cv::Mat& src_img) {
    std::vector<Candidate> out;

    cv::Mat gray;
    if (src_img.channels() == 3) {
        cv::cvtColor(src_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src_img;
    }

    cv::Mat bin;
    cv::threshold(gray, bin, 210, 255, cv::THRESH_BINARY);
    const cv::Mat morph_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, morph_kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> bars;
    bars.reserve(contours.size());

    for (const auto& cnt : contours) {
        const double area = cv::contourArea(cnt);
        if (area < 12.0) {
            continue;
        }

        const cv::RotatedRect rr = cv::minAreaRect(cnt);
        if (rr.size.width < 2.0F || rr.size.height < 2.0F) {
            continue;
        }

        const LightBar bar = buildLightBar(rr, area);
        const float ratio = bar.long_side / std::max(bar.short_side, 1.0F);
        if (bar.long_side < 6.0F || bar.short_side < 1.5F) {
            continue;
        }
        if (ratio < 1.6F || ratio > 20.0F) {
            continue;
        }
        bars.push_back(bar);
    }

    int next_id = 0;
    for (std::size_t i = 0; i < bars.size(); ++i) {
        for (std::size_t j = i + 1; j < bars.size(); ++j) {
            const LightBar* left = &bars[i];
            const LightBar* right = &bars[j];
            if (left->center.x > right->center.x) {
                std::swap(left, right);
            }

            const float avg_len = (left->long_side + right->long_side) * 0.5F;
            if (avg_len <= 1.0F) {
                continue;
            }

            const float dx = right->center.x - left->center.x;
            if (dx <= 1.0F) {
                continue;
            }
            const float dy = std::abs(right->center.y - left->center.y);
            const float gap_ratio = dx / avg_len;
            const float y_misaligned_ratio = dy / avg_len;
            const float len_ratio = std::max(left->long_side, right->long_side)
                / std::max(1.0F, std::min(left->long_side, right->long_side));
            const float angle_diff = angleDiffAbs(left->long_angle_deg, right->long_angle_deg);

            if (gap_ratio < 0.5F || gap_ratio > 8.0F) {
                continue;
            }
            if (y_misaligned_ratio > 1.0F) {
                continue;
            }
            if (len_ratio > 1.8F) {
                continue;
            }
            if (angle_diff > 20.0F) {
                continue;
            }

            std::vector<cv::Point2f> semantic_kps;
            if (!buildSemantic8FromBarPair(*left, *right, &semantic_kps)) {
                continue;
            }

            const double area_score = clamp01((left->contour_area + right->contour_area) / 1500.0);
            const double align_score = clamp01(1.0 - static_cast<double>(y_misaligned_ratio));
            const double angle_score = clamp01(1.0 - static_cast<double>(angle_diff) / 20.0);
            const double shape_score = clamp01(1.0 - static_cast<double>(len_ratio - 1.0) / 0.8);
            const float conf = static_cast<float>(clamp01(
                0.15 + 0.35 * area_score + 0.2 * align_score + 0.15 * angle_score + 0.15 * shape_score
            ));

            Candidate c;
            c.candidate_id = next_id++;
            c.raw_class = TargetClass::UNKNOWN;
            c.keypoints = std::move(semantic_kps);
            c.bbox = bboxFromPoints(c.keypoints);
            c.detector_confidence = conf;
            c.from_classic = true;
            out.push_back(std::move(c));
        }
    }

    // Fallback: keep a minimal candidate path when pairing fails.
    if (out.empty()) {
        for (const auto& bar : bars) {
            Candidate c;
            c.candidate_id = next_id++;
            c.raw_class = TargetClass::UNKNOWN;
            c.bbox = bar.rect.boundingRect2f();
            c.keypoints = rectPoints(bar.rect);
            c.detector_confidence = static_cast<float>(clamp01(bar.contour_area / 800.0));
            c.from_classic = true;
            out.push_back(std::move(c));
        }
    }

    return out;
}

void LaserPipeline::preprocessFrame(
    const cv::Mat& src_img,
    cv::Mat* enhanced_bgr,
    cv::Mat* enhanced_gray
) const {
    if (enhanced_bgr == nullptr || enhanced_gray == nullptr) {
        return;
    }

    cv::Mat bgr;
    if (src_img.channels() == 1) {
        cv::cvtColor(src_img, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = src_img.clone();
    }

    std::vector<cv::Mat> bgr_channels;
    cv::split(bgr, bgr_channels);
    for (auto& ch : bgr_channels) {
        cv::normalize(ch, ch, 0, 255, cv::NORM_MINMAX);
    }
    cv::merge(bgr_channels, *enhanced_bgr);

    cv::Mat gray;
    cv::cvtColor(*enhanced_bgr, gray, cv::COLOR_BGR2GRAY);
    auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, *enhanced_gray);
}

void LaserPipeline::refineKeypointsSubpix(const cv::Mat& gray_img, Candidate* candidate) const {
    if (candidate == nullptr || candidate->keypoints.empty() || gray_img.empty()) {
        return;
    }

    std::vector<cv::Point2f> refined = candidate->keypoints;
    cv::cornerSubPix(
        gray_img,
        refined,
        cv::Size(3, 3),
        cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03)
    );

    double max_shift = 0.0;
    for (std::size_t i = 0; i < refined.size(); ++i) {
        max_shift = std::max(max_shift, static_cast<double>(cv::norm(refined[i] - candidate->keypoints[i])));
    }
    if (max_shift > 4.0) {
        candidate->detector_confidence *= 0.8F;
        return;
    }
    candidate->keypoints = std::move(refined);
}

void LaserPipeline::classifyColor(
    const cv::Mat& src_img,
    std::vector<Candidate>& candidates,
    TeamColor team_color,
    TeamColor enemy_color,
    bool strict_unknown
) const {
    if (candidates.empty()) {
        return;
    }

    cv::Mat hsv;
    cv::cvtColor(src_img, hsv, cv::COLOR_BGR2HSV);
    cv::Mat lab;
    cv::cvtColor(src_img, lab, cv::COLOR_BGR2Lab);

    const double min_score = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().enemy_color_score_min_param.get()
        : 0.05;
    const double min_s = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().color_min_s_param.get()
        : 80.0;
    const double min_v = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().color_min_v_param.get()
        : 80.0;

    for (auto& c : candidates) {
        const cv::Rect roi = roiFromRect2f(c.bbox, src_img.size());

        if (roi.width <= 1 || roi.height <= 1) {
            continue;
        }

        const cv::Mat patch_hsv = hsv(roi);
        const cv::Mat patch_lab = lab(roi);
        cv::Mat red_mask_1;
        cv::Mat red_mask_2;
        cv::Mat blue_mask;
        cv::inRange(patch_hsv, cv::Scalar(0, min_s, min_v), cv::Scalar(10, 255, 255), red_mask_1);
        cv::inRange(patch_hsv, cv::Scalar(160, min_s, min_v), cv::Scalar(180, 255, 255), red_mask_2);
        cv::inRange(patch_hsv, cv::Scalar(95, min_s, min_v), cv::Scalar(140, 255, 255), blue_mask);

        std::vector<cv::Mat> lab_channels;
        cv::split(patch_lab, lab_channels);
        cv::Mat red_lab_mask;
        cv::Mat blue_lab_mask;
        cv::threshold(lab_channels[1], red_lab_mask, 145, 255, cv::THRESH_BINARY);
        cv::threshold(lab_channels[2], blue_lab_mask, 135, 255, cv::THRESH_BINARY_INV);

        const cv::Mat red_mask = (red_mask_1 | red_mask_2) & red_lab_mask;
        const cv::Mat blue_mask_final = blue_mask & blue_lab_mask;
        const double area = static_cast<double>(roi.area());
        const double red_pixels = static_cast<double>(cv::countNonZero(red_mask));
        const double blue_pixels = static_cast<double>(cv::countNonZero(blue_mask_final));
        const double p_red = clamp01(red_pixels / std::max(area, 1.0));
        const double p_blue = clamp01(blue_pixels / std::max(area, 1.0));

        c.color.p_red = p_red;
        c.color.p_blue = p_blue;

        if (enemy_color == TeamColor::RED) {
            c.color.color_score = p_red - p_blue;
        } else if (enemy_color == TeamColor::BLUE) {
            c.color.color_score = p_blue - p_red;
        } else {
            c.color.color_score = 0.0;
        }

        if (team_color == TeamColor::UNKNOWN && strict_unknown) {
            c.color.decided = TeamColor::UNKNOWN;
            c.color.confident = false;
        } else if (std::abs(c.color.color_score) < min_score) {
            c.color.decided = TeamColor::UNKNOWN;
            c.color.confident = false;
        } else {
            c.color.decided = (c.color.color_score > 0.0) ? enemy_color : team_color;
            c.color.confident = true;
        }
    }
}

void LaserPipeline::classifyLaserModule(std::vector<Candidate>& candidates) {
    if (cfg_hub_ == nullptr) {
        return;
    }

    auto& p_cfg = cfg_hub_->pipeline();
    const double margin = p_cfg.armor_laser_error_margin_param.get();

    for (auto& c : candidates) {
        c.topology_ok = c.keypoints.size() >= 4;
        if (!c.topology_ok) {
            continue;
        }
        const bool use_semantic8 = c.keypoints.size() >= static_cast<std::size_t>(kLaserSemanticKpCount);
        const char* laser_model = use_semantic8 ? "laser_oct" : "laser_quad";
        const char* armor_model = use_semantic8 ? "armor_oct" : "armor_quad";

        cv::Mat rvec_laser_best;
        cv::Mat tvec_laser_best;
        cv::Mat rvec_armor_best;
        cv::Mat tvec_armor_best;
        c.laser_reproj_error = std::numeric_limits<double>::infinity();
        c.armor_reproj_error = std::numeric_limits<double>::infinity();

        auto trySolveBest = [
                               &c,
                               &use_semantic8,
                               &laser_model,
                               &armor_model,
                               this
                           ](
                               bool is_laser,
                               cv::Mat* best_rvec,
                               cv::Mat* best_tvec,
                               double* best_err
                           ) {
            auto& epnp_solver = is_laser ? laser_pnp_solver_epnp_ : armor_pnp_solver_epnp_;
            auto& ippe_solver = is_laser ? laser_pnp_solver_ippe_ : armor_pnp_solver_ippe_;
            const char* model_name = is_laser ? laser_model : armor_model;

            cv::Mat rvec_e;
            cv::Mat tvec_e;
            if (epnp_solver.solvePnP(c.keypoints, rvec_e, tvec_e, model_name, camera_k_, camera_d_)) {
                const double err = epnp_solver.calculateReprojectionError(
                    c.keypoints,
                    rvec_e,
                    tvec_e,
                    model_name,
                    camera_k_,
                    camera_d_
                );
                if (err < *best_err) {
                    *best_err = err;
                    *best_rvec = rvec_e;
                    *best_tvec = tvec_e;
                }
            }

            if (use_semantic8) {
                cv::Mat rvec_i;
                cv::Mat tvec_i;
                if (ippe_solver.solvePnP(c.keypoints, rvec_i, tvec_i, model_name, camera_k_, camera_d_)) {
                    const double err = ippe_solver.calculateReprojectionError(
                        c.keypoints,
                        rvec_i,
                        tvec_i,
                        model_name,
                        camera_k_,
                        camera_d_
                    );
                    if (err < *best_err) {
                        *best_err = err;
                        *best_rvec = rvec_i;
                        *best_tvec = tvec_i;
                    }
                }
            }
        };

        trySolveBest(true, &rvec_laser_best, &tvec_laser_best, &c.laser_reproj_error);
        trySolveBest(false, &rvec_armor_best, &tvec_armor_best, &c.armor_reproj_error);

        const bool laser_ok = std::isfinite(c.laser_reproj_error);
        const bool armor_ok = std::isfinite(c.armor_reproj_error);

        if (laser_ok && (!armor_ok || c.laser_reproj_error + margin < c.armor_reproj_error)) {
            c.raw_class = TargetClass::LASER_MODULE;
            c.pnp_ok = true;
            c.rvec = rvec_laser_best;
            c.tvec = tvec_laser_best;
        } else if (armor_ok) {
            c.raw_class = TargetClass::ARMOR;
            c.pnp_ok = true;
            c.rvec = rvec_armor_best;
            c.tvec = tvec_armor_best;
        } else {
            c.raw_class = TargetClass::UNKNOWN;
            c.pnp_ok = false;
        }
    }
}

void LaserPipeline::refineAndSolvePose(const cv::Mat& gray_img, std::vector<Candidate>& candidates) {
    for (auto& c : candidates) {
        refineKeypointsSubpix(gray_img, &c);
        const auto semantic_center = centerFromSemantic8(c.keypoints);
        c.geometric_center_px = semantic_center.value_or(
            cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F)
        );
        if (!c.pnp_ok) {
            if (semantic_center.has_value()) {
                c.aim_center_px = *semantic_center;
            } else {
                c.aim_center_px = cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F);
            }
            continue;
        }
        c.aim_center_px = solveAimCenter(c);
    }
}

std::optional<Candidate> LaserPipeline::selectBest(const std::vector<Candidate>& candidates) const {
    double best_score = -1e9;
    std::optional<Candidate> best;

    for (const auto& c : candidates) {
        const double score = candidateScore(c, stage_);

        if (score > best_score) {
            best_score = score;
            best = c;
            best->fused_score = score;
        }
    }
    if (best.has_value()) {
        best->fused_score = best_score;
    }
    return best;
}

std::optional<Candidate> LaserPipeline::selectBackup(
    const std::vector<Candidate>& candidates,
    int primary_id
) const {
    double best_score = -1e9;
    std::optional<Candidate> best;
    for (const auto& c : candidates) {
        if (c.candidate_id == primary_id) {
            continue;
        }
        const double score = candidateScore(c, stage_);
        if (score > best_score) {
            best_score = score;
            best = c;
            best->fused_score = score;
        }
    }
    return best;
}

TrackState LaserPipeline::updateTrack(
    const std::optional<Candidate>& selected,
    const FrameContext& frame,
    const TeamPolicy& policy
) {
    TrackState out;
    const double predict_horizon_s = std::max(0.0, cfg_hub_->pipeline().control_latency_ms_param.get() * 1e-3);

    if (!selected.has_value()) {
        track_filter_.predictTo(frame.img_frame.timestamp);
        out.valid = track_filter_.valid();
        out.pos = track_filter_.pos();
        out.vel = track_filter_.vel();
        out.predicted_pos = out.pos + out.vel * predict_horizon_s;
        out.enemy_prob_avg = track_filter_.enemyProbAvg();
        out.laser_prob_avg = track_filter_.laserProbAvg();
        out.continuous_confirm_frames = track_filter_.continuousConfirmFrames();
        return out;
    }

    const auto& c = *selected;
    const double enemy_prob = (policy.enemy_color == TeamColor::UNKNOWN)
        ? 0.0
        : clamp01((c.color.color_score + 1.0) * 0.5);
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
        gate.reason = "team color unknown: track only";
    } else if (!gate.enemy_conf_ok) {
        gate.reason = "enemy confidence low";
    } else if (!gate.laser_conf_ok) {
        gate.reason = "laser confidence low";
    } else if (!gate.track_stable_ok) {
        gate.reason = "ekf not stable";
    } else if (!gate.pnp_error_ok) {
        gate.reason = "pnp error too large";
    } else if (!gate.center_in_window_ok) {
        gate.reason = "center not in effective window";
    } else if (!gate.predicted_window_ok) {
        gate.reason = "predicted center not in effective window";
    }

    return gate;
}

cv::Point2f LaserPipeline::solveAimCenter(const Candidate& c) {
    const auto semantic_center = centerFromSemantic8(c.keypoints);
    if (!c.pnp_ok || c.rvec.empty() || c.tvec.empty()) {
        if (semantic_center.has_value()) {
            return *semantic_center;
        }
        return cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F);
    }

    const auto projected = laser_pnp_solver_epnp_.getImagePoints(
        c.rvec,
        c.tvec,
        "laser_center",
        camera_k_,
        camera_d_
    );

    if (projected.empty()) {
        if (semantic_center.has_value()) {
            return *semantic_center;
        }
        return cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F);
    }
    return projected.front();
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
    const double err_x = (static_cast<double>(predicted_aim_px.x) - cx) / std::max(1.0, static_cast<double>(img_size.width));
    const double err_y = (static_cast<double>(predicted_aim_px.y) - cy) / std::max(1.0, static_cast<double>(img_size.height));

    const double yaw_cmd = yaw_pid_.update(0.0, err_x, dt);
    const double pitch_cmd = pitch_pid_.update(0.0, err_y, dt);
    return { yaw_cmd, pitch_cmd };
}

double LaserPipeline::candidateScore(const Candidate& c, LockStage stage) const {
    double cls_bonus = 0.0;
    if (c.raw_class == TargetClass::LASER_MODULE) {
        cls_bonus = 1.1;
    } else if (c.raw_class == TargetClass::ARMOR) {
        cls_bonus = -0.6;
    }

    const double pnp_bonus = c.pnp_ok ? std::max(0.0, 1.0 - c.laser_reproj_error / 6.0) : -0.5;
    const double enemy_bonus = c.is_enemy_laser ? 1.2 : 0.0;

    double color_weight = 1.0;
    if (stage == LockStage::STAGE_3 && cfg_hub_ != nullptr) {
        color_weight = cfg_hub_->pipeline().stage3_color_weight_param.get();
    }

    return c.fused_score + static_cast<double>(c.detector_confidence) + color_weight * c.color.color_score + cls_bonus
        + pnp_bonus + enemy_bonus;
}

double LaserPipeline::computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width, b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);

    const float inter_w = std::max(0.0F, x2 - x1);
    const float inter_h = std::max(0.0F, y2 - y1);
    const float inter = inter_w * inter_h;
    const float union_area = a.width * a.height + b.width * b.height - inter;
    if (union_area <= 1e-6F) {
        return 0.0;
    }
    return static_cast<double>(inter / union_area);
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

void drawPipelineDebug(cv::Mat& debug_img, const PipelineDebugFrame& dbg) {
    for (const auto& c : dbg.candidates) {
        cv::Scalar box_color(200, 200, 200);
        if (c.raw_class == TargetClass::LASER_MODULE) {
            box_color = cv::Scalar(50, 255, 50);
        } else if (c.raw_class == TargetClass::ARMOR) {
            box_color = cv::Scalar(50, 200, 255);
        }

        cv::rectangle(debug_img, c.bbox, box_color, 2);
        cv::circle(debug_img, c.aim_center_px, 4, cv::Scalar(0, 0, 255), -1);

        for (std::size_t i = 0; i < c.keypoints.size(); ++i) {
            const cv::Scalar kp_color = (i < static_cast<std::size_t>(kLaserSemanticKpCount))
                ? cv::Scalar(255, 120, 20)
                : cv::Scalar(180, 180, 180);
            cv::circle(debug_img, c.keypoints[i], 3, kp_color, -1);
            cv::putText(
                debug_img,
                "k" + std::to_string(i),
                c.keypoints[i] + cv::Point2f(2.0F, -2.0F),
                cv::FONT_HERSHEY_SIMPLEX,
                0.35,
                kp_color,
                1
            );
        }

        const std::string txt = toString(c.raw_class) + " conf=" + std::to_string(c.detector_confidence);
        cv::putText(
            debug_img,
            txt,
            cv::Point(static_cast<int>(c.bbox.x), static_cast<int>(c.bbox.y) - 4),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            box_color,
            1
        );
    }

    const cv::Point center(debug_img.cols / 2, debug_img.rows / 2);
    cv::circle(debug_img, center, 4, cv::Scalar(255, 255, 255), 2);

    const auto& out = dbg.output;
    const auto gate = out.gate;
    const std::string line0 = "team=" + toString(dbg.policy.team_color) + " enemy=" + toString(dbg.policy.enemy_color)
        + " safe_track_only=" + std::string(dbg.policy.isSafeTrackOnly() ? "true" : "false");
    const std::string line1 = "stage=" + toString(out.lock_stage) + " fire=" + std::string(out.fire_enable ? "true" : "false")
        + " route=" + toString(out.route_mode) + " reason=" + gate.reason;
    const std::string line2 = "enemy_prob=" + std::to_string(out.track.enemy_prob_avg)
        + " laser_prob=" + std::to_string(out.track.laser_prob_avg)
        + " confirm=" + std::to_string(out.track.continuous_confirm_frames);
    const std::string line3 = "yaw_cmd=" + std::to_string(out.yaw_cmd) + " pitch_cmd=" + std::to_string(out.pitch_cmd)
        + " pred_ok=" + std::string(gate.predicted_window_ok ? "true" : "false")
        + " track_ok=" + std::string(gate.track_stable_ok ? "true" : "false");

    cv::putText(debug_img, line0, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1);
    cv::putText(debug_img, line1, cv::Point(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 0), 1);
    cv::putText(debug_img, line2, cv::Point(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 255), 1);
    cv::putText(debug_img, line3, cv::Point(10, 95), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 255, 180), 1);

    cv::circle(debug_img, out.predicted_aim_center_px, 4, cv::Scalar(255, 0, 180), 2);
    if (out.has_backup) {
        cv::circle(debug_img, out.backup_aim_center_px, 4, cv::Scalar(180, 180, 255), 1);
    }
}

} // namespace laser_aim::modules::perception
