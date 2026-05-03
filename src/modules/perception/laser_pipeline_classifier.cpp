#include "laser_aim/modules/perception/laser_pipeline.hpp"
#include "pipeline_internals.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

namespace laser_aim::modules::perception {

using internal::clamp01;
using internal::bboxFromPoints;
using internal::centerFromSemantic8;

namespace {

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

} // namespace

void LaserPipeline::refineKeypointsSubpix(const cv::Mat& gray_img, Candidate* candidate) const {
    if (candidate == nullptr || candidate->keypoints.empty() || gray_img.empty()) {
        return;
    }

    constexpr int kWinRadius = 3;
    constexpr int kRoiMargin = 6;
    constexpr double kMaxPerPointShiftPx = 2.8;
    constexpr double kMaxRmsShiftPx = 1.2;
    constexpr double kMinAcceptRatio = 0.60;
    constexpr float kFailConfScale = 0.80F;
    constexpr float kPartialConfScale = 0.92F;

    const std::vector<cv::Point2f> original = candidate->keypoints;
    cv::Rect2f roi_box = candidate->bbox;
    if (roi_box.width <= 1.0F || roi_box.height <= 1.0F) {
        roi_box = bboxFromPoints(original);
    }

    cv::Rect roi(
        static_cast<int>(std::floor(roi_box.x)) - kRoiMargin,
        static_cast<int>(std::floor(roi_box.y)) - kRoiMargin,
        static_cast<int>(std::ceil(roi_box.width)) + 2 * kRoiMargin,
        static_cast<int>(std::ceil(roi_box.height)) + 2 * kRoiMargin
    );
    roi &= cv::Rect(0, 0, gray_img.cols, gray_img.rows);

    const int border = kWinRadius + 1;
    if (roi.width <= 2 * border || roi.height <= 2 * border) {
        candidate->detector_confidence *= kFailConfScale;
        return;
    }

    std::vector<int> refine_idx;
    std::vector<cv::Point2f> roi_points;
    refine_idx.reserve(original.size());
    roi_points.reserve(original.size());

    for (std::size_t i = 0; i < original.size(); ++i) {
        const cv::Point2f local = original[i] - cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
        if (local.x >= border && local.y >= border
            && local.x < static_cast<float>(roi.width - border)
            && local.y < static_cast<float>(roi.height - border)) {
            refine_idx.push_back(static_cast<int>(i));
            roi_points.push_back(local);
        }
    }

    const int min_accept_overall = std::max(
        2,
        static_cast<int>(std::ceil(kMinAcceptRatio * static_cast<double>(original.size())))
    );
    if (static_cast<int>(refine_idx.size()) < min_accept_overall) {
        candidate->detector_confidence *= kFailConfScale;
        return;
    }

    cv::cornerSubPix(
        gray_img(roi),
        roi_points,
        cv::Size(kWinRadius, kWinRadius),
        cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03)
    );

    std::vector<cv::Point2f> refined = original;
    int accepted = 0;
    double sum_sq_shift = 0.0;
    for (std::size_t k = 0; k < roi_points.size(); ++k) {
        const int idx = refine_idx[k];
        const cv::Point2f p_global = roi_points[k] + cv::Point2f(static_cast<float>(roi.x), static_cast<float>(roi.y));
        if (!std::isfinite(p_global.x) || !std::isfinite(p_global.y)) {
            continue;
        }
        if (p_global.x < 0.0F || p_global.x >= static_cast<float>(gray_img.cols)
            || p_global.y < 0.0F || p_global.y >= static_cast<float>(gray_img.rows)) {
            continue;
        }

        const double shift = cv::norm(p_global - original[static_cast<std::size_t>(idx)]);
        if (shift > kMaxPerPointShiftPx) {
            continue;
        }
        refined[static_cast<std::size_t>(idx)] = p_global;
        accepted += 1;
        sum_sq_shift += shift * shift;
    }

    if (accepted < min_accept_overall) {
        candidate->detector_confidence *= kFailConfScale;
        return;
    }

    const double rms_shift = std::sqrt(sum_sq_shift / std::max(1, accepted));
    if (rms_shift > kMaxRmsShiftPx) {
        candidate->detector_confidence *= kFailConfScale;
        return;
    }

    if (accepted < static_cast<int>(refine_idx.size())) {
        candidate->detector_confidence *= kPartialConfScale;
    }
    candidate->keypoints = std::move(refined);
}

void LaserPipeline::refineCandidatesKeypoints(const cv::Mat& gray_img, std::vector<Candidate>& candidates) const {
    for (auto& c : candidates) {
        refineKeypointsSubpix(gray_img, &c);
    }
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
    const double min_l = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().color_lab_min_l_param.get()
        : 35.0;
    const double red_a_min = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().color_lab_red_a_min_param.get()
        : 145.0;
    const double blue_b_max = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().color_lab_blue_b_max_param.get()
        : 135.0;
    const int min_sample_pixels = (cfg_hub_ != nullptr)
        ? std::max(4, cfg_hub_->pipeline().color_sample_min_pixels_param.get())
        : 24;
    const double min_sample_ratio = (cfg_hub_ != nullptr)
        ? std::max(0.001, cfg_hub_->pipeline().color_sample_min_ratio_param.get())
        : 0.015;
    const double sample_radius_ratio = (cfg_hub_ != nullptr)
        ? std::clamp(cfg_hub_->pipeline().color_sample_radius_ratio_param.get(), 0.03, 0.35)
        : 0.10;

    for (auto& c : candidates) {
        c.color = ColorEvidence {};
        const cv::Rect roi = roiFromRect2f(c.bbox, src_img.size());

        if (roi.width <= 1 || roi.height <= 1) {
            c.color.decided = TeamColor::UNKNOWN;
            c.color.confident = false;
            continue;
        }

        const cv::Mat patch_hsv = hsv(roi);
        const cv::Mat patch_lab = lab(roi);
        std::vector<cv::Mat> lab_channels;
        std::vector<cv::Mat> hsv_channels;
        cv::split(patch_lab, lab_channels);
        cv::split(patch_hsv, hsv_channels);
        const cv::Mat& h_ch = hsv_channels[0];
        const cv::Mat& s_ch = hsv_channels[1];
        const cv::Mat& v_ch = hsv_channels[2];
        const cv::Mat& l_ch = lab_channels[0];
        const cv::Mat& a_ch = lab_channels[1];
        const cv::Mat& b_ch = lab_channels[2];

        cv::Mat sample_mask = cv::Mat::zeros(roi.size(), CV_8UC1);
        if (!c.keypoints.empty()) {
            const int radius = std::max(
                2,
                static_cast<int>(std::round(sample_radius_ratio * static_cast<double>(std::min(roi.width, roi.height))))
            );
            const int thickness = std::max(1, radius / 2);
            auto toLocal = [&roi](const cv::Point2f& p) {
                return cv::Point2f(p.x - static_cast<float>(roi.x), p.y - static_cast<float>(roi.y));
            };
            auto drawPointIfValid = [&sample_mask, &radius](const cv::Point2f& p) {
                if (p.x >= 0.0F && p.x < static_cast<float>(sample_mask.cols)
                    && p.y >= 0.0F && p.y < static_cast<float>(sample_mask.rows)) {
                    cv::circle(sample_mask, p, radius, cv::Scalar(255), cv::FILLED);
                }
            };
            for (const auto& kp : c.keypoints) {
                drawPointIfValid(toLocal(kp));
            }

            auto drawPair = [&sample_mask, &toLocal, thickness](const cv::Point2f& p1, const cv::Point2f& p2) {
                const cv::Point2f l1 = toLocal(p1);
                const cv::Point2f l2 = toLocal(p2);
                if (l1.x < 0.0F || l1.x >= static_cast<float>(sample_mask.cols) || l1.y < 0.0F
                    || l1.y >= static_cast<float>(sample_mask.rows)) {
                    return;
                }
                if (l2.x < 0.0F || l2.x >= static_cast<float>(sample_mask.cols) || l2.y < 0.0F
                    || l2.y >= static_cast<float>(sample_mask.rows)) {
                    return;
                }
                cv::line(sample_mask, l1, l2, cv::Scalar(255), thickness);
            };

            if (c.keypoints.size() >= static_cast<std::size_t>(kLaserSemanticKpCount)) {
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP4_UPPER_LEFT_OUTER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP5_UPPER_RIGHT_OUTER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP6_LOWER_LEFT_OUTER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP7_LOWER_RIGHT_OUTER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP4_UPPER_LEFT_OUTER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP5_UPPER_RIGHT_OUTER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP6_LOWER_LEFT_OUTER)]
                );
                drawPair(
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)],
                    c.keypoints[static_cast<std::size_t>(LaserSemanticKp::KP7_LOWER_RIGHT_OUTER)]
                );
            }
        }

        if (cv::countNonZero(sample_mask) <= 0) {
            cv::Mat loose_v_mask;
            cv::Mat loose_l_mask;
            cv::threshold(v_ch, loose_v_mask, std::max(35.0, min_v * 0.72), 255, cv::THRESH_BINARY);
            cv::threshold(l_ch, loose_l_mask, std::max(20.0, min_l * 0.72), 255, cv::THRESH_BINARY);
            sample_mask = loose_v_mask & loose_l_mask;
            cv::morphologyEx(
                sample_mask,
                sample_mask,
                cv::MORPH_OPEN,
                cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3))
            );
        }

        cv::Mat sat_mask;
        cv::Mat val_mask;
        cv::Mat lum_mask;
        cv::threshold(s_ch, sat_mask, min_s, 255, cv::THRESH_BINARY);
        cv::threshold(v_ch, val_mask, min_v, 255, cv::THRESH_BINARY);
        cv::threshold(l_ch, lum_mask, min_l, 255, cv::THRESH_BINARY);
        cv::Mat eligible_mask = sample_mask & sat_mask & val_mask & lum_mask;

        const int sample_pixels = cv::countNonZero(sample_mask);
        const int eligible_pixels = cv::countNonZero(eligible_mask);
        const int min_ratio_pixels = std::max(
            min_sample_pixels,
            static_cast<int>(std::ceil(min_sample_ratio * static_cast<double>(roi.area())))
        );
        if (sample_pixels < min_ratio_pixels || eligible_pixels < min_sample_pixels) {
            c.color.decided = TeamColor::UNKNOWN;
            c.color.confident = false;
            c.color.color_score = 0.0;
            continue;
        }

        cv::Mat red_h_mask1;
        cv::Mat red_h_mask2;
        cv::Mat blue_h_mask;
        cv::inRange(h_ch, cv::Scalar(0), cv::Scalar(10), red_h_mask1);
        cv::inRange(h_ch, cv::Scalar(160), cv::Scalar(180), red_h_mask2);
        cv::inRange(h_ch, cv::Scalar(95), cv::Scalar(140), blue_h_mask);
        cv::Mat red_h_mask = red_h_mask1 | red_h_mask2;

        cv::Mat red_lab_mask;
        cv::Mat blue_lab_mask;
        cv::threshold(a_ch, red_lab_mask, red_a_min, 255, cv::THRESH_BINARY);
        cv::threshold(b_ch, blue_lab_mask, blue_b_max, 255, cv::THRESH_BINARY_INV);

        cv::Mat red_mask = eligible_mask & red_h_mask & red_lab_mask;
        cv::Mat blue_mask = eligible_mask & blue_h_mask & blue_lab_mask;

        const double valid_area = static_cast<double>(std::max(1, eligible_pixels));
        const double red_pixels = static_cast<double>(cv::countNonZero(red_mask));
        const double blue_pixels = static_cast<double>(cv::countNonZero(blue_mask));
        const double p_red = clamp01(red_pixels / valid_area);
        const double p_blue = clamp01(blue_pixels / valid_area);

        c.color.p_red = p_red;
        c.color.p_blue = p_blue;

        if (enemy_color == TeamColor::RED) {
            c.color.color_score = p_red - p_blue;
        } else if (enemy_color == TeamColor::BLUE) {
            c.color.color_score = p_blue - p_red;
        } else {
            c.color.color_score = 0.0;
        }

        double decision_threshold = min_score;
        if (eligible_pixels < min_ratio_pixels * 2) {
            decision_threshold += 0.04;
        }
        if (std::max(p_red, p_blue) < 0.02) {
            decision_threshold += 0.05;
        }

        const bool mixed_color = std::abs(p_red - p_blue) < 0.03 && std::max(p_red, p_blue) > 0.08;
        if (team_color == TeamColor::UNKNOWN && strict_unknown) {
            c.color.decided = TeamColor::UNKNOWN;
            c.color.confident = false;
            c.color.color_score = 0.0;
        } else if (mixed_color || std::abs(c.color.color_score) < decision_threshold) {
            c.color.decided = TeamColor::UNKNOWN;
            c.color.confident = false;
            c.color.color_score = 0.0;
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
    const double legacy_margin = std::max(0.0, p_cfg.armor_laser_error_margin_param.get());
    const double laser_err_max = std::max(0.05, p_cfg.pnp_cls_laser_reproj_max_param.get());
    const double armor_err_max = std::max(0.05, p_cfg.pnp_cls_armor_reproj_max_param.get());
    const double margin_abs = std::max(0.0, p_cfg.pnp_cls_margin_abs_param.get());
    const double margin_ratio = clamp01(p_cfg.pnp_cls_margin_ratio_param.get());
    const double ambiguous_band = std::max(0.0, p_cfg.pnp_cls_ambiguous_band_param.get());
    const bool ambiguous_to_unknown = p_cfg.pnp_cls_ambiguous_to_unknown_param.get();
    const double effective_margin_abs = std::max(margin_abs, legacy_margin);

    for (auto& c : candidates) {
        c.topology_ok = c.keypoints.size() >= 4;
        c.raw_class = TargetClass::UNKNOWN;
        c.pnp_ok = false;
        c.laser_template_valid = false;
        c.armor_template_valid = false;
        c.pnp_ambiguous = false;
        c.pnp_margin_delta = 0.0;
        c.pnp_margin_ratio = 0.0;
        c.rvec.release();
        c.tvec.release();
        if (!c.topology_ok) {
            continue;
        }
        const bool use_semantic8 = c.keypoints.size() >= static_cast<std::size_t>(kLaserSemanticKpCount);
        const char* laser_model = use_semantic8 ? "laser_oct" : "laser_quad";
        const std::array<const char*, 2> armor_models = use_semantic8
            ? std::array<const char*, 2> { "armor_small_oct", "armor_big_oct" }
            : std::array<const char*, 2> { "armor_small_quad", "armor_big_quad" };

        cv::Mat rvec_laser_best;
        cv::Mat tvec_laser_best;
        cv::Mat rvec_armor_best;
        cv::Mat tvec_armor_best;
        c.laser_reproj_error = std::numeric_limits<double>::infinity();
        c.armor_reproj_error = std::numeric_limits<double>::infinity();

        auto trySolveModel = [
                                   &c,
                                   &use_semantic8,
                                   this
                               ](
                                   bool try_ippe,
                                   const char* model_name,
                                   wust_vl::algorithm::PnPSolver* epnp_solver,
                                   wust_vl::algorithm::PnPSolver* ippe_solver,
                                   cv::Mat* best_rvec,
                                   cv::Mat* best_tvec,
                                   double* best_err
                               ) {
            cv::Mat rvec_e;
            cv::Mat tvec_e;
            if (epnp_solver->solvePnP(c.keypoints, rvec_e, tvec_e, model_name, camera_k_, camera_d_)) {
                const double err = epnp_solver->calculateReprojectionError(
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

            if (try_ippe && use_semantic8) {
                cv::Mat rvec_i;
                cv::Mat tvec_i;
                if (ippe_solver->solvePnP(c.keypoints, rvec_i, tvec_i, model_name, camera_k_, camera_d_)) {
                    const double err = ippe_solver->calculateReprojectionError(
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

        trySolveModel(
            true,
            laser_model,
            &laser_pnp_solver_epnp_,
            &laser_pnp_solver_ippe_,
            &rvec_laser_best,
            &tvec_laser_best,
            &c.laser_reproj_error
        );
        for (const char* armor_model : armor_models) {
            trySolveModel(
                true,
                armor_model,
                &armor_pnp_solver_epnp_,
                &armor_pnp_solver_ippe_,
                &rvec_armor_best,
                &tvec_armor_best,
                &c.armor_reproj_error
            );
        }

        const bool laser_ok = std::isfinite(c.laser_reproj_error);
        const bool armor_ok = std::isfinite(c.armor_reproj_error);
        c.laser_template_valid = laser_ok && c.laser_reproj_error <= laser_err_max;
        c.armor_template_valid = armor_ok && c.armor_reproj_error <= armor_err_max;

        if (laser_ok && armor_ok) {
            c.pnp_margin_delta = c.armor_reproj_error - c.laser_reproj_error;
            const double denom = std::max(1e-6, std::min(c.armor_reproj_error, c.laser_reproj_error));
            c.pnp_margin_ratio = c.pnp_margin_delta / denom;
        }

        const bool laser_margin_abs_ok = laser_ok && armor_ok
            && (c.laser_reproj_error + effective_margin_abs <= c.armor_reproj_error);
        const bool armor_margin_abs_ok = laser_ok && armor_ok
            && (c.armor_reproj_error + effective_margin_abs <= c.laser_reproj_error);
        const bool laser_margin_ratio_ok = laser_ok && armor_ok
            && (c.laser_reproj_error <= c.armor_reproj_error * (1.0 - margin_ratio));
        const bool armor_margin_ratio_ok = laser_ok && armor_ok
            && (c.armor_reproj_error <= c.laser_reproj_error * (1.0 - margin_ratio));

        bool choose_laser = c.laser_template_valid
            && (!c.armor_template_valid || laser_margin_abs_ok || laser_margin_ratio_ok);
        bool choose_armor = c.armor_template_valid
            && (!c.laser_template_valid || armor_margin_abs_ok || armor_margin_ratio_ok);

        if (c.laser_template_valid && c.armor_template_valid) {
            const bool close_band = std::abs(c.pnp_margin_delta) < ambiguous_band;
            const bool no_strong_winner = !choose_laser && !choose_armor;
            c.pnp_ambiguous = close_band || no_strong_winner;
        }

        if (c.pnp_ambiguous && ambiguous_to_unknown) {
            choose_laser = false;
            choose_armor = false;
        }

        if (choose_laser && !choose_armor) {
            c.raw_class = TargetClass::LASER_MODULE;
            c.pnp_ok = true;
            c.rvec = rvec_laser_best;
            c.tvec = tvec_laser_best;
        } else if (choose_armor && !choose_laser) {
            c.raw_class = TargetClass::ARMOR;
            c.pnp_ok = true;
            c.rvec = rvec_armor_best;
            c.tvec = tvec_armor_best;
        } else if (c.laser_template_valid && !c.armor_template_valid) {
            c.raw_class = TargetClass::LASER_MODULE;
            c.pnp_ok = true;
            c.rvec = rvec_laser_best;
            c.tvec = tvec_laser_best;
        } else if (c.armor_template_valid && !c.laser_template_valid) {
            c.raw_class = TargetClass::ARMOR;
            c.pnp_ok = true;
            c.rvec = rvec_armor_best;
            c.tvec = tvec_armor_best;
        } else if (c.laser_template_valid && c.armor_template_valid) {
            if (c.laser_reproj_error <= c.armor_reproj_error) {
                c.raw_class = TargetClass::LASER_MODULE;
                c.pnp_ok = true;
                c.rvec = rvec_laser_best;
                c.tvec = tvec_laser_best;
            } else {
                c.raw_class = TargetClass::ARMOR;
                c.pnp_ok = true;
                c.rvec = rvec_armor_best;
                c.tvec = tvec_armor_best;
            }
            c.pnp_ambiguous = true;
            if (ambiguous_to_unknown) {
                c.raw_class = TargetClass::UNKNOWN;
                c.pnp_ok = false;
                c.rvec.release();
                c.tvec.release();
            }
        } else {
            c.raw_class = TargetClass::UNKNOWN;
            c.pnp_ok = false;
        }
    }
}

void LaserPipeline::solveCandidateAimCenters(std::vector<Candidate>& candidates) {
    for (auto& c : candidates) {
        const auto semantic_center = centerFromSemantic8(c.keypoints);
        const cv::Point2f fallback_center = semantic_center.value_or(
            cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F)
        );
        c.geometric_center_px = fallback_center;
        c.aim_center_from_pnp = false;

        if (!c.pnp_ok) {
            c.aim_center_px = fallback_center;
            continue;
        }

        const auto projected_center = solveAimCenter(c);
        if (projected_center.has_value()) {
            c.aim_center_px = *projected_center;
            c.aim_center_from_pnp = true;
            continue;
        }

        c.aim_center_px = fallback_center;
        c.pnp_ok = false;
    }
}

std::optional<cv::Point2f> LaserPipeline::solveAimCenter(const Candidate& c) {
    if (!c.pnp_ok || c.rvec.empty() || c.tvec.empty()) {
        return std::nullopt;
    }

    wust_vl::algorithm::PnPSolver* solver = nullptr;
    const char* center_model = nullptr;
    if (c.raw_class == TargetClass::LASER_MODULE) {
        solver = &laser_pnp_solver_epnp_;
        center_model = "laser_center";
    } else if (c.raw_class == TargetClass::ARMOR) {
        solver = &armor_pnp_solver_epnp_;
        center_model = "armor_center";
    } else {
        return std::nullopt;
    }

    const auto projected = solver->getImagePoints(
        c.rvec,
        c.tvec,
        center_model,
        camera_k_,
        camera_d_
    );

    if (projected.empty()) {
        return std::nullopt;
    }
    if (!std::isfinite(projected.front().x) || !std::isfinite(projected.front().y)) {
        return std::nullopt;
    }
    return projected.front();
}

} // namespace laser_aim::modules::perception
