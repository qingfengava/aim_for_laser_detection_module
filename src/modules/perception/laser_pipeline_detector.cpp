#include "laser_aim/modules/perception/laser_pipeline.hpp"
#include "pipeline_internals.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>

namespace laser_aim::modules::perception {

using internal::clamp01;
using internal::bboxFromPoints;
using internal::centerFromSemantic8;

namespace {

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

bool splitBarInnerOuterByTopBottom(
    const LightBar& bar,
    const cv::Point2f& top_to_bottom_unit,
    const cv::Point2f& left_to_right_unit,
    bool is_top_bar,
    cv::Point2f* inner_left,
    cv::Point2f* inner_right,
    cv::Point2f* outer_left,
    cv::Point2f* outer_right
) {
    if (inner_left == nullptr || inner_right == nullptr || outer_left == nullptr || outer_right == nullptr) {
        return false;
    }

    struct ScoredPt {
        cv::Point2f pt;
        float tb_score { 0.0F };
        float lr_score { 0.0F };
    };

    std::array<ScoredPt, 4> scored;
    for (int i = 0; i < 4; ++i) {
        const auto& p = bar.corners[static_cast<std::size_t>(i)];
        const cv::Point2f delta = p - bar.center;
        const float tb = delta.x * top_to_bottom_unit.x + delta.y * top_to_bottom_unit.y;
        const float lr = delta.x * left_to_right_unit.x + delta.y * left_to_right_unit.y;
        scored[static_cast<std::size_t>(i)] = { p, tb, lr };
    }

    std::sort(
        scored.begin(),
        scored.end(),
        [](const ScoredPt& a, const ScoredPt& b) { return a.tb_score > b.tb_score; }
    );

    std::array<ScoredPt, 2> inners {};
    std::array<ScoredPt, 2> outers {};
    if (is_top_bar) {
        inners = { scored[0], scored[1] };
        outers = { scored[2], scored[3] };
    } else {
        inners = { scored[2], scored[3] };
        outers = { scored[0], scored[1] };
    }

    std::sort(
        inners.begin(),
        inners.end(),
        [](const ScoredPt& a, const ScoredPt& b) { return a.lr_score < b.lr_score; }
    );
    std::sort(
        outers.begin(),
        outers.end(),
        [](const ScoredPt& a, const ScoredPt& b) { return a.lr_score < b.lr_score; }
    );

    *inner_left = inners[0].pt;
    *inner_right = inners[1].pt;
    *outer_left = outers[0].pt;
    *outer_right = outers[1].pt;
    return true;
}

bool buildSemantic8FromBarPair(
    const LightBar& top_bar,
    const LightBar& bottom_bar,
    std::vector<cv::Point2f>* keypoints
) {
    if (keypoints == nullptr) {
        return false;
    }

    const cv::Point2f tb = bottom_bar.center - top_bar.center;
    const float tb_norm = std::sqrt(tb.x * tb.x + tb.y * tb.y);
    if (tb_norm < 1e-3F) {
        return false;
    }
    const cv::Point2f tb_unit(tb.x / tb_norm, tb.y / tb_norm);
    cv::Point2f lr_unit(-tb_unit.y, tb_unit.x);
    if (lr_unit.x < 0.0F) {
        lr_unit *= -1.0F;
    }

    cv::Point2f top_inner_left;
    cv::Point2f top_inner_right;
    cv::Point2f top_outer_left;
    cv::Point2f top_outer_right;
    cv::Point2f bottom_inner_left;
    cv::Point2f bottom_inner_right;
    cv::Point2f bottom_outer_left;
    cv::Point2f bottom_outer_right;

    const bool top_ok = splitBarInnerOuterByTopBottom(
        top_bar,
        tb_unit,
        lr_unit,
        true,
        &top_inner_left,
        &top_inner_right,
        &top_outer_left,
        &top_outer_right
    );
    const bool bottom_ok = splitBarInnerOuterByTopBottom(
        bottom_bar,
        tb_unit,
        lr_unit,
        false,
        &bottom_inner_left,
        &bottom_inner_right,
        &bottom_outer_left,
        &bottom_outer_right
    );

    if (!top_ok || !bottom_ok) {
        return false;
    }

    keypoints->assign(static_cast<std::size_t>(kLaserSemanticKpCount), cv::Point2f(0.0F, 0.0F));
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)] = top_inner_left;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)] = top_inner_right;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)] = bottom_inner_left;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)] = bottom_inner_right;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP4_UPPER_LEFT_OUTER)] = top_outer_left;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP5_UPPER_RIGHT_OUTER)] = top_outer_right;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP6_LOWER_LEFT_OUTER)] = bottom_outer_left;
    (*keypoints)[static_cast<std::size_t>(LaserSemanticKp::KP7_LOWER_RIGHT_OUTER)] = bottom_outer_right;
    return true;
}

} // namespace

std::vector<Candidate> LaserPipeline::runDualCandidateGeneration(const cv::Mat& src_img, TeamColor enemy_color) {
    const auto classic_candidates = runClassicCandidateGeneration(src_img, enemy_color);
    if (cfg_hub_ != nullptr && cfg_hub_->pipeline().use_classic_only_param.get()) {
        return classic_candidates;
    }
    if (recapture_mode_) {
        return classic_candidates;
    }
    const auto yolo_candidates = runYoloCandidateGeneration(src_img);
    return fuseCandidates(yolo_candidates, classic_candidates);
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

std::vector<Candidate> LaserPipeline::runClassicCandidateGeneration(const cv::Mat& src_img, TeamColor enemy_color) {
    std::vector<Candidate> out;
    const auto* p_cfg = (cfg_hub_ != nullptr) ? &cfg_hub_->pipeline() : nullptr;
    const int bin_threshold = p_cfg != nullptr ? p_cfg->classic_bin_threshold_param.get() : 210;
    const bool use_color_diff_cfg = p_cfg != nullptr && p_cfg->classic_use_color_diff_param.get();
    const double color_diff_threshold = p_cfg != nullptr
        ? p_cfg->classic_color_diff_threshold_param.get()
        : 45.0;
    const double brightness_threshold = p_cfg != nullptr
        ? p_cfg->classic_brightness_threshold_param.get()
        : 120.0;
    const bool unknown_use_color_diff = p_cfg != nullptr && p_cfg->classic_unknown_use_color_diff_param.get();
    int morph_open_kernel = p_cfg != nullptr ? p_cfg->classic_morph_open_kernel_param.get() : 3;
    const double contour_area_min = p_cfg != nullptr ? p_cfg->classic_contour_area_min_param.get() : 12.0;
    const double min_rect_side = p_cfg != nullptr ? p_cfg->classic_min_rect_side_param.get() : 2.0;
    const double bar_long_side_min = p_cfg != nullptr ? p_cfg->classic_bar_long_side_min_param.get() : 6.0;
    const double bar_short_side_min = p_cfg != nullptr ? p_cfg->classic_bar_short_side_min_param.get() : 1.5;
    double bar_ratio_min = p_cfg != nullptr ? p_cfg->classic_bar_ratio_min_param.get() : 1.6;
    double bar_ratio_max = p_cfg != nullptr ? p_cfg->classic_bar_ratio_max_param.get() : 20.0;
    const double pair_top_bottom_gap_min = p_cfg != nullptr ? p_cfg->classic_pair_top_bottom_gap_min_param.get() : 1.0;
    double pair_top_bottom_gap_ratio_min = p_cfg != nullptr
        ? p_cfg->classic_pair_top_bottom_gap_ratio_min_param.get()
        : 0.5;
    double pair_top_bottom_gap_ratio_max = p_cfg != nullptr
        ? p_cfg->classic_pair_top_bottom_gap_ratio_max_param.get()
        : 8.0;
    const double pair_left_right_misaligned_ratio_max = p_cfg != nullptr
        ? p_cfg->classic_pair_left_right_misaligned_ratio_max_param.get()
        : 1.0;
    const double pair_len_ratio_max = p_cfg != nullptr ? p_cfg->classic_pair_len_ratio_max_param.get() : 1.8;
    const double pair_angle_diff_max_deg = p_cfg != nullptr
        ? p_cfg->classic_pair_angle_diff_max_deg_param.get()
        : 20.0;
    const double conf_area_norm = p_cfg != nullptr ? p_cfg->classic_conf_area_norm_param.get() : 1500.0;
    const double conf_fallback_area_norm = p_cfg != nullptr
        ? p_cfg->classic_conf_fallback_area_norm_param.get()
        : 800.0;
    const double geom_score_min = p_cfg != nullptr ? p_cfg->classic_geom_score_min_param.get() : 0.55;
    const double geom_low_quality_penalty = p_cfg != nullptr
        ? p_cfg->classic_geom_low_quality_penalty_param.get()
        : 0.35;
    const bool enable_single_bar_fallback = p_cfg != nullptr
        && p_cfg->classic_enable_single_bar_fallback_param.get();

    morph_open_kernel = std::max(1, morph_open_kernel);
    if ((morph_open_kernel % 2) == 0) {
        morph_open_kernel += 1;
    }
    if (bar_ratio_min > bar_ratio_max) {
        std::swap(bar_ratio_min, bar_ratio_max);
    }
    if (pair_top_bottom_gap_ratio_min > pair_top_bottom_gap_ratio_max) {
        std::swap(pair_top_bottom_gap_ratio_min, pair_top_bottom_gap_ratio_max);
    }
    const double angle_norm = std::max(1e-6, pair_angle_diff_max_deg);
    const double shape_denom = std::max(1e-6, pair_len_ratio_max - 1.0);
    const double conf_area_norm_safe = std::max(1e-6, conf_area_norm);
    const double conf_fallback_area_norm_safe = std::max(1e-6, conf_fallback_area_norm);

    cv::Mat bgr;
    if (src_img.channels() == 3) {
        bgr = src_img;
    } else {
        cv::cvtColor(src_img, bgr, cv::COLOR_GRAY2BGR);
    }

    cv::Mat bin;
    const bool enemy_color_known = enemy_color == TeamColor::RED || enemy_color == TeamColor::BLUE;
    const bool use_color_diff = use_color_diff_cfg && (enemy_color_known || unknown_use_color_diff);
    if (use_color_diff) {
        std::vector<cv::Mat> bgr_channels;
        cv::split(bgr, bgr_channels);
        const cv::Mat& blue_ch = bgr_channels[0];
        const cv::Mat& red_ch = bgr_channels[2];

        cv::Mat red_diff;
        cv::Mat blue_diff;
        cv::subtract(red_ch, blue_ch, red_diff);
        cv::subtract(blue_ch, red_ch, blue_diff);

        cv::Mat color_mask;
        if (enemy_color == TeamColor::RED) {
            cv::threshold(red_diff, color_mask, color_diff_threshold, 255, cv::THRESH_BINARY);
        } else if (enemy_color == TeamColor::BLUE) {
            cv::threshold(blue_diff, color_mask, color_diff_threshold, 255, cv::THRESH_BINARY);
        } else {
            cv::Mat red_mask;
            cv::Mat blue_mask;
            cv::threshold(red_diff, red_mask, color_diff_threshold, 255, cv::THRESH_BINARY);
            cv::threshold(blue_diff, blue_mask, color_diff_threshold, 255, cv::THRESH_BINARY);
            color_mask = red_mask | blue_mask;
        }

        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> hsv_channels;
        cv::split(hsv, hsv_channels);
        cv::Mat bright_mask;
        cv::threshold(hsv_channels[2], bright_mask, brightness_threshold, 255, cv::THRESH_BINARY);
        bin = color_mask & bright_mask;
    } else {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, bin, bin_threshold, 255, cv::THRESH_BINARY);
    }
    const cv::Mat morph_kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(morph_open_kernel, morph_open_kernel)
    );
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, morph_kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<LightBar> bars;
    bars.reserve(contours.size());

    for (const auto& cnt : contours) {
        const double area = cv::contourArea(cnt);
        if (area < contour_area_min) {
            continue;
        }

        const cv::RotatedRect rr = cv::minAreaRect(cnt);
        if (rr.size.width < min_rect_side || rr.size.height < min_rect_side) {
            continue;
        }

        const LightBar bar = buildLightBar(rr, area);
        const float ratio = bar.long_side / std::max(bar.short_side, 1.0F);
        if (bar.long_side < bar_long_side_min || bar.short_side < bar_short_side_min) {
            continue;
        }
        if (ratio < bar_ratio_min || ratio > bar_ratio_max) {
            continue;
        }
        bars.push_back(bar);
    }

    int next_id = 0;
    for (std::size_t i = 0; i < bars.size(); ++i) {
        for (std::size_t j = i + 1; j < bars.size(); ++j) {
            const LightBar* top = &bars[i];
            const LightBar* bottom = &bars[j];
            if (top->center.y > bottom->center.y) {
                std::swap(top, bottom);
            }

            const float avg_len = (top->long_side + bottom->long_side) * 0.5F;
            if (avg_len <= 1.0F) {
                continue;
            }

            const float dy = bottom->center.y - top->center.y;
            if (dy <= pair_top_bottom_gap_min) {
                continue;
            }
            const float dx = std::abs(bottom->center.x - top->center.x);
            const float gap_ratio = dy / avg_len;
            const float x_misaligned_ratio = dx / avg_len;
            const float len_ratio = std::max(top->long_side, bottom->long_side)
                / std::max(1.0F, std::min(top->long_side, bottom->long_side));
            const float angle_diff = angleDiffAbs(top->long_angle_deg, bottom->long_angle_deg);

            if (gap_ratio < pair_top_bottom_gap_ratio_min || gap_ratio > pair_top_bottom_gap_ratio_max) {
                continue;
            }
            if (x_misaligned_ratio > pair_left_right_misaligned_ratio_max) {
                continue;
            }
            if (len_ratio > pair_len_ratio_max) {
                continue;
            }
            if (angle_diff > pair_angle_diff_max_deg) {
                continue;
            }

            std::vector<cv::Point2f> semantic_kps;
            if (!buildSemantic8FromBarPair(*top, *bottom, &semantic_kps)) {
                continue;
            }

            const double area_score = clamp01((top->contour_area + bottom->contour_area) / conf_area_norm_safe);
            const double align_score = clamp01(1.0 - static_cast<double>(x_misaligned_ratio));
            const double angle_score = clamp01(1.0 - static_cast<double>(angle_diff) / angle_norm);
            const double shape_score = clamp01(1.0 - static_cast<double>(len_ratio - 1.0) / shape_denom);
            const double gap_center = (pair_top_bottom_gap_ratio_min + pair_top_bottom_gap_ratio_max) * 0.5;
            const double gap_half = std::max(
                1e-6,
                (pair_top_bottom_gap_ratio_max - pair_top_bottom_gap_ratio_min) * 0.5
            );
            const double gap_balance_score = clamp01(1.0 - std::abs(static_cast<double>(gap_ratio) - gap_center) / gap_half);
            const double top_inner_w = cv::norm(
                semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)]
                - semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)]
            );
            const double bottom_inner_w = cv::norm(
                semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)]
                - semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)]
            );
            const double top_outer_w = cv::norm(
                semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP4_UPPER_LEFT_OUTER)]
                - semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP5_UPPER_RIGHT_OUTER)]
            );
            const double bottom_outer_w = cv::norm(
                semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP6_LOWER_LEFT_OUTER)]
                - semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP7_LOWER_RIGHT_OUTER)]
            );
            const double left_h = cv::norm(
                semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP0_UPPER_LEFT_INNER)]
                - semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP2_LOWER_LEFT_INNER)]
            );
            const double right_h = cv::norm(
                semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP1_UPPER_RIGHT_INNER)]
                - semantic_kps[static_cast<std::size_t>(LaserSemanticKp::KP3_LOWER_RIGHT_INNER)]
            );
            const double inner_sym = 1.0 - std::abs(top_inner_w - bottom_inner_w) / std::max(1e-6, std::max(top_inner_w, bottom_inner_w));
            const double outer_sym = 1.0 - std::abs(top_outer_w - bottom_outer_w) / std::max(1e-6, std::max(top_outer_w, bottom_outer_w));
            const double lr_height_sym = 1.0 - std::abs(left_h - right_h) / std::max(1e-6, std::max(left_h, right_h));
            const double structure_score = clamp01((clamp01(inner_sym) + clamp01(outer_sym) + clamp01(lr_height_sym)) / 3.0);
            const double geom_consistency_score = clamp01(
                0.24 * align_score
                + 0.20 * angle_score
                + 0.20 * shape_score
                + 0.18 * gap_balance_score
                + 0.18 * structure_score
            );
            const float conf = static_cast<float>(clamp01(
                0.15 + 0.35 * area_score + 0.2 * align_score + 0.15 * angle_score + 0.15 * shape_score
            ));

            Candidate c;
            c.candidate_id = next_id++;
            c.raw_class = TargetClass::UNKNOWN;
            c.keypoints = std::move(semantic_kps);
            c.bbox = bboxFromPoints(c.keypoints);
            c.detector_confidence = conf;
            c.geom_consistency_score = geom_consistency_score;
            if (c.geom_consistency_score < geom_score_min) {
                c.detector_confidence = static_cast<float>(
                    c.detector_confidence * clamp01(geom_low_quality_penalty)
                );
            }
            c.from_classic = true;
            out.push_back(std::move(c));
        }
    }

    // Fallback: keep a minimal candidate path when pairing fails.
    if (out.empty() && enable_single_bar_fallback) {
        for (const auto& bar : bars) {
            Candidate c;
            c.candidate_id = next_id++;
            c.raw_class = TargetClass::UNKNOWN;
            c.bbox = bar.rect.boundingRect2f();
            c.keypoints = rectPoints(bar.rect);
            c.detector_confidence = static_cast<float>(clamp01(bar.contour_area / conf_fallback_area_norm_safe));
            c.geom_consistency_score = 0.10;
            if (c.geom_consistency_score < geom_score_min) {
                c.detector_confidence = static_cast<float>(
                    c.detector_confidence * clamp01(geom_low_quality_penalty)
                );
            }
            c.from_classic = true;
            out.push_back(std::move(c));
        }
    }

    return out;
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

} // namespace laser_aim::modules::perception
