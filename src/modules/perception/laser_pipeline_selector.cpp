#include "laser_aim/modules/perception/laser_pipeline.hpp"
#include "pipeline_internals.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace laser_aim::modules::perception {

using internal::clamp01;
using internal::centerFromSemantic8;

double LaserPipeline::candidateScore(const Candidate& c, LockStage stage) const {
    double cls_bonus = 0.0;
    if (c.raw_class == TargetClass::LASER_MODULE) {
        cls_bonus = 1.1;
    } else if (c.raw_class == TargetClass::ARMOR) {
        cls_bonus = -0.6;
    }

    const double pnp_bonus_raw = c.pnp_ok ? std::max(0.0, 1.0 - c.laser_reproj_error / 6.0) : -0.5;
    const double enemy_bonus = c.is_enemy_laser ? 1.2 : 0.0;

    double color_weight = 1.0;
    double pnp_weight = 1.0;
    const double geom_weight = (cfg_hub_ != nullptr)
        ? cfg_hub_->pipeline().classic_geom_score_weight_param.get()
        : 0.8;
    double geom_weight_stage = geom_weight;
    double temporal_weight = 0.0;
    if (stage == LockStage::STAGE_3 && cfg_hub_ != nullptr) {
        color_weight = cfg_hub_->pipeline().stage3_color_weight_param.get();
        geom_weight_stage = cfg_hub_->pipeline().stage3_geom_weight_param.get();
        pnp_weight = cfg_hub_->pipeline().stage3_pnp_weight_param.get();
        temporal_weight = cfg_hub_->pipeline().stage3_temporal_weight_param.get();
    }

    double temporal_bonus = 0.0;
    if (last_selected_center_.has_value()) {
        const auto semantic_center = centerFromSemantic8(c.keypoints);
        const cv::Point2f center = semantic_center.value_or(
            cv::Point2f(c.bbox.x + c.bbox.width * 0.5F, c.bbox.y + c.bbox.height * 0.5F)
        );
        const double dist = cv::norm(center - *last_selected_center_);
        temporal_bonus = clamp01(1.0 - dist / 220.0);
    }

    return c.fused_score + static_cast<double>(c.detector_confidence) + color_weight * c.color.color_score + cls_bonus
        + pnp_weight * pnp_bonus_raw + enemy_bonus + geom_weight_stage * c.geom_consistency_score
        + temporal_weight * temporal_bonus;
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

void LaserPipeline::smoothSelectedColor(std::optional<Candidate>* selected, const TeamPolicy& policy) {
    if (cfg_hub_ == nullptr || selected == nullptr) {
        return;
    }

    if (color_smooth_enemy_ref_ != policy.enemy_color) {
        color_score_window_.clear();
        color_valid_window_.clear();
        color_stable_decision_ = TeamColor::UNKNOWN;
        color_smooth_enemy_ref_ = policy.enemy_color;
    }

    if (!selected->has_value()) {
        return;
    }

    auto& c = selected->value();
    const auto& p_cfg = cfg_hub_->pipeline();
    const int window_size = std::max(1, p_cfg.color_smooth_window_param.get());
    const double min_valid_ratio = std::clamp(p_cfg.color_smooth_valid_ratio_min_param.get(), 0.0, 1.0);
    const double hysteresis = std::max(0.0, p_cfg.color_smooth_hysteresis_param.get());
    const double min_score = std::max(0.01, p_cfg.enemy_color_score_min_param.get());

    bool measurement_valid = c.color.confident
        && c.color.decided != TeamColor::UNKNOWN
        && policy.enemy_color != TeamColor::UNKNOWN
        && policy.team_color != TeamColor::UNKNOWN;
    double measurement_score = std::clamp(c.color.color_score, -1.0, 1.0);
    if (!std::isfinite(measurement_score)) {
        measurement_score = 0.0;
        measurement_valid = false;
    }
    if (!measurement_valid) {
        measurement_score = 0.0;
    }

    color_score_window_.push_back(measurement_score);
    color_valid_window_.push_back(measurement_valid ? 1 : 0);
    while (static_cast<int>(color_score_window_.size()) > window_size) {
        color_score_window_.pop_front();
    }
    while (static_cast<int>(color_valid_window_.size()) > window_size) {
        color_valid_window_.pop_front();
    }
    if (color_score_window_.empty() || color_valid_window_.empty()) {
        c.color.decided = TeamColor::UNKNOWN;
        c.color.confident = false;
        c.color.color_score = 0.0;
        return;
    }

    const double avg_score = std::accumulate(color_score_window_.begin(), color_score_window_.end(), 0.0)
        / static_cast<double>(color_score_window_.size());
    const double valid_count = static_cast<double>(std::accumulate(color_valid_window_.begin(), color_valid_window_.end(), 0));
    const double valid_ratio = valid_count / static_cast<double>(color_valid_window_.size());

    TeamColor decided = TeamColor::UNKNOWN;
    bool confident = false;
    if (!(policy.team_color == TeamColor::UNKNOWN && policy.unknown_color_safe_mode)) {
        if (valid_ratio >= min_valid_ratio && std::abs(avg_score) >= (min_score + hysteresis)) {
            decided = (avg_score > 0.0) ? policy.enemy_color : policy.team_color;
            confident = decided != TeamColor::UNKNOWN;
        }
    }

    if (confident && color_stable_decision_ != TeamColor::UNKNOWN && decided != color_stable_decision_) {
        if (std::abs(avg_score) < (min_score + 2.0 * hysteresis)) {
            decided = color_stable_decision_;
        }
    }

    if (confident && decided != TeamColor::UNKNOWN) {
        color_stable_decision_ = decided;
    } else if (valid_ratio < 0.2) {
        color_stable_decision_ = TeamColor::UNKNOWN;
    }

    c.color.decided = confident ? decided : TeamColor::UNKNOWN;
    c.color.confident = confident;
    c.color.color_score = confident ? avg_score : 0.0;
}

} // namespace laser_aim::modules::perception
