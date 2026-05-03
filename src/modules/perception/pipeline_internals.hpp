#pragma once

#include "laser_aim/common/types.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <optional>

namespace laser_aim::modules::perception::internal {

inline double clamp01(double x) {
    if (x < 0.0) {
        return 0.0;
    }
    if (x > 1.0) {
        return 1.0;
    }
    return x;
}

inline cv::Rect2f bboxFromPoints(const std::vector<cv::Point2f>& pts) {
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

inline std::optional<cv::Point2f> centerFromSemantic8(const std::vector<cv::Point2f>& keypoints) {
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

} // namespace laser_aim::modules::perception::internal
