#include "laser_aim/modules/perception/laser_pipeline.hpp"

#include <opencv2/imgproc.hpp>

namespace laser_aim::modules::perception {

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

        const std::string txt = toString(c.raw_class) + " conf=" + std::to_string(c.detector_confidence)
            + " geom=" + std::to_string(c.geom_consistency_score)
            + " aim=" + (c.aim_center_from_pnp ? "3D" : "2D");
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
        + " route=" + toString(out.route_mode) + " reason=" + out.gate_reason;
    const std::string line2 = "enemy_prob=" + std::to_string(out.track.enemy_prob_avg)
        + " laser_prob=" + std::to_string(out.track.laser_prob_avg)
        + " confirm=" + std::to_string(out.track.continuous_confirm_frames)
        + " innov=" + std::to_string(out.track.innovation_norm_m)
        + " rej=" + std::string(out.track.innovation_rejected ? "1" : "0");
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
