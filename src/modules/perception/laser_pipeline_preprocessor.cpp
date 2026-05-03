#include "laser_aim/modules/perception/laser_pipeline.hpp"

#include <wust_vl/common/utils/logger.hpp>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace laser_aim::modules::perception {

namespace {

constexpr const char* kLogNode = "laser.pipeline";

} // namespace

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

} // namespace laser_aim::modules::perception
