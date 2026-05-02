#pragma once

#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <sys/mman.h>
#include <unistd.h>
#include <wust_vl/video/icamera.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace laser_aim {

template<typename T, int MaxN>
class LogsStream {
public:
    explicit LogsStream(std::string name): name_(std::move(name)) {}

    void handleOnce(const T& v, nlohmann::json& j) {
        data_.push_back(v);
        trim();
        j[name_] = data_;
    }

    void clear() {
        data_.clear();
    }

private:
    void trim() {
        while (data_.size() > static_cast<std::size_t>(MaxN)) {
            data_.erase(data_.begin());
        }
    }

    std::string name_;
    std::vector<T> data_;
};

template<typename DebugT, typename DrawFn, typename OutputFn>
void drawDebugOverlayImpl(
    const DebugT& dbg,
    bool auto_fps,
    DrawFn&& draw_fn,
    OutputFn&& output_fn
) {
    static auto last_show_time = std::chrono::steady_clock::now();

    if (dbg.frame.img_frame.src_img.empty()) {
        return;
    }

    constexpr double min_interval_ms = 1000.0 / 30.0;
    const auto now = std::chrono::steady_clock::now();
    if (auto_fps
        && std::chrono::duration<double, std::milli>(now - last_show_time).count() < min_interval_ms) {
        return;
    }

    last_show_time = now;

    cv::Mat debug_img;
    const auto& src = dbg.frame.img_frame.src_img;
    const auto pf = dbg.frame.img_frame.pixel_format;
    if (pf == wust_vl::video::PixelFormat::GRAY) {
        cv::cvtColor(src, debug_img, cv::COLOR_GRAY2RGB);
    } else if (pf == wust_vl::video::PixelFormat::BGR) {
        cv::cvtColor(src, debug_img, cv::COLOR_BGR2RGB);
    } else {
        debug_img = src;
    }

    if (debug_img.empty()) {
        return;
    }

    draw_fn(debug_img, dbg);
    output_fn(debug_img);
}

inline auto writeToFile = [](const cv::Mat& img) {
    cv::Mat bgr;
    cv::cvtColor(img, bgr, cv::COLOR_RGB2BGR);

    std::vector<uchar> buf;
    cv::imencode(".jpg", bgr, buf);

    std::ofstream ofs("/dev/shm/debug_frame.jpg.tmp", std::ios::binary);
    ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    ofs.close();
    std::rename("/dev/shm/debug_frame.jpg.tmp", "/dev/shm/debug_frame.jpg");
};

class ShmWriter {
public:
    static constexpr size_t kShmMaxSize = 2 * 1024 * 1024;

    explicit ShmWriter(const char* name, mode_t mode = 0666) {
        fd_ = shm_open(name, O_CREAT | O_RDWR, mode);
        if (fd_ == -1) {
            std::cerr << "[SHM] shm_open failed\n";
            return;
        }

        if (ftruncate(fd_, static_cast<off_t>(kShmMaxSize)) == -1) {
            std::cerr << "[SHM] ftruncate failed\n";
            close(fd_);
            fd_ = -1;
            return;
        }

        ptr_ = mmap(nullptr, kShmMaxSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (ptr_ == MAP_FAILED) {
            std::cerr << "[SHM] mmap failed\n";
            close(fd_);
            fd_ = -1;
            ptr_ = nullptr;
        }
    }

    ~ShmWriter() {
        if (ptr_ != nullptr) {
            munmap(ptr_, kShmMaxSize);
        }
        if (fd_ != -1) {
            close(fd_);
        }
    }

    void operator()(const cv::Mat& img) const {
        if (ptr_ == nullptr) {
            return;
        }

        static const std::vector<int> jpeg_params = { cv::IMWRITE_JPEG_QUALITY, 75 };

        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf, jpeg_params);
        if (buf.size() + 4U > kShmMaxSize) {
            return;
        }

        const uint32_t size = static_cast<uint32_t>(buf.size());
        std::memcpy(ptr_, &size, 4);
        std::memcpy(static_cast<char*>(ptr_) + 4, buf.data(), size);
    }

private:
    int fd_ { -1 };
    void* ptr_ { nullptr };
};

inline auto showWindow(const char* win_name) {
    return [win_name](const cv::Mat& img) {
        cv::imshow(win_name, img);
        cv::waitKey(1);
    };
}

inline void writeJsonAtomically(const std::string& path, const nlohmann::json& j) {
    const std::string tmp_path = path + ".tmp";
    std::ofstream ofs(tmp_path);
    if (!ofs.is_open()) {
        return;
    }
    ofs << j.dump();
    ofs.close();
    std::rename(tmp_path.c_str(), path.c_str());
}

} // namespace laser_aim
