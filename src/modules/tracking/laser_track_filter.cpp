#include "laser_aim/modules/tracking/laser_track_filter.hpp"

#include <algorithm>

namespace laser_aim::modules::tracking {

namespace {

using Matrix66 = Eigen::Matrix<double, 6, 6>;
using Matrix33 = Eigen::Matrix<double, 3, 3>;
using Vector3 = Eigen::Matrix<double, 3, 1>;

Matrix66 defaultP0() {
    Matrix66 p = Matrix66::Identity();
    p(0, 0) = 0.5;
    p(1, 1) = 0.5;
    p(2, 2) = 0.5;
    p(3, 3) = 1.0;
    p(4, 4) = 1.0;
    p(5, 5) = 1.0;
    return p;
}

} // namespace

LaserTrackFilter::LaserTrackFilter():
    filter_(predict_model_, measure_model_, []() { return Matrix66::Identity() * 1e-2; }, [](const Vector3&) {
                return Matrix33::Identity() * 4e-3;
            }, defaultP0()) {
    filter_.setUpdateQ([this]() {
        Matrix66 q = Matrix66::Identity();
        const double q_pos = std::max(1e-8, tuning_.q_pos);
        const double q_vel = std::max(1e-8, tuning_.q_vel);
        q(0, 0) = q_pos;
        q(1, 1) = q_pos;
        q(2, 2) = q_pos;
        q(3, 3) = q_vel;
        q(4, 4) = q_vel;
        q(5, 5) = q_vel;
        return q;
    });
    filter_.setUpdateR([this](const Vector3&) {
        Matrix33 r = Matrix33::Identity();
        const double r_pos = std::max(1e-8, tuning_.r_pos);
        r(0, 0) = r_pos;
        r(1, 1) = r_pos;
        r(2, 2) = r_pos;
        return r;
    });
    filter_.setIterationNum(2);
}

void LaserTrackFilter::setTuning(const Tuning& tuning) {
    tuning_ = tuning;
    tuning_.q_pos = std::max(1e-8, tuning_.q_pos);
    tuning_.q_vel = std::max(1e-8, tuning_.q_vel);
    tuning_.r_pos = std::max(1e-8, tuning_.r_pos);
    tuning_.dt_min_s = std::clamp(tuning_.dt_min_s, 0.0, 0.2);
    tuning_.dt_max_s = std::clamp(tuning_.dt_max_s, tuning_.dt_min_s, 0.5);
    tuning_.max_obs_jump_m = std::max(0.0, tuning_.max_obs_jump_m);
    tuning_.prob_smooth = std::clamp(tuning_.prob_smooth, 0.0, 0.99);
}

const LaserTrackFilter::Tuning& LaserTrackFilter::tuning() const {
    return tuning_;
}

void LaserTrackFilter::reset(const Observation& obs) {
    Eigen::Matrix<double, kStateN, 1> x0;
    x0 << obs.pos.x(), obs.pos.y(), obs.pos.z(), 0.0, 0.0, 0.0;
    filter_.setState(x0);
    valid_ = true;
    last_ts_ = obs.timestamp;
    enemy_prob_avg_ = obs.enemy_prob;
    laser_prob_avg_ = obs.laser_prob;
    continuous_confirm_frames_ = 1;
    last_innovation_norm_ = 0.0;
    last_update_rejected_ = false;
}

void LaserTrackFilter::predictTo(std::chrono::steady_clock::time_point now) {
    if (!valid_) {
        return;
    }

    const double dt = std::chrono::duration<double>(now - last_ts_).count();
    if (dt <= 0.0) {
        return;
    }
    if (dt < tuning_.dt_min_s) {
        last_ts_ = now;
        return;
    }
    const double dt_used = std::min(dt, tuning_.dt_max_s);

    predict_model_.dt = dt_used;
    filter_.setPredictFunc(predict_model_);
    filter_.predict();
    last_ts_ = now;
}

void LaserTrackFilter::update(const Observation& obs) {
    if (!valid_) {
        reset(obs);
        return;
    }

    predictTo(obs.timestamp);

    const auto x_pred = filter_.getState();
    const Eigen::Vector3d pred_pos(x_pred[0], x_pred[1], x_pred[2]);
    last_innovation_norm_ = (obs.pos - pred_pos).norm();
    if (tuning_.max_obs_jump_m > 0.0 && last_innovation_norm_ > tuning_.max_obs_jump_m) {
        last_update_rejected_ = true;
        const double a = tuning_.prob_smooth;
        enemy_prob_avg_ = a * enemy_prob_avg_ + (1.0 - a) * obs.enemy_prob;
        laser_prob_avg_ = a * laser_prob_avg_ + (1.0 - a) * obs.laser_prob;
        continuous_confirm_frames_ = 0;
        return;
    }

    Eigen::Matrix<double, kMeasureN, 1> z;
    z << obs.pos.x(), obs.pos.y(), obs.pos.z();
    filter_.update(z);
    last_update_rejected_ = false;

    const double a = tuning_.prob_smooth;
    enemy_prob_avg_ = a * enemy_prob_avg_ + (1.0 - a) * obs.enemy_prob;
    laser_prob_avg_ = a * laser_prob_avg_ + (1.0 - a) * obs.laser_prob;

    if (obs.enemy_prob > 0.5 && obs.laser_prob > 0.5) {
        continuous_confirm_frames_ += 1;
    } else {
        continuous_confirm_frames_ = 0;
    }
}

bool LaserTrackFilter::valid() const {
    return valid_;
}

Eigen::Vector3d LaserTrackFilter::pos() const {
    if (!valid_) {
        return Eigen::Vector3d::Zero();
    }
    const auto x = filter_.getState();
    return Eigen::Vector3d(x[0], x[1], x[2]);
}

Eigen::Vector3d LaserTrackFilter::vel() const {
    if (!valid_) {
        return Eigen::Vector3d::Zero();
    }
    const auto x = filter_.getState();
    return Eigen::Vector3d(x[3], x[4], x[5]);
}

double LaserTrackFilter::enemyProbAvg() const {
    return enemy_prob_avg_;
}

double LaserTrackFilter::laserProbAvg() const {
    return laser_prob_avg_;
}

int LaserTrackFilter::continuousConfirmFrames() const {
    return continuous_confirm_frames_;
}

double LaserTrackFilter::lastInnovationNorm() const {
    return last_innovation_norm_;
}

bool LaserTrackFilter::lastUpdateRejected() const {
    return last_update_rejected_;
}

} // namespace laser_aim::modules::tracking
