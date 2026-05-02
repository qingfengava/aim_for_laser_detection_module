#include "laser_aim/modules/tracking/laser_track_filter.hpp"

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

Matrix66 defaultQ() {
    Matrix66 q = Matrix66::Identity() * 1e-2;
    q(0, 0) = 2e-3;
    q(1, 1) = 2e-3;
    q(2, 2) = 2e-3;
    return q;
}

Matrix33 defaultR(const Vector3&) {
    Matrix33 r = Matrix33::Identity() * 4e-3;
    return r;
}

} // namespace

LaserTrackFilter::LaserTrackFilter():
    filter_(predict_model_, measure_model_, defaultQ, defaultR, defaultP0()) {
    filter_.setIterationNum(2);
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
}

void LaserTrackFilter::predictTo(std::chrono::steady_clock::time_point now) {
    if (!valid_) {
        return;
    }

    const double dt = std::chrono::duration<double>(now - last_ts_).count();
    if (dt <= 0.0) {
        return;
    }

    predict_model_.dt = dt;
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

    Eigen::Matrix<double, kMeasureN, 1> z;
    z << obs.pos.x(), obs.pos.y(), obs.pos.z();
    filter_.update(z);

    constexpr double kSmooth = 0.85;
    enemy_prob_avg_ = kSmooth * enemy_prob_avg_ + (1.0 - kSmooth) * obs.enemy_prob;
    laser_prob_avg_ = kSmooth * laser_prob_avg_ + (1.0 - kSmooth) * obs.laser_prob;

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

} // namespace laser_aim::modules::tracking
