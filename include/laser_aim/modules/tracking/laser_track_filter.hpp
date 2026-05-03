#pragma once

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "KalmanHyLib/kalman_hybird_lib.hpp"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <Eigen/Dense>

#include <chrono>

namespace laser_aim::modules::tracking {

class LaserTrackFilter {
public:
    struct Tuning {
        double q_pos { 2e-3 };
        double q_vel { 1e-2 };
        double r_pos { 4e-3 };
        double dt_min_s { 1e-3 };
        double dt_max_s { 6e-2 };
        double max_obs_jump_m { 1.2 };
        double prob_smooth { 0.85 };
    };

    struct Observation {
        Eigen::Vector3d pos { Eigen::Vector3d::Zero() };
        double enemy_prob { 0.0 };
        double laser_prob { 0.0 };
        std::chrono::steady_clock::time_point timestamp;
    };

    LaserTrackFilter();

    void setTuning(const Tuning& tuning);
    [[nodiscard]] const Tuning& tuning() const;
    void reset(const Observation& obs);
    void predictTo(std::chrono::steady_clock::time_point now);
    void update(const Observation& obs);

    [[nodiscard]] bool valid() const;
    [[nodiscard]] Eigen::Vector3d pos() const;
    [[nodiscard]] Eigen::Vector3d vel() const;
    [[nodiscard]] double enemyProbAvg() const;
    [[nodiscard]] double laserProbAvg() const;
    [[nodiscard]] int continuousConfirmFrames() const;
    [[nodiscard]] double lastInnovationNorm() const;
    [[nodiscard]] bool lastUpdateRejected() const;

private:
    static constexpr int kStateN = 6;  // x y z vx vy vz
    static constexpr int kMeasureN = 3; // x y z

    struct PredictModel {
        double dt { 0.01 };

        template<typename T>
        void operator()(const T x0[kStateN], T x1[kStateN]) const {
            x1[0] = x0[0] + x0[3] * T(dt);
            x1[1] = x0[1] + x0[4] * T(dt);
            x1[2] = x0[2] + x0[5] * T(dt);
            x1[3] = x0[3];
            x1[4] = x0[4];
            x1[5] = x0[5];
        }
    };

    struct MeasureModel {
        template<typename T>
        void operator()(const T x[kStateN], T z[kMeasureN]) const {
            z[0] = x[0];
            z[1] = x[1];
            z[2] = x[2];
        }
    };

    using Filter = kalman_hybird_lib::ErrorStateEKF<kStateN, kMeasureN, PredictModel, MeasureModel>;

    Filter filter_;
    PredictModel predict_model_;
    MeasureModel measure_model_;

    bool valid_ { false };
    std::chrono::steady_clock::time_point last_ts_;
    Tuning tuning_ {};
    double enemy_prob_avg_ { 0.0 };
    double laser_prob_avg_ { 0.0 };
    int continuous_confirm_frames_ { 0 };
    double last_innovation_norm_ { 0.0 };
    bool last_update_rejected_ { false };
};

} // namespace laser_aim::modules::tracking
