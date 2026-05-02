#include "laser_aim/app/laser_vision_node.hpp"
#include "laser_aim/utils/config_paths.hpp"

#include <wust_vl/common/utils/logger.hpp>
#include <wust_vl/common/utils/signal.hpp>

#include <chrono>
#include <string>
#include <thread>

int main(int argc, char** argv) {
    bool debug_mode = false;
    if (argc > 1) {
        const std::string v = argv[1];
        debug_mode = (v == "1" || v == "true");
    }

    laser_aim::app::LaserVisionNode node(
        laser_aim::kCommonConfig,
        laser_aim::kCameraConfig,
        laser_aim::kPipelineConfig,
        laser_aim::kModelConfig
    );

    if (!node.init(debug_mode)) {
        WUST_ERROR("main") << "failed to init node";
        return -1;
    }

    node.start();

    wust_vl::common::utils::SignalHandler sig;
    sig.start([&node]() { node.stop(); });

    while (!sig.shouldExit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    node.stop();
    return 0;
}
