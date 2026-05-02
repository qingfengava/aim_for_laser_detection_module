#pragma once

#include "laser_aim/modules/config/config_groups.hpp"

#include <wust_vl/common/utils/parameter.hpp>

#include <memory>
#include <string>

namespace laser_aim::modules::config {

class ParameterHub {
public:
    bool init(const std::string& common_path, const std::string& pipeline_path, const std::string& model_path);

    void reloadAll();

    [[nodiscard]] TeamPolicy teamPolicy() const;

    RuntimeConfig& runtime();
    PipelineConfig& pipeline();
    ModelConfig& model();

private:
    std::string common_path_;
    std::string pipeline_path_;
    std::string model_path_;

    wust_vl::common::utils::Parameter common_param_;
    wust_vl::common::utils::Parameter pipeline_param_;
    wust_vl::common::utils::Parameter model_param_;

    std::shared_ptr<LoggerConfig> logger_cfg_;
    std::shared_ptr<SystemConfig> system_cfg_;
    std::shared_ptr<RuntimeConfig> runtime_cfg_;
    std::shared_ptr<PipelineConfig> pipeline_cfg_;
    std::shared_ptr<ModelConfig> model_cfg_;
};

} // namespace laser_aim::modules::config
