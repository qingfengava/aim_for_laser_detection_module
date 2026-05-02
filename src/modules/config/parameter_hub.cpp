#include "laser_aim/modules/config/parameter_hub.hpp"

namespace laser_aim::modules::config {

bool ParameterHub::init(
    const std::string& common_path,
    const std::string& pipeline_path,
    const std::string& model_path
) {
    common_path_ = common_path;
    pipeline_path_ = pipeline_path;
    model_path_ = model_path;

    logger_cfg_ = LoggerConfig::create();
    system_cfg_ = SystemConfig::create();
    runtime_cfg_ = RuntimeConfig::create();
    pipeline_cfg_ = PipelineConfig::create();
    model_cfg_ = ModelConfig::create();

    common_param_.loadFromFile(common_path_);
    common_param_.registerGroup(*logger_cfg_);
    common_param_.registerGroup(*system_cfg_);
    common_param_.registerGroup(*runtime_cfg_);
    common_param_.reloadFromOldPath();

    pipeline_param_.loadFromFile(pipeline_path_);
    pipeline_param_.registerGroup(*pipeline_cfg_);
    pipeline_param_.reloadFromOldPath();

    model_param_.loadFromFile(model_path_);
    model_param_.registerGroup(*model_cfg_);
    model_param_.reloadFromOldPath();

    wust_vl::common::utils::ParameterManager::instance().registerParameter(common_param_);
    wust_vl::common::utils::ParameterManager::instance().registerParameter(pipeline_param_);
    wust_vl::common::utils::ParameterManager::instance().registerParameter(model_param_);
    return true;
}

void ParameterHub::reloadAll() {
    common_param_.reloadFromOldPath();
    pipeline_param_.reloadFromOldPath();
    model_param_.reloadFromOldPath();
}

TeamPolicy ParameterHub::teamPolicy() const {
    return system_cfg_->toPolicy();
}

RuntimeConfig& ParameterHub::runtime() {
    return *runtime_cfg_;
}

PipelineConfig& ParameterHub::pipeline() {
    return *pipeline_cfg_;
}

ModelConfig& ParameterHub::model() {
    return *model_cfg_;
}

} // namespace laser_aim::modules::config
