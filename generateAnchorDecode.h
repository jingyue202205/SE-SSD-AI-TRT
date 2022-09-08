#ifndef _GENERATE_ANCHOR_DECODE_H_
#define _GENERATE_ANCHOR_DECODE_H_


#include "NvInferPlugin.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
// #include <cmath>
#include "params.h"

using namespace std;

namespace nvinfer1
{
    class GenerateAnchorDecodePlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            GenerateAnchorDecodePlugin() = delete;
   
            GenerateAnchorDecodePlugin(float min_x_range,float max_x_range,float min_y_range,float max_y_range,float min_z_range,
                                    float max_z_range, int feature_map_height, int feature_map_width, float car_length,
                                    float car_width, float car_height, float direction_angle_0, float direction_angle_1,
                                    int direction_angle_num);
            GenerateAnchorDecodePlugin(const void* data, size_t length);
    
            // ~GenerateAnchorDecodePlugin() override;

            // IPluginV2DynamicExt Methods
            nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
            nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
                const nvinfer1::DimsExprs* inputs, int nbInputs,
                nvinfer1::IExprBuilder& exprBuilder) noexcept override;
            bool supportsFormatCombination(
                int pos, const nvinfer1::PluginTensorDesc* inOut, 
                int nbInputs, int nbOutputs) noexcept override;
            void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
                const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept override;
            size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
                const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;
            int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
                const nvinfer1::PluginTensorDesc* outputDesc,
                const void* const* inputs, void* const* outputs, 
                void* workspace, cudaStream_t stream) noexcept override;
            // IPluginV2Ext Methods
            nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
                int nbInputs) const noexcept override;
            // IPluginV2 Methods
            const char* getPluginType() const noexcept override;
            const char* getPluginVersion() const noexcept override;
            int getNbOutputs() const noexcept override;
            int initialize() noexcept override;
            void terminate() noexcept override;
            size_t getSerializationSize() const noexcept override;
            void serialize(void* buffer) const noexcept override;
            void destroy() noexcept override;
            void setPluginNamespace(const char* pluginNamespace) noexcept override;
            const char* getPluginNamespace() const noexcept override;
        private:
            std::string mNamespace;
            float min_x_range_;
            float max_x_range_;
            float min_y_range_;
            float max_y_range_;
            float min_z_range_;
            float max_z_range_;
            int feature_map_height_;
            int feature_map_width_;
            float car_length_;
            float car_width_;
            float car_height_;
            float direction_angle_0_;
            float direction_angle_1_;
            int direction_angle_num_;

            

    
            
            
};

class GenerateAnchorDecodePluginCreator : public nvinfer1::IPluginCreator
{
    public:
        GenerateAnchorDecodePluginCreator();
        ~GenerateAnchorDecodePluginCreator() override;
        const char* getPluginName() const noexcept override;
        const char* getPluginVersion() const noexcept override;
        const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
        nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
        nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
        void setPluginNamespace(const char* pluginNamespace) noexcept override;
        const char* getPluginNamespace() const noexcept override;
    private:
        static nvinfer1::PluginFieldCollection mFC;
        static std::vector<nvinfer1::PluginField> mPluginAttributes;
        std::string mNamespace;
};
REGISTER_TENSORRT_PLUGIN(GenerateAnchorDecodePluginCreator);
};

#endif 
