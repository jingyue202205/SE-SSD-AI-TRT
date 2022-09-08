#ifndef _VOXEL_GENERATOR_H_
#define _VOXEL_GENERATOR_H_


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
    class VoxelGeneratorPlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            VoxelGeneratorPlugin() = delete;
            VoxelGeneratorPlugin(int max_voxels, int max_points, int voxel_features, float x_min,
              float x_max, float y_min, float y_max, float z_min, float z_max,
              float voxel_x, float voxel_y, float voxel_z);
            VoxelGeneratorPlugin(int max_voxels, int max_points, int voxel_features, float x_min,
              float x_max, float y_min, float y_max, float z_min, float z_max,
              float voxel_x, float voxel_y, float voxel_z, int point_features,
              int grid_x, int grid_y, int grid_z);
            VoxelGeneratorPlugin(const void* data, size_t length);
    
            // ~VoxelGeneratorPlugin() override;

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
              // Shape Num for *input*
            int voxelNum_; // 20000
            int pointNum_;  // 5
            int featureNum_;  // 4
            float min_x_range_;
            float max_x_range_;
            float min_y_range_;
            float max_y_range_;
            float min_z_range_;
            float max_z_range_;
            float voxel_x_size_;  // 0.05
            float voxel_y_size_;  // 0.05
            float voxel_z_size_;  // 0.1
            // feature number of pointcloud points: 4 or 5
            int pointFeatureNum_;  // 4
            int grid_x_size_;
            int grid_y_size_;
            int grid_z_size_;
            
            
};

class VoxelGeneratorPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        VoxelGeneratorPluginCreator();
        ~VoxelGeneratorPluginCreator() override;
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
        int *max_num_points_per_voxel_ptr = nullptr;
        int *max_voxels_ptr = nullptr;
        float *voxel_size_ptr = nullptr;
        float *point_cloud_range_ptr = nullptr;
        int *voxel_feature_num_ptr = nullptr;
};
REGISTER_TENSORRT_PLUGIN(VoxelGeneratorPluginCreator);
};

#endif 
