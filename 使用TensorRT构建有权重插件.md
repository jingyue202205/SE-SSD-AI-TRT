## 如何使用TensorRT构建有权重插件

深度学习网络一般包含两类层，一类是没有卷积权重等可变参数的层，比如激活层，池化层等,这里我简称为(without parameters,w/o_p_layer)。另一种是带有权重等可变参数的层，比如卷积层，批正则化层等，简称为w_p_layer。网上，w/o_p_layer相关的教程特别多，但w_p_layer相关的教程非常少。

如果你对TensorRT插件不熟悉，阅读本文内容前，可以先阅读相关资料

实现TensorRT自定义插件(plugin)自由！

https://zhuanlan.zhihu.com/p/297002406 

英文版

Realize tensorrt-7.0 plug-in freedom! (if you don’t step on the pit, use tensorrt plug-in function) 

https://developpaper.com/realize-tensorrt-7-0-plug-in-freedom-if-you-dont-step-on-the-pit-use-tensorrt-plug-in-function/



#############################################################################

### 认真阅读完以上内容后，那我们开始今天的内容吧：

#### 使用TensorRT 构建有权重插件

> submConv3dlayer.cu
>
> submConv3dlayer.h

上述文件实现了3D子流型稀疏卷积的有权重TensorRT插件，我以此代码为例子，讲解如何构建有权重插件

#### 先看看submConv3dlayer.h中的内容

```c++
#ifndef _SUBM_CONV3D_LAYER_H_
#define _SUBM_CONV3D_LAYER_H_

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
    class SubmConv3dLayerPlugin: public nvinfer1::IPluginV2DynamicExt
    {
        public:
            SubmConv3dLayerPlugin() = delete;
            // SubmConv3dLayerPlugin(int in_channel, int out_channel,int max_voxels, int feature_num, int out_shape_z,int out_shape_y,int out_shape_x,
            //                         int spatial_shape_z, int spatial_shape_y, int spatial_shape_x,int ksize,
            //                         int stride, int padding, int dilation, int out_padding, const std::vector<float> & weights);
            SubmConv3dLayerPlugin(int in_channel, int out_channel,int max_voxels, int feature_num, int out_shape_z,int out_shape_y,int out_shape_x,
                                    int spatial_shape_z, int spatial_shape_y, int spatial_shape_x,int ksize,
                                    int stride, int padding, int dilation, int out_padding, int weights_size, nvinfer1::Weights const& weights);
            SubmConv3dLayerPlugin(const void* data, size_t length);
    
            ~SubmConv3dLayerPlugin() override;

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
              
            int in_channel_;
            int out_channel_;
            
            int max_voxels_; // 20000
            int feature_num_;  // 4
            int out_shape_z_;
            int out_shape_y_;
            int out_shape_x_;
            int spatial_shape_z_;
            int spatial_shape_y_;
            int spatial_shape_x_;
            int ksize_;
            int stride_;
            int padding_;
            int dilation_;
            int out_padding_;
            int weights_size_;

            float* weights_data_ = nullptr;

            float *weights_dev_ = nullptr;

            nvinfer1::Weights weights_{DataType::kFLOAT, nullptr, 0};
            
            
};

class SubmConv3dLayerPluginCreator : public nvinfer1::IPluginCreator
{
    public:
        SubmConv3dLayerPluginCreator();
        ~SubmConv3dLayerPluginCreator() override;
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
        int *out_shape = nullptr;
        int *spatial_shape = nullptr;
        int *ksize = nullptr;
        int *stride = nullptr;
        int *padding = nullptr;
        int *dilation = nullptr;
        int *out_padding = nullptr;
        float *d_weights = nullptr;  //device gpu
        std::vector<float> h_weights; // host

};
REGISTER_TENSORRT_PLUGIN(SubmConv3dLayerPluginCreator);
};

#endif 

```

submConv3dlayer.h头文件中包含两个类，一个类是**SubmConv3dLayerPlugin**

```c++
class SubmConv3dLayerPlugin: public nvinfer1::IPluginV2DynamicExt
```

**SubmConv3dLayerPlugin**是插件本身的定义，定义了插件的属性和方法。

```c++
            std::string mNamespace;
            int in_channel_;
            int out_channel_;
            
            int max_voxels_; // 20000
            int feature_num_;  // 4
            int out_shape_z_;
            int out_shape_y_;
            int out_shape_x_;
            int spatial_shape_z_;
            int spatial_shape_y_;
            int spatial_shape_x_;
            int ksize_;
            int stride_;
            int padding_;
            int dilation_;
            int out_padding_;
            int weights_size_;

            float* weights_data_ = nullptr;  //cpu memory

            float *weights_dev_ = nullptr;  // gpu memory

            nvinfer1::Weights weights_{DataType::kFLOAT, nullptr, 0}; // cpu memory
```

插件的属性共20个，其中最后三个是权重参数，weiget_data_保存在cpu上，weight_dev_保存在gpu上。weights_在cpu上，用于接收SubmConv3dLayerPluginCreator传递过来的权重参数。

**SubmConv3dLayerPlugin**中的重要方法：

```c++
SubmConv3dLayerPlugin(int in_channel, int out_channel,int max_voxels, int feature_num, int 						out_shape_z,int out_shape_y,int out_shape_x,
                 int spatial_shape_z, int spatial_shape_y, int spatial_shape_x,int ksize,
                 int stride, int padding, int dilation, int out_padding, int weights_size,                 nvinfer1::Weights const& weights); // very important，
			   //此函数会被SubmConv3dLayerPlugin::clone()和   SubmConv3dLayerPluginCreator::createPlugin()调用，用于生成插件。形参与插件的属性一一对应，并一对一赋值。
SubmConv3dLayerPlugin(const void* data, size_t length);//very important, 用于读取序列化的数据
 // IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;//very important, clone 调用构造函数
nvinfer1::DimsExprs getOutputDimensions(int outputIndex, 
           const nvinfer1::DimsExprs* inputs, int nbInputs,
          nvinfer1::IExprBuilder& exprBuilder) noexcept override; //very important, 指定每一个输出的维度
bool supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, 
    int nbInputs, int nbOutputs) noexcept override; // very important
            
size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
   const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept override;// very important
int enqueue(const nvinfer1::PluginTensorDesc* inputDesc, 
            const nvinfer1::PluginTensorDesc* outputDesc,
            const void* const* inputs, void* const* outputs, 
            void* workspace, cudaStream_t stream) noexcept override; // very important, 插件真正处理数据的地方，类似与pytorch module的forward,里面的所有变量保存在gpu上，无法打印，只有copytocpu后才能打印。可以用cout打印。
// IPluginV2Ext Methods
nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, 
                                     int nbInputs) const noexcept override; // very important，指定每一个输出的dtype
// IPluginV2 Methods
const char* getPluginType() const noexcept override;//very important
const char* getPluginVersion() const noexcept override;//very important
int getNbOutputs() const noexcept override; // very important,返回插件输出个数
int initialize() noexcept override;
void terminate() noexcept override; // very important，其被析构函数调用，作用释放分配的内存
size_t getSerializationSize() const noexcept override; // very important 返回要序列化的数据的size,包含权重参数的大小，以及插件的属性的大小
void serialize(void* buffer) const noexcept override; // 序列化插件的属性以及权重参数
void destroy() noexcept override; // very important
void setPluginNamespace(const char* pluginNamespace) noexcept override; // very important 
const char* getPluginNamespace() const noexcept override; // very important
```

另一个类是**SubmConv3dLayerPluginCreator**

```c++
class SubmConv3dLayerPluginCreator : public nvinfer1::IPluginCreator
```

**SubmConv3dLayerPluginCreator**的作用是做一些生成插件前的准备工作，比如读取配置参数、读取权重参数等

```c++
        static nvinfer1::PluginFieldCollection mFC;
        static std::vector<nvinfer1::PluginField> mPluginAttributes;
        std::string mNamespace;
        int *out_shape = nullptr;
        int *spatial_shape = nullptr;
        int *ksize = nullptr;
        int *stride = nullptr;
        int *padding = nullptr;
        int *dilation = nullptr;
        int *out_padding = nullptr;
        float *d_weights = nullptr;  //device gpu
        std::vector<float> h_weights; // host

```

**SubmConv3dLayerPluginCreator**的属性用于保存插件的配置参数和权重参数

**SubmConv3dLayerPluginCreator**的所有方法都是为生成插件服务的，以下重要方法定义。

```c++
SubmConv3dLayerPluginCreator(); // very important 初始化PluginField列表
const char* getPluginName() const noexcept override; // very important
const char* getPluginVersion() const noexcept override; // very important
const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override; // very important
nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override; // very important,从PluginFiled中加载参数和权重，并调用SubmConv3dLayerPlugin的第一个构造函数，生成插件
nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override; //very import 使用反序列的参数和权重，调用SubmConv3dLayerPlugin的第二个构造函数，生成插件
void setPluginNamespace(const char* pluginNamespace) noexcept override; // very important
const char* getPluginNamespace() const noexcept override; // very important
```

#### submConv3dlayer.cu中的内容

**SubmConv3dLayerPlugin**中的方法具体实现

```c++
SubmConv3dLayerPlugin::SubmConv3dLayerPlugin(
        int in_channel, int out_channel,int max_voxels, int feature_num, int out_shape_x,int out_shape_y,int out_shape_z,
        int spatial_shape_x, int spatial_shape_y, int spatial_shape_z,int ksize,
        int stride, int padding, int dilation, int out_padding, int weights_size, nvinfer1::Weights const& weights
) : in_channel_(in_channel), out_channel_(out_channel),max_voxels_(max_voxels), feature_num_(feature_num), 
    out_shape_x_(out_shape_x),out_shape_y_(out_shape_y),out_shape_z_(out_shape_z),
        spatial_shape_x_(spatial_shape_x), spatial_shape_y_(spatial_shape_y), spatial_shape_z_(spatial_shape_z),ksize_(ksize),
        stride_(stride), padding_(padding), dilation_(dilation), out_padding_(out_padding),weights_size_(weights_size)
{
    // std::cout << "构造函数 start" << std::endl;
    weights_data_ = (float*)malloc(weights_size_*sizeof(float));
    const float* temp_values = (const float*)weights.values;
    for(int i = 0;i < weights_size_;i++)
    {
        weights_data_[i] = temp_values[i];
    }
    weights_.count = weights.count;
    weights_.values = weights_data_;

   
    //  cudamalloc for conv
    checkCudaErrors(cudaMalloc(&weights_dev_,sizeof(float)*ksize_*ksize_*ksize_*in_channel_*out_channel_));
    
    // copy to gpu
 checkCudaErrors(cudaMemcpy(weights_dev_,weights_data_,sizeof(float)*ksize_*ksize_*ksize_*in_channel_*out_channel_,cudaMemcpyHostToDevice));
   
}
```

weights_dev_的权重参数保存在gpu上。用于在enqueue函数中做卷积。

weights_中的权重参数保存在cpu上，用于中clone函数，生成插件。



************************************************************

```c++
SubmConv3dLayerPlugin::SubmConv3dLayerPlugin(const void* data, size_t length)
{   
    const char* d = reinterpret_cast<const char*>(data);
    in_channel_ = readFromBuffer<int>(d);
    out_channel_ = readFromBuffer<int>(d);
    max_voxels_ = readFromBuffer<int>(d);
    feature_num_ = readFromBuffer<int>(d);
    out_shape_x_ = readFromBuffer<int>(d);
    out_shape_y_ = readFromBuffer<int>(d);
    out_shape_z_ = readFromBuffer<int>(d);
    spatial_shape_x_ = readFromBuffer<int>(d);
    spatial_shape_y_ = readFromBuffer<int>(d);
    spatial_shape_z_ = readFromBuffer<int>(d);
    ksize_ = readFromBuffer<int>(d);
    stride_ = readFromBuffer<int>(d);
    padding_ = readFromBuffer<int>(d);
    dilation_ = readFromBuffer<int>(d);
    out_padding_ = readFromBuffer<int>(d);
    weights_size_ = readFromBuffer<int>(d);
    weights_.count = weights_size_;
    weights_data_ = (float *)malloc(weights_size_*sizeof(float));
    for(int i=0;i < weights_size_; i++) 
    {
        weights_data_[i] = readFromBuffer<float>(d);
        // std::cout << weights_data_[i] << std::endl;
    }
        
    weights_.values = weights_data_;
    // std::cout << "构造函数2 end" << std::endl;
    // std::cout << in_channel_ << "," << out_channel_ << "," << max_voxels_ << "," << feature_num_ << "," << out_shape_x_ << "," << 
    //             out_shape_y_ << "," << out_shape_z_ << "," << spatial_shape_x_ << "," << spatial_shape_y_ << "," << spatial_shape_z_ << 
    //             "," << ksize_ << "," << stride_ << "," << padding_ << "," << dilation_ << "," << out_padding_ << "," << weights_size_ << std::endl;
}
```

SubmConv3dLayerPlugin::SubmConv3dLayerPlugin(const void* data, size_t length)的作用是从反序列化数据指针中逐条读出数据给相应属性。卷积的权重参数也要按照这种方式读出来。



```c++
IPluginV2DynamicExt* SubmConv3dLayerPlugin::clone() const noexcept
{
    // std::cout << "clone    start" << std::endl;
    auto* plugin = new SubmConv3dLayerPlugin(in_channel_, out_channel_,max_voxels_, feature_num_, out_shape_x_,out_shape_y_,out_shape_z_,
        spatial_shape_x_, spatial_shape_y_, spatial_shape_z_,ksize_,stride_, padding_, dilation_, out_padding_, weights_size_, weights_);
    plugin->setPluginNamespace(mNamespace.c_str());
    // std::cout << "clone   end" << std::endl;
    return plugin;
}
```

IPluginV2DynamicExt* SubmConv3dLayerPlugin::clone() 使用配置参数和卷积权重参数clone插件



```c++
nvinfer1::DimsExprs SubmConv3dLayerPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // auto batch_size = 1;
    // std::cout  << inputs[0].nbDims << " " << inputs[0].d[0]->getConstantValue() << " " << inputs[0].d[1]->getConstantValue() << " " << inputs[0].d[2]->getConstantValue() << std::endl;
    // std::cout  << inputs[1].nbDims << " " << inputs[1].d[0]->getConstantValue() << std::endl;
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(max_voxels_);
        dim0.d[2] = exprBuilder.constant(out_channel_);
        return dim0; 
    }
   
}
```

nvinfer1::DimsExprs SubmConv3dLayerPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) 作用是给每一个输出设置维度，因为3D 子流行卷积就一个输出，因此只写了outputIndex == 0的情况。dim0.nbDims = 3表示，输出维度为3维，具体维度为{batch_size,max_voxels_,out_channel_}



```c++
bool SubmConv3dLayerPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // voxelfeatures --- x, y, z, i   dim: 1  20000 4
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // coors 1 20000 4
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // voxel_num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // out_voxel_feature, dim: 1 x 20000 x out_channel
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}
```

bool SubmConv3dLayerPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs)指定输入输出的数据格式，submConv3dLayerPlugin一共有三个输入，一个输出，因此，pos的值由0到3。第一个输入为voxel_features，类型为nvinfer1::DataType::kFLOAT第二个输入为coords，类型为nvinfer1::DataType::kINT32,第三个输入为voxel_num，表示有多少需要处理的有效数据行数，为了不造成数据溢出，voxel_features和coords的行数设置的比较大。类型为nvinfer1::DataType::kINT32。输出为output_voxel_features，类型为nvinfer1::DataType::kFLOAT.



```c++
size_t SubmConv3dLayerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
  
    int dense_voxel_num = out_shape_x_ * out_shape_y_ * out_shape_z_;
    size_t grids_out_size = batchSize * dense_voxel_num * sizeof(int);  // 41*1600*1408 -1
    size_t indice_pairs_size = batchSize * ksize_ * ksize_ * ksize_ * MAX_VOXELS * 2 * sizeof(int);    // 27,2,16825 -1
    size_t indice_num_size = batchSize * ksize_ * ksize_ * ksize_ * sizeof(unsigned int); // 27 ,0
    size_t valid_points_size = batchSize * MAX_VOXELS * 27 *4 * sizeof(unsigned int); // 20000*27*4
    size_t valid_points_num_size = batchSize * MAX_VOXELS * sizeof(unsigned int);// 20000*1;
    
    size_t workspaces[5];
    workspaces[0] = grids_out_size;
    workspaces[1] = indice_pairs_size;
    workspaces[2] = indice_num_size;
    workspaces[3] = valid_points_size;
    workspaces[4] = valid_points_num_size;
    
    return  calculateTotalWorkspaceSize(workspaces, 5);
}
```

size_t SubmConv3dLayerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,const nvinfer1::PluginTensorDesc* outputs, int nbOutputs)作用是为enqueue函数中用到的临时变量分配显存。enqueue中使用的变量不能通过cudamalloc申请，必须先通过这个函数分配好显存。



```c++
int SubmConv3dLayerPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,cudaStream_t stream) noexcept
{
    // std::cout << "enqueue start" << std::endl;
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    // std::cout << "batch_size: " << batchSize << std::endl;
    float *input_voxel_features = const_cast<float *>((const float*)inputs[0]);
    unsigned int *input_coords = const_cast<unsigned int *>((const unsigned int*)inputs[1]);
    unsigned int * voxel_num = const_cast<unsigned int *>((const unsigned int *)inputs[2]);
    //TRT-output
    float *output_voxel_features = (float *)(outputs[0]);  // 1 * 20000 * 16
    // unsigned int *output_coords = (unsigned int *)(outputs[1]); // 1* 20000*4
    // unsigned int *params_data = (unsigned int *)(outputs[2]);
    int dense_voxel_num = out_shape_x_ * out_shape_y_ * out_shape_z_;
    size_t grids_out_size = batchSize * dense_voxel_num * sizeof(int);  // 41*1600*1408 -1
    size_t indice_pairs_size = batchSize * ksize_ * ksize_ * ksize_ * MAX_VOXELS * 2 * sizeof(int);    // 27,2,16825 -1
    size_t indice_num_size = batchSize * ksize_ * ksize_ * ksize_ * sizeof(unsigned int); // 27 ,0
    size_t valid_points_size = batchSize * MAX_VOXELS * 27 * 4 * sizeof(unsigned int); // 20000*256*4
    size_t valid_points_num_size = batchSize * MAX_VOXELS * sizeof(unsigned int);// 20000*1;
    size_t workspaces[5];
    workspaces[0] = grids_out_size;
    workspaces[1] = indice_pairs_size;
    workspaces[2] = indice_num_size;
    workspaces[3] = valid_points_size;
    workspaces[4] = valid_points_num_size;

    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 5);
    // std::cout << "enqueue11111" << std::endl;
    int* grids_out = static_cast<int*>(workspace);
    // unsigned int* coor_to_voxelidx = static_cast<unsigned int*>(workspace);
    int* indice_pairs = reinterpret_cast<int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(grids_out), grids_out_size));
    unsigned int* indice_num = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(indice_pairs), indice_pairs_size));
    unsigned int* valid_points = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(indice_num), indice_num_size));
    unsigned int* valid_points_num = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(valid_points), valid_points_size));
 
    // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(grids_out, -1, grids_out_size, stream)); // total_workspace
    checkCudaErrors(cudaMemsetAsync(indice_pairs, -1, indice_pairs_size, stream)); // total_workspace
    checkCudaErrors(cudaMemsetAsync(indice_num, 0, indice_num_size, stream)); // total_workspace
    checkCudaErrors(cudaMemsetAsync(valid_points, 0, valid_points_size, stream)); // total_workspace
    checkCudaErrors(cudaMemsetAsync(valid_points_num, 0, valid_points_num_size, stream)); // total_workspace

   
    // FOR tensorrt output
    unsigned int output_voxel_features_size = batchSize * max_voxels_  * out_channel_ * sizeof(float);
    
    checkCudaErrors(cudaMemsetAsync(output_voxel_features, 0, output_voxel_features_size, stream));

    checkCudaErrors(prepareSubMGridKernel_launch(
          input_coords,grids_out,voxel_num,out_shape_x_,out_shape_y_,out_shape_z_, stream));

    checkCudaErrors(getValidOutPosKernel_launch(input_coords, valid_points,valid_points_num,
        voxel_num,ksize_,stride_, padding_, dilation_, out_shape_x_,out_shape_y_, out_shape_z_,stream));

    checkCudaErrors(getSubMIndicePairsKernel_launch(input_coords,grids_out,indice_pairs,indice_num,valid_points,valid_points_num, voxel_num,
                    out_shape_x_,out_shape_y_,out_shape_z_,stream));

    int kernel_volume = ksize_ * ksize_ * ksize_;
  
    for(int i = 0; i < kernel_volume; i++)
    {
        checkCudaErrors(sparseGatherConvScatter_launch(input_voxel_features,indice_pairs,indice_num,output_voxel_features,i,weights_dev_,
                                                        in_channel_,out_channel_,stream));
    }

    return 0;
}

```

int SubmConv3dLayerPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,cudaStream_t stream)类似于pytorch moudle里的forward，算法真正运行的地方。整个流程大致可以分为：

1. 获取TRT-input
2. 获取TRT-output 
3.  为临时变量申请显存
4. 初始化临时变量
5. 初始化TRT-output
6. 调用kernel函数对上述数据进行处理



**当前的实现速度比较慢，如果你想优化的话可以先从enqueue的kernel函数入手，看看那个kernel函数运行比较慢，就自己写一个速度更快的代替**



```c++
nvinfer1::DataType SubmConv3dLayerPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // if(index == 0)
    return inputTypes[0]; //只有一个输出，类型为KFLOAT，跟第一个输入类似，因此返回inputTypes[0]
    // return inputTypes[1];
}

const char* SubmConv3dLayerPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* SubmConv3dLayerPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int SubmConv3dLayerPlugin::getNbOutputs() const noexcept
{
    return 1; // 只有一个输出因此为1
}
```

```c++
size_t SubmConv3dLayerPlugin::getSerializationSize() const noexcept
{
    return ksize_*ksize_*ksize_*in_channel_*out_channel_ * sizeof(float) + 16 * sizeof(int); // 获取序列化数据的size，包括配置参数和卷积权重参数
}
```

```c++
void SubmConv3dLayerPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);

    writeToBuffer<int>(d, in_channel_);
    writeToBuffer<int>(d, out_channel_);
    writeToBuffer<int>(d, max_voxels_);
    writeToBuffer<int>(d, feature_num_);
    writeToBuffer<int>(d, out_shape_x_);
    writeToBuffer<int>(d, out_shape_y_);
    writeToBuffer<int>(d, out_shape_z_);
    writeToBuffer<int>(d, spatial_shape_x_);
    writeToBuffer<int>(d, spatial_shape_y_);
    writeToBuffer<int>(d, spatial_shape_z_);
    writeToBuffer<int>(d, ksize_);
    writeToBuffer<int>(d, stride_);
    writeToBuffer<int>(d, padding_);
    writeToBuffer<int>(d, dilation_);
    writeToBuffer<int>(d, out_padding_);
    writeToBuffer<int>(d, weights_size_);
    const float * data = (const float*)weights_.values;
    for(int i=0; i < weights_size_; i++)
    {
        writeToBuffer<float>(d,data[i]);
    }
}
```

void SubmConv3dLayerPlugin::serialize(void* buffer) 的作用是把配置参数和卷积权重参数序列化到engine文件中



##############################################################################

**SubmConv3dLayerPluginCreator**中的方法具体实现

```c++
SubmConv3dLayerPluginCreator::SubmConv3dLayerPluginCreator()
{
    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("in_channel", nullptr, PluginFieldType::kINT32, 1));  // 4
    mPluginAttributes.emplace_back(PluginField("out_channel", nullptr, PluginFieldType::kINT32, 1));  //16
    mPluginAttributes.emplace_back(PluginField("max_voxels", nullptr, PluginFieldType::kINT32, 1));  //16825
    mPluginAttributes.emplace_back(PluginField("feature_num", nullptr, PluginFieldType::kINT32, 1));  //4
    mPluginAttributes.emplace_back(PluginField("out_shape", nullptr, PluginFieldType::kINT32, 1)); // [41,1600,1408]
    mPluginAttributes.emplace_back(PluginField("spatial_shape", nullptr, PluginFieldType::kINT32, 1)); // [41,1600,1408]
    mPluginAttributes.emplace_back(PluginField("ksize", nullptr, PluginFieldType::kINT32, 1));  // [3,3,3]
    mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1)); // [1,1,1]
    mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1)); // [0,0,0]
    mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1)); // [1,1,1]
    mPluginAttributes.emplace_back(PluginField("out_padding", nullptr, PluginFieldType::kINT32, 1)); //[0,0,0]
    mPluginAttributes.emplace_back(PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1));  // 3x3x4x16


    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
```

SubmConv3dLayerPluginCreator::SubmConv3dLayerPluginCreator()作用是初始化化pluginfieldcollection，每一个pluginfield包含名字，地址指针，数据类型，个数等，以PluginField("weights", nullptr, PluginFieldType::kFLOAT32, 1)为例，name为"weights",指针初始化为nullptr。



```c++
IPluginV2* SubmConv3dLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    // std::cout << "createplugin start <<<<<<<<<<<<<<<<<<<<<<" << std::endl;
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    // std::cout << nbFields << std::endl;

    int in_channel = 0;
    int out_channel = 0;
    int max_voxels = 0;
    int feature_num = 0;
    int out_shape_x = 0;
    int out_shape_y = 0;
    int out_shape_z = 0;
    int spatial_shape_x = 0;
    int spatial_shape_y = 0;
    int spatial_shape_z = 0;
    int ksize = 0;
    int stride = 0;
    int padding = 0;
    int dilation = 0;
    int out_padding = 0;
    int weights_size = 0;
    const float *weight;
    // std::cout << fields[0].name << std::endl;
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name =  fields[i].name;
        // std::cout << " createplugn <<" << attr_name << std::endl;
        if (!strcmp(attr_name, "in_channel"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            in_channel = d[0];
            // std::cout << "in_channel <<<<" << in_channel << std::endl;
        }
        else if (!strcmp(attr_name, "out_channel"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            out_channel = d[0];
            // std::cout << "out_channel <<<<" << out_channel << std::endl;
        }
        else if (!strcmp(attr_name, "max_voxels"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_voxels = d[0];
        }
          else if (!strcmp(attr_name, "feature_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_num = d[0];
        }

        else if (!strcmp(attr_name, "out_shape"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            out_shape_x = d[0];
            out_shape_y = d[1];
            out_shape_z = d[2];
        }
        else if (!strcmp(attr_name, "spatial_shape"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            spatial_shape_x = d[0];
            spatial_shape_y = d[1];
            spatial_shape_z = d[2];
        }
        else if (!strcmp(attr_name, "ksize"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            ksize = d[0];
        }
        else if (!strcmp(attr_name, "stride"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            stride = d[0];
        }
        else if (!strcmp(attr_name, "padding"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            padding = d[0];
        }
        else if (!strcmp(attr_name, "dilation"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            dilation = d[0];
        }
        else if (!strcmp(attr_name, "out_padding"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            out_padding = d[0];
        }
        else if (!strcmp(attr_name, "weights"))
        {
            const float * d = static_cast<const float*>(fields[i].data);
            weight = d;
        }
    }
    weights_size = in_channel * out_channel * ksize * ksize * ksize;
    Weights wt{DataType::kFLOAT, nullptr, 0};
    wt.count = weights_size;
    wt.values = weight;
    // std::cout << max_voxels << " " << in_channel << " " <<out_channel << " " << feature_num << " " << out_shape_x << " "
    // << out_shape_y << " "<< out_shape_z << " " << spatial_shape_x << " " << spatial_shape_y << " " << spatial_shape_z << " "
    // << ksize << " " << stride << " " << padding << " " << dilation << " " << out_padding << " " << std::endl;
    
    IPluginV2DynamicExt* plugin = new SubmConv3dLayerPlugin(in_channel, out_channel,max_voxels, feature_num, out_shape_x,out_shape_y,out_shape_z,
                                    spatial_shape_x, spatial_shape_y, spatial_shape_z,ksize,
                                    stride, padding, dilation, out_padding, weights_size, wt);
    return plugin;
}
```

IPluginV2* SubmConv3dLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)的作用是从PluginFieldCollection中读取参数，然后通过这些参数构建插件，并把构建的插件返回。



```c++
IPluginV2* SubmConv3dLayerPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new SubmConv3dLayerPlugin(serialData, serialLength); //通过反序列的参数初始化插件
}

void SubmConv3dLayerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SubmConv3dLayerPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}
```

### 如何调用上述代码生成插件

se-ssd-ai-trt.cpp中有个函数用于生成subconv3dlayer插件

```c++
IPluginV2Layer* add_subm_conv3d_layer(INetworkDefinition *network,ITensor* voxel_features,ITensor* coors,ITensor* voxel_num, nvinfer1::Weights const& weights,
                      int max_voxels, int in_channel, int out_channel, int out_shape_x,
                      int out_shape_y, int out_shape_z)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));//生成新的pluginFieldCollection,后面会把新的pluginFieldCollection传递给SubmConv3dLayerPluginCreator::createPlugin
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    int *out_shape = (int*)malloc(3*sizeof(int));
    out_shape[0] = out_shape_x;
    out_shape[1] = out_shape_y;
    out_shape[2] = out_shape_z;

    int ksize = 3;
    int stride = 1;
    int padding = 1;
    int dilation = 1;
    int out_padding = 0;

    auto submConv3dLayercreator = getPluginRegistry()->getPluginCreator("SubmConv3dLayerPlugin", "1"); // 得到submConv3dLayercreator
    const PluginFieldCollection* submConv3dLayerpluginData = submConv3dLayercreator->getFieldNames(); // 返回SubmConv3dLayerPluginCreator::SubmConv3dLayerPluginCreator()初始化的pluginFieldCollection，获取这个pluginFieldcollection的目的是从中得到pluginField的名字

    const PluginField* fields = submConv3dLayerpluginData->fields;
    int nbFields = submConv3dLayerpluginData->nbFields;
    
    for (int i = 0; i < nbFields; ++i) //遍历pluginfieldcollection，读取name，并以此name和对应指针生成新的pluginField，加载到新的pluginFieldCollection中
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "in_channel"))
        {
            
            new_pluginData_list.emplace_back(PluginField("in_channel",  &(in_channel), PluginFieldType::kINT32, 1));
            
            
        }
        else if (!strcmp(attr_name, "out_channel"))
        {
            
            new_pluginData_list.emplace_back(PluginField("out_channel",  &(out_channel), PluginFieldType::kINT32, 1)); 
           
        }
        else if (!strcmp(attr_name, "max_voxels"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_voxels",  &(max_voxels), PluginFieldType::kINT32, 1));  
          
        }
          else if (!strcmp(attr_name, "feature_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("feature_num",  &(in_channel), PluginFieldType::kINT32, 1));  
           
        }

        else if (!strcmp(attr_name, "out_shape"))
        {
            

            new_pluginData_list.emplace_back(PluginField("out_shape",  out_shape, PluginFieldType::kINT32, 1));
            
        }
        else if (!strcmp(attr_name, "spatial_shape"))
        {
           

            new_pluginData_list.emplace_back(PluginField("spatial_shape",  out_shape, PluginFieldType::kINT32, 1));
           
        }
        else if (!strcmp(attr_name, "ksize"))
        {
            
            new_pluginData_list.emplace_back(PluginField("ksize",  &(ksize), PluginFieldType::kINT32, 1)); 
           
        }
        else if (!strcmp(attr_name, "stride"))
        {
           
            new_pluginData_list.emplace_back(PluginField("stride",  &(stride), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "padding"))
        {
           
            new_pluginData_list.emplace_back(PluginField("padding",  &(padding), PluginFieldType::kINT32, 1)); 
           
        }
        else if (!strcmp(attr_name, "dilation"))
        {
           
            new_pluginData_list.emplace_back(PluginField("dilation",  &(dilation), PluginFieldType::kINT32, 1)); 
          
        }
        else if (!strcmp(attr_name, "out_padding"))
        {
            
            new_pluginData_list.emplace_back(PluginField("out_padding",  &(out_padding), PluginFieldType::kINT32, 1));
            
        }
        else if (!strcmp(attr_name, "weights"))
        {
            
            new_pluginData_list.emplace_back(PluginField("weights", weights.values, PluginFieldType::kFLOAT32, 1));  // 3x3x4x16
          
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

   

    IPluginV2 *pluginObj_submConv3dLayer = submConv3dLayercreator->createPlugin("submConv3dLayer", newPluginFieldCollection); //使用新的pluginFieldCollection,生成插件，前一个参数是为插件起的名字，后一个参数是，新的pluginFieldCollection
    ITensor* inputTensors_submConv3dLayer[] = {voxel_features,coors,voxel_num};//插件的输入
    auto submConv3dLayer = network->addPluginV2(inputTensors_submConv3dLayer, 3, *pluginObj_submConv3dLayer);//把插件加入到网络中，3表示，此插件有三个输入，
    pluginObj_submConv3dLayer->destroy();//销毁新生成的插件，已经没用了
    free(out_shape);//释放参数空间
    free(newPluginFieldCollection);//释放新的pluginFieldCollection
    return submConv3dLayer;
}
```

函数的调用

```c++
auto submConv3dLayer0 = add_subm_conv3d_layer(network,voxelGenerator->getOutput(0),voxelGenerator->getOutput(1),voxelGenerator->getOutput(2),
                                    weightMap["backbone.middle_conv.0.weight"],   SUBM_0_MAX_VOXELS,SUBM_0_IN_CHANNEL,SUBM_0_OUT_CHANNEL,SUBM_0_OUT_SHAPE_X,SUBM_0_OUT_SHAPE_Y,SUBM_0_OUT_SHAPE_Z);
    
```

