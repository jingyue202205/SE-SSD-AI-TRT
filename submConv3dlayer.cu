#include "submConv3dlayer.h"


#include <fstream>
#include <iostream>
#include <iomanip> //设置输出格式
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "params.h"
#include <time.h>
#include <chrono>
#include <cmath>
#include <string>
#include <string.h>

using namespace std::chrono;
using std::string;

using namespace nvinfer1;
using nvinfer1::SubmConv3dLayerPlugin;
using nvinfer1::SubmConv3dLayerPluginCreator;
using namespace std;

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)



#define CUDA_MEM_ALIGN 256

static const char* PLUGIN_VERSION{"1"};
static const char* PLUGIN_NAME{"SubmConv3dLayerPlugin"};

// Static class fields initialization
PluginFieldCollection SubmConv3dLayerPluginCreator::mFC{};
std::vector<PluginField> SubmConv3dLayerPluginCreator::mPluginAttributes;

// Helper function for serializing plugin
template <typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// Helper function for deserializing plugin
template <typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

// Mimic np.round as in voxel generator in spconv implementation
int np_round(float x) {
  // half way round to nearest-even
  int x2 = int(x * 2.0f);
  if(x != int(x) && x2 == x * 2.0f) {
    return int(x / 2.0f + 0.5f) * 2;
  }
  return int(x + 0.5f);
}

// ALIGNPTR
int8_t* alignPtr(int8_t* ptr, uintptr_t to)
{
    uintptr_t addr = (uintptr_t) ptr;
    if (addr % to)
    {
        addr += to - addr % to;
    }
    return (int8_t*) addr;
}

// NEXTWORKSPACEPTR
int8_t* nextWorkspacePtr(int8_t* ptr, uintptr_t previousWorkspaceSize)
{
    uintptr_t addr = (uintptr_t) ptr;
    addr += previousWorkspaceSize;
    return alignPtr((int8_t*) addr, CUDA_MEM_ALIGN);
}

// CALCULATE TOTAL WORKSPACE SIZE
size_t calculateTotalWorkspaceSize(size_t* workspaces, int count)
{
    size_t total = 0;
    for (int i = 0; i < count; i++)
    {
        total += workspaces[i];
        if (workspaces[i] % CUDA_MEM_ALIGN)
        {
            total += CUDA_MEM_ALIGN - (workspaces[i] % CUDA_MEM_ALIGN);
        }
    }
    return total;
}

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

IPluginV2DynamicExt* SubmConv3dLayerPlugin::clone() const noexcept
{
    // std::cout << "clone    start" << std::endl;
    auto* plugin = new SubmConv3dLayerPlugin(in_channel_, out_channel_,max_voxels_, feature_num_, out_shape_x_,out_shape_y_,out_shape_z_,
        spatial_shape_x_, spatial_shape_y_, spatial_shape_z_,ksize_,stride_, padding_, dilation_, out_padding_, weights_size_, weights_);
    plugin->setPluginNamespace(mNamespace.c_str());
    // std::cout << "clone   end" << std::endl;
    return plugin;
}

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
        return dim0; // voxels 1 20000 4
    }
   
}

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

void SubmConv3dLayerPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t SubmConv3dLayerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
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



__global__ void getValidOutPosKernel(unsigned int *input_coords, unsigned int * valid_points,unsigned int* valid_points_num,
                                    unsigned int *voxel_num,int ksize,int stride, int padding, 
                                    int dilation, int out_shape_x, int out_shape_y, int out_shape_z)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxelidx >= *voxel_num) return;

    uint4 coords = ((uint4*)input_coords)[voxelidx];
    
    
    // getValidOutPos
    int NDim = 3;
    int coords_[3] = {coords.y,coords.z,coords.w};
    int ksize_[3] = {ksize,ksize,ksize};
    int stride_[3] = {stride,stride,stride};
    int padding_[3] = {padding,padding,padding};
    int dilation_[3] = {dilation,dilation,dilation};
    int out_shape[3] = {out_shape_z,out_shape_y,out_shape_x};
    int lowers[3] = {0,0,0};
    int uppers[3] = {0,0,0};
    int counter[3] = {0,0,0};
    int counter_size[3] = {0,0,0};
    int point_counter = 0;
    int val = 0;
    int num_points = 1;
    int m = 1;
    int offset = 0;
    int valid = 0;


    for (int i = 0; i < NDim; i++) {
        lowers[i] = (coords_[i] - (ksize_[i] - 1) * dilation_[i] - 1 +
                    stride_[i] + padding_[i]) / stride_[i];
        uppers[i] = (coords_[i] + padding_[i]) / stride_[i];
    }


    for (int i = 0; i < NDim; i++) {
        counter_size[i] = ((uppers[i] - lowers[i]) / dilation_[i] + 1);
        num_points *= counter_size[i];
    }

    
    // printf("numpoints:%d\n",numPoints);
    for (int i = 0; i < num_points; i++) {
        valid = 1;
        m = 1;
        offset = 0;
        for (int j = NDim - 1; j >= 0; j--) 
        { // 2,1,0
            val = uppers[j] - counter[j] * dilation_[j];
    
            atomicExch(valid_points+voxelidx*(27*4) + point_counter * 4 + j,val);

            if (val < 0 || (val > out_shape[j] - 1)) {
                valid = 0;
             }
            offset += m * (coords_[j] - val * stride_[j] + padding_[j]) / dilation_[j];
            m *= ksize_[j];
        }
        atomicExch(valid_points + voxelidx*(27*4) +point_counter * 4 + NDim,offset);
    // if(voxelidx == 0)
    //   {
    //       printf("offset:%d\n",offset);
    //   }
    //     if (coords.y == 38 && coords.z == 800 && coords.w == 366 && i == 1)
    // {
    //     printf("6666661111:%d,%d,%d,%d\n",*(valid_points+voxelidx*(256*4) + point_counter * 4 + 0),*(valid_points+voxelidx*(256*4) + point_counter * 4 + 1),
    //     *(valid_points+voxelidx*(256*4) + point_counter * 4 + 2),*(valid_points+voxelidx*(256*4) + point_counter * 4 + 3));
    // }
        if (valid) point_counter += 1;
        counter[NDim - 1] += 1;
        for (int c = NDim - 1; c >= 0; c--) 
        {
            if (counter[c] == counter_size[c] && c > 0) 
            {
            counter[c - 1] += 1;
            counter[c] = 0;
            }
   
        } 
    }
        atomicExch(valid_points_num+voxelidx,point_counter);
    //  if (coords.y == 38 && coords.z == 800 && coords.w == 366)
    // {
    //     printf("endendend:%d,%d,%d,%d\n",point_counter,counter_size[0],counter_size[1],counter_size[2]);
    // }
}


cudaError_t getValidOutPosKernel_launch(unsigned int *input_coords, unsigned int * valid_points,unsigned int* valid_points_num,
                                    unsigned int *voxel_num,int ksize,int stride, int padding, 
                                    int dilation, int out_shape_x, int out_shape_y, int out_shape_z,cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  getValidOutPosKernel<<<blocks, threads, 0, stream>>>
       (input_coords, valid_points,valid_points_num,
        voxel_num,ksize,stride, padding,dilation,out_shape_x, out_shape_y, out_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void getSubMIndicePairsKernel(unsigned int *input_coords, int * grids_out, int *indice_pairs, unsigned int *indice_num,
                                         unsigned int * valid_points, unsigned int * valid_points_num, unsigned int *voxel_num,int out_shape_x,
                                         int out_shape_y, int out_shape_z)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxelidx >= *voxel_num) return;

    // int out_volume = out_shape_x * out_shape_y * out_shape_z;

    // int num_valid_points = 0;
    // printf("voxel_num: %d\n",*voxel_num);

    int index = 0;
    
    uint4 coords = ((uint4*)input_coords)[voxelidx];
   
    int point_counter = valid_points_num[voxelidx];
    for (int i = 0; i < point_counter; ++i) 
    {
      unsigned int index_z = *(valid_points + voxelidx*(27*4)+i*4+0);
      unsigned int index_y = *(valid_points + voxelidx*(27*4)+i*4+1);
      unsigned int index_x = *(valid_points + voxelidx*(27*4)+i*4+2);
      unsigned int offset = *(valid_points + voxelidx*(27*4)+i*4+3);
    //    if (coords.y == 38 && coords.z == 800 && coords.w == 366)
    // {
    //     // printf("point_counter:%d\n",point_counter);
    //     printf("index_:%d,%d,%d,%d\n",index_z,index_y,index_x,offset);
    // }

      index = coords.x * out_shape_x * out_shape_y * out_shape_z + 
                index_z * out_shape_y * out_shape_x +
                index_y * out_shape_x + 
                index_x;


      if (grids_out[index] > -1)
       {
        unsigned int old_num = atomicAdd(indice_num + offset, 1);

        int * indice_pairs_address_1 = indice_pairs + offset*(2*MAX_VOXELS) + 1 * MAX_VOXELS + old_num;
        int old_1 = atomicExch(indice_pairs_address_1,grids_out[index]);  // output index

        int * indice_pairs_address_0 = indice_pairs + offset*(2*MAX_VOXELS) + 0 * MAX_VOXELS + old_num;
        int old_0 = atomicExch(indice_pairs_address_0,voxelidx);  // input index

       }
    }

}


cudaError_t getSubMIndicePairsKernel_launch(unsigned int *input_coords, int * grids_out, int *indice_pairs, unsigned int *indice_num,unsigned int * valid_points, unsigned int *valid_points_num,
                                    unsigned int *voxel_num,int out_shape_x,
                                         int out_shape_y, int out_shape_z,cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  getSubMIndicePairsKernel<<<blocks, threads, 0, stream>>>
       (input_coords, grids_out, indice_pairs, indice_num,valid_points,valid_points_num, voxel_num,out_shape_x,out_shape_y,out_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void prepareSubMGridKernel(unsigned int *input_coords, int* grids_out,
        unsigned int * voxel_num, unsigned int out_shape_x, unsigned int out_shape_y, unsigned int out_shape_z)
{
    // printf("voxel_num:%d\n",*voxel_num);
    // int batch_size = 1;
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *voxel_num) return;
    // printf("prepareSubMGridKernel");
    uint4 coords = ((uint4*)input_coords)[voxel_idx];
    // printf("x:%d,y:%d,z:%d,w:%d\n",coords.x,coords.y,coords.z,coords.w);
    // printf("spatial:%d,%d,%d\n",out_shape_z,out_shape_y,out_shape_x);
    unsigned int ix = coords.x * out_shape_x * out_shape_y * out_shape_z + 
                        coords.y * out_shape_y*out_shape_x +
                        coords.z * out_shape_x + 
                        coords.w; 
    // grids_out[ix] = voxel_idx;
    int oldnum = atomicExch(grids_out+ix,voxel_idx);
   
}

cudaError_t prepareSubMGridKernel_launch(unsigned int *input_coords, int* grids_out,unsigned int *voxel_num,
        unsigned int out_shape_x, unsigned int out_shape_y, unsigned int out_shape_z,cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

//   unsigned int pointNum_cpu = 0;
//   CHECK(cudaMemcpy(&pointNum_cpu, points_size, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//   std::cout << pointNum_cpu << std::endl;

  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  prepareSubMGridKernel<<<blocks, threads, 0, stream>>>
       (input_coords,grids_out,voxel_num,out_shape_x,out_shape_y,out_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void sparseGatherConvScatter(float* input_voxel_features,int *indice_pairs,unsigned int *indice_num,
                        float *output_voxel_features,int kernel_volume_index,float* weights_dev_,int in_channel, int out_channel)
{
    // printf("point_size:%d\n",*points_size);
    // int batch_size = 1;
    int act_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_act = indice_num[kernel_volume_index];

    // uint4 coords = ((uint4*)coors)[act_idx];
    //  printf("in_channel: %d,out_channel:%d\n",in_channel,out_channel);

    if(act_idx >= num_act) return;
    

    int input_index = *(indice_pairs + kernel_volume_index * 2*MAX_VOXELS + 0*MAX_VOXELS + act_idx);


    // // get kernel in act_idx positon
    int kernel_num = in_channel * out_channel;
    
    // get position for output
    int output_index = *(indice_pairs + kernel_volume_index * 2 * MAX_VOXELS + 1*MAX_VOXELS + act_idx);
  

    for(int i=0; i < out_channel; i++)
    {   
        float current_sum = 0.0;
        for(int j=0;j<in_channel;j++)
        {
            // current_sum += temp_input_features[j] * kernel_weights[i+j*out_channel];
            
            current_sum += input_voxel_features[input_index * in_channel + j] * weights_dev_[kernel_volume_index * kernel_num+i+j*out_channel];
        }
       
        // *(output_voxel_features + output_index * out_channel + i) += current_sum;
        float old_value = atomicAdd(output_voxel_features + output_index * out_channel + i,current_sum);

    }

}


cudaError_t sparseGatherConvScatter_launch(float* input_voxel_features,int *indice_pairs,unsigned int *indice_num,
                        float *output_voxel_features,int kernel_volume_index,float* weights_dev_,
                        int in_channel, int out_channel, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

//   unsigned int pointNum_cpu = 0;
//   CHECK(cudaMemcpy(&pointNum_cpu, points_size, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
//   std::cout << pointNum_cpu << std::endl;

  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  sparseGatherConvScatter<<<blocks, threads, 0, stream>>>
       (input_voxel_features,indice_pairs,indice_num,output_voxel_features,kernel_volume_index,weights_dev_,in_channel,out_channel);
  cudaError_t err = cudaGetLastError();
  return err;
}

int SubmConv3dLayerPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
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
    // unsigned int output_coords_size = batchSize * max_voxels_ * 4 * sizeof(unsigned int);
    // unsigned int voxel_num_data_size = batchSize * sizeof(unsigned int);
    // unsigned int params_data_size = batchSize * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(output_voxel_features, 0, output_voxel_features_size, stream));

    checkCudaErrors(prepareSubMGridKernel_launch(
          input_coords,grids_out,voxel_num,out_shape_x_,out_shape_y_,out_shape_z_, stream));

    checkCudaErrors(getValidOutPosKernel_launch(input_coords, valid_points,valid_points_num,
        voxel_num,ksize_,stride_, padding_, dilation_, out_shape_x_,out_shape_y_, out_shape_z_,stream));

    checkCudaErrors(getSubMIndicePairsKernel_launch(input_coords,grids_out,indice_pairs,indice_num,valid_points,valid_points_num, voxel_num,
                    out_shape_x_,out_shape_y_,out_shape_z_,stream));

    // unsigned int voxel_num_cpu = 0;
    // float * voxel_features_cpu = (float*)malloc(sizeof(float)*20000*4);
    // std::cout << ksize_ * ksize_ * ksize_ << std::endl;
    // unsigned int * indice_num_cpu = (unsigned int*)malloc(sizeof(unsigned int)*27);
    // CHECK(cudaMemcpyAsync(indice_num_cpu, indice_num, 27 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // for(int i=0;i<27;i++)
    // {
    //     std::cout << i << " : " << indice_num_cpu[i] << std::endl;
    // }
    // CHECK(cudaMemcpyAsync(voxel_features_cpu, input_voxel_features, 1 * 20000 * 4* sizeof(float), cudaMemcpyDeviceToHost, stream));
    // unsigned int input_index = 0;
    // unsigned int* coors_cpu = (unsigned int*)malloc(sizeof(unsigned int)*20000*4);
    // CHECK(cudaMemcpyAsync(coors_cpu, input_coords, 1 * 20000 * 4* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // for(int i=0;i<16825;i++)
    // {
    //     if(coors_cpu[i*4+1] == 38 && coors_cpu[i*4+2] == 800 && coors_cpu[i*4+3] == 366)
    //     {
    //         input_index = i;
    //     }
    // }
    // std::cout << "input_index:   " << input_index << std::endl;
    // free(coors_cpu);
    
    // find max number of activate
    int kernel_volume = ksize_ * ksize_ * ksize_;
  

    for(int i = 0; i < kernel_volume; i++)
    {
        // if(i == max_num_act_index) center_flag = 1;
        // std::cout << i << std::endl;
        checkCudaErrors(sparseGatherConvScatter_launch(input_voxel_features,indice_pairs,indice_num,output_voxel_features,i,weights_dev_,
                                                        in_channel_,out_channel_,stream));
    }
    // std::cout << "enqueue  finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    return 0;
}

nvinfer1::DataType SubmConv3dLayerPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    // if(index == 0)
    return inputTypes[0];
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
    return 1;
}

int SubmConv3dLayerPlugin::initialize() noexcept
{   
    return 0;
}

void SubmConv3dLayerPlugin::terminate() noexcept
{
    // // cudafree
    cudaFree(weights_dev_);
    // //c free
    // free(weights_data_);
}

SubmConv3dLayerPlugin::~SubmConv3dLayerPlugin()
{
    terminate();
}

size_t SubmConv3dLayerPlugin::getSerializationSize() const noexcept
{
    return ksize_*ksize_*ksize_*in_channel_*out_channel_ * sizeof(float) + 16 * sizeof(int);
}

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

void SubmConv3dLayerPlugin::destroy() noexcept
{
    delete this;
}

void SubmConv3dLayerPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SubmConv3dLayerPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

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

const char* SubmConv3dLayerPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* SubmConv3dLayerPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* SubmConv3dLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

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

IPluginV2* SubmConv3dLayerPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new SubmConv3dLayerPlugin(serialData, serialLength);
}

void SubmConv3dLayerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SubmConv3dLayerPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

SubmConv3dLayerPluginCreator::~SubmConv3dLayerPluginCreator()
{

}