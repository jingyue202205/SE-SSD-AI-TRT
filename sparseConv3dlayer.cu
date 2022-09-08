#include "sparseConv3dlayer.h"


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
#include <malloc.h>

using namespace std::chrono;
using std::string;

using namespace nvinfer1;
using nvinfer1::SparseConv3dLayerPlugin;
using nvinfer1::SparseConv3dLayerPluginCreator;
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
static const char* PLUGIN_NAME{"SparseConv3dLayerPlugin"};

// Static class fields initialization
PluginFieldCollection SparseConv3dLayerPluginCreator::mFC{};
std::vector<PluginField> SparseConv3dLayerPluginCreator::mPluginAttributes;

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

__device__ void cuda_sleep(int64_t num_cycles)
{
    int64_t cycles = 0;
    int64_t start = clock64();
    while(cycles < num_cycles)
    {
        cycles = clock64() - start;
    }
}

SparseConv3dLayerPlugin::SparseConv3dLayerPlugin(
        int in_channel, int out_channel,int max_voxels, int feature_num, int out_shape_x,int out_shape_y,int out_shape_z,
        int spatial_shape_x, int spatial_shape_y, int spatial_shape_z,int ksize0,int ksize1, int ksize2,int stride0,int stride1,
        int stride2, int padding0,int padding1,int padding2, int dilation, int out_padding, int weights_size, nvinfer1::Weights const& weights
) : in_channel_(in_channel), out_channel_(out_channel),max_voxels_(max_voxels), feature_num_(feature_num), 
    out_shape_x_(out_shape_x),out_shape_y_(out_shape_y),out_shape_z_(out_shape_z),
        spatial_shape_x_(spatial_shape_x), spatial_shape_y_(spatial_shape_y), spatial_shape_z_(spatial_shape_z),ksize0_(ksize0),ksize1_(ksize1),ksize2_(ksize2),
        stride0_(stride0),stride1_(stride1),stride2_(stride2), padding0_(padding0),padding1_(padding1),padding2_(padding2), dilation_(dilation), out_padding_(out_padding),weights_size_(weights_size)
{

    weights_data_ = (float*)malloc(weights_size_*sizeof(float));
    const float* temp_values = (const float*)weights.values;
    for(int i = 0;i < weights_size_;i++)
    {
        weights_data_[i] = temp_values[i];
    }
    weights_.count = weights.count;
    weights_.values = weights_data_;

    
    //  cudamalloc for conv
    checkCudaErrors(cudaMalloc(&weights_dev_,sizeof(float)*ksize0_*ksize1_*ksize2_*in_channel_*out_channel_));
    
    // copy to gpu
    checkCudaErrors(cudaMemcpy(weights_dev_,weights_data_,sizeof(float)*ksize0_*ksize1_*ksize2_*in_channel_*out_channel_,cudaMemcpyHostToDevice));
 
}

SparseConv3dLayerPlugin::SparseConv3dLayerPlugin(const void* data, size_t length)
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
    ksize0_ = readFromBuffer<int>(d);
    ksize1_ = readFromBuffer<int>(d);
    ksize2_ = readFromBuffer<int>(d);
    stride0_ = readFromBuffer<int>(d);
    stride1_ = readFromBuffer<int>(d);
    stride2_ = readFromBuffer<int>(d);
    padding0_ = readFromBuffer<int>(d);
    padding1_ = readFromBuffer<int>(d);
    padding2_ = readFromBuffer<int>(d);
    dilation_ = readFromBuffer<int>(d);
    out_padding_ = readFromBuffer<int>(d);
    weights_size_ = readFromBuffer<int>(d);
    weights_.count = weights_size_;
    weights_data_ = (float *)malloc(weights_size_*sizeof(float));
    for(int i=0;i < weights_size_; i++) 
    {
        weights_data_[i] = readFromBuffer<float>(d);
    }
        
    weights_.values = weights_data_;
    // std::cout << "构造函数2 end" << std::endl;
    // std::cout << in_channel_ << "," << out_channel_ << "," << max_voxels_ << "," << feature_num_ << "," << out_shape_x_ << "," << 
    //             out_shape_y_ << "," << out_shape_z_ << "," << spatial_shape_x_ << "," << spatial_shape_y_ << "," << spatial_shape_z_ << 
    //             "," << ksize0_ << "," << stride0_ << "," << padding0_ << "," << dilation_ << "," << out_padding_ << "," << weights_size_ << std::endl;
}

IPluginV2DynamicExt* SparseConv3dLayerPlugin::clone() const noexcept
{
    auto* plugin = new SparseConv3dLayerPlugin(in_channel_, out_channel_,max_voxels_, feature_num_, out_shape_x_,out_shape_y_,out_shape_z_,
        spatial_shape_x_, spatial_shape_y_, spatial_shape_z_,ksize0_,ksize1_,ksize2_,stride0_,stride1_,stride2_, padding0_,padding1_,padding2_, dilation_, out_padding_, weights_size_, weights_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs SparseConv3dLayerPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // int input_max_voxel_= inputs[0].d[1]->getConstantValue()*2;
    // std::cout << input_max_voxel_ << "<<<<<<<<<<<<" << endl;
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
    if(outputIndex == 1){
        
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 3;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(max_voxels_);
        dim1.d[2] = exprBuilder.constant(4);
        return dim1; // coors 1 20000 4
    }

    nvinfer1::DimsExprs dim2{};
    dim2.nbDims = 1;
    dim2.d[0] = batch_size;
    return dim2;
}

bool SparseConv3dLayerPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
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
    if (pos == 4)    // out_coors  dim: 1x20000x4
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 5)    // out_voxel_num 
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void SparseConv3dLayerPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t SparseConv3dLayerPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
   
    int dense_voxel_num = out_shape_x_ * out_shape_y_ * out_shape_z_;
    size_t grids_out_size = batchSize * dense_voxel_num * sizeof(int);  // 41*1600*1408 -1
    size_t indice_pairs_size = batchSize * ksize0_ * ksize1_ * ksize2_ * MAX_VOXELS * 2 * sizeof(int);    // 27,2,16825 -1
    size_t indice_num_size = batchSize * ksize0_ * ksize1_ * ksize2_ * sizeof(unsigned int); // 27 ,0
    size_t valid_points_size = batchSize * MAX_VOXELS *27*4 * sizeof(unsigned int); // 20000*27*4
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
                                    unsigned int *voxel_num,int ksize0,int ksize1,int ksize2,int stride0,int stride1,
                                    int stride2, int padding0,int padding1,int padding2, 
                                    int dilation, int out_shape_x, int out_shape_y, int out_shape_z)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxelidx >= *voxel_num) return;

    uint4 coords = ((uint4*)input_coords)[voxelidx];

    // getValidOutPos
    int NDim = 3;
    int coords_[3] = {coords.y,coords.z,coords.w};
    int ksize_[3] = {ksize0,ksize1,ksize2};
    int stride_[3] = {stride0,stride1,stride2};
    int padding_[3] = {padding0,padding1,padding2};
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
}


cudaError_t getValidOutPosKernel_launch(unsigned int *input_coords, unsigned int * valid_points,unsigned int* valid_points_num,
                                    unsigned int *voxel_num,int ksize0,int ksize1,int ksize2,int stride0,
                                    int stride1,int stride2, int padding0, int padding1,int padding2,
                                    int dilation, int out_shape_x, int out_shape_y, int out_shape_z,cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  getValidOutPosKernel<<<blocks, threads, 0, stream>>>
       (input_coords, valid_points,valid_points_num,
        voxel_num,ksize0,ksize1,ksize2,stride0,stride1,stride2, padding0,padding1,padding2,dilation,out_shape_x, out_shape_y, out_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void getSparseIndicePairsKernel(unsigned int *input_coords, int * grids_out, int *indice_pairs, unsigned int *indice_num,
                                         unsigned int * valid_points, unsigned int * valid_points_num, unsigned int *voxel_num,int out_shape_x,
                                         int out_shape_y, int out_shape_z,unsigned int *output_coords, unsigned int *output_voxel_num)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxelidx >= *voxel_num) return;


    int index = 0;
    
    uint4 coords = ((uint4*)input_coords)[voxelidx];
  
    unsigned int index_z = 0;
    unsigned int index_y = 0;
    unsigned int index_x = 0;
    unsigned int offset = 0;
    int point_counter = valid_points_num[voxelidx];
    for (int i = 0; i < point_counter; ++i) 
    {
      index_z = *(valid_points + voxelidx*(27*4)+i*4+0);
      index_y = *(valid_points + voxelidx*(27*4)+i*4+1);
      index_x = *(valid_points + voxelidx*(27*4)+i*4+2);
      offset = *(valid_points + voxelidx*(27*4)+i*4+3);
   

    index = coords.x * out_shape_x * out_shape_y * out_shape_z + 
                index_z * out_shape_y * out_shape_x +
                index_y * out_shape_x + 
                index_x;
    
    if(grids_out[index] != -1)
      {
        
        atomicExch(output_coords + grids_out[index] * 4 + 1,index_z);
        atomicExch(output_coords + grids_out[index] * 4 + 2,index_y);
        atomicExch(output_coords + grids_out[index] * 4 + 3,index_x);
        atomicExch(output_coords + grids_out[index] * 4 + 0,0); // batch_size == 1;
      }
   
      unsigned int old_num = atomicAdd(indice_num + offset, 1);

      int * indice_pairs_address_1 = indice_pairs + offset*(2*MAX_VOXELS) + 1 * MAX_VOXELS + old_num;
    
      int old_1 = atomicExch(indice_pairs_address_1,grids_out[index]);  // output index

      int * indice_pairs_address_0 = indice_pairs + offset*(2*MAX_VOXELS) + 0 * MAX_VOXELS + old_num;
      int old_0 = atomicExch(indice_pairs_address_0,voxelidx);  // input index
    }

}


cudaError_t getSparseIndicePairsKernel_launch(unsigned int *input_coords, int * grids_out, int *indice_pairs, 
                                        unsigned int *indice_num,unsigned int * valid_points, unsigned int *valid_points_num,
                                        unsigned int *voxel_num,int out_shape_x,
                                         int out_shape_y, int out_shape_z,
                                         unsigned int *output_coords, unsigned int *output_voxel_num,
                                         cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  getSparseIndicePairsKernel<<<blocks, threads, 0, stream>>>
       (input_coords, grids_out, indice_pairs, indice_num,valid_points,valid_points_num, voxel_num,out_shape_x,out_shape_y,out_shape_z,
       output_coords, output_voxel_num);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void fillGridKernel1(unsigned int *input_coords, int * grids_out,
                                         unsigned int * valid_points, unsigned int * valid_points_num, unsigned int *voxel_num,int out_shape_x,
                                         int out_shape_y, int out_shape_z)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxelidx >= *voxel_num) return;

    int index = 0;
    
    uint4 coords = ((uint4*)input_coords)[voxelidx];
   
    unsigned int index_z = 0;
    unsigned int index_y = 0;
    unsigned int index_x = 0;
    unsigned int offset = 0;
    int point_counter = valid_points_num[voxelidx];
    for (int i = 0; i < point_counter; ++i) 
    {
      index_z = *(valid_points + voxelidx*(27*4)+i*4+0);
      index_y = *(valid_points + voxelidx*(27*4)+i*4+1);
      index_x = *(valid_points + voxelidx*(27*4)+i*4+2);
      offset = *(valid_points + voxelidx*(27*4)+i*4+3);

    index = coords.x * out_shape_x * out_shape_y * out_shape_z + 
                index_z * out_shape_y * out_shape_x +
                index_y * out_shape_x + 
                index_x;
   
    if(grids_out[index] == -1)
      {
        atomicExch(grids_out+index,1);

      }
    
    }

}


cudaError_t fillGridKernel_launch1(unsigned int *input_coords, int * grids_out,unsigned int * valid_points, unsigned int *valid_points_num,
                                    unsigned int *voxel_num,int out_shape_x,
                                         int out_shape_y, int out_shape_z,
                                         cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  fillGridKernel1<<<blocks, threads, 0, stream>>>
       (input_coords, grids_out,valid_points,valid_points_num, voxel_num,out_shape_x,out_shape_y,out_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void fillGridKernel2(int * grids_out, 
                                unsigned int * output_voxel_num,int out_shape_x,
                                int out_shape_y, int out_shape_z)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (grids_out[voxelidx] == 1)
    {   
       int old_value =  atomicAdd(output_voxel_num,1);
        grids_out[voxelidx] = old_value;
    }
    
}


cudaError_t fillGridKernel_launch2(int * grids_out, unsigned int *output_voxel_num,
                                int out_shape_x,int out_shape_y, int out_shape_z,
                                cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((out_shape_x * out_shape_y*out_shape_z+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  fillGridKernel2<<<blocks, threads, 0, stream>>>
       (grids_out,output_voxel_num,out_shape_x,out_shape_y,out_shape_z);
  cudaError_t err = cudaGetLastError();
  return err;
}

__global__ void sparseGatherConvScatter(float* input_voxel_features,int *indice_pairs,unsigned int *indice_num,
                        float *output_voxel_features,int kernel_volume_index,float* weights_dev_,int in_channel, int out_channel)
{
    
    int act_idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int num_act = indice_num[kernel_volume_index];


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
            current_sum += input_voxel_features[input_index * in_channel + j] * weights_dev_[kernel_volume_index * kernel_num+i+j*out_channel];
        }
       
        float old_value = atomicAdd(output_voxel_features + output_index * out_channel + i,current_sum);

    }

}


cudaError_t sparseGatherConvScatter_launch(float* input_voxel_features,int *indice_pairs,unsigned int *indice_num,
                        float *output_voxel_features,int kernel_volume_index,float* weights_dev_,
                        int in_channel, int out_channel, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  sparseGatherConvScatter<<<blocks, threads, 0, stream>>>
       (input_voxel_features,indice_pairs,indice_num,output_voxel_features,kernel_volume_index,weights_dev_,in_channel,out_channel);
  cudaError_t err = cudaGetLastError();
  return err;
}

int SparseConv3dLayerPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    // std::cout << "enqueue start" << std::endl;
    int batchSize = inputDesc[0].dims.d[0];
   
    //TRT-input
    // std::cout << "batch_size: " << batchSize << std::endl;
    float *input_voxel_features = const_cast<float *>((const float*)inputs[0]);
    unsigned int *input_coords = const_cast<unsigned int *>((const unsigned int*)inputs[1]);
    unsigned int * voxel_num = const_cast<unsigned int *>((const unsigned int *)inputs[2]);

    //TRT-output
    float *output_voxel_features = (float *)(outputs[0]);  // 1 * 20000 * 27 * 32
    unsigned int *output_coords = (unsigned int *)(outputs[1]); // 1* 20000*27*4
    unsigned int *output_voxel_num = (unsigned int *)(outputs[2]); // 1


    int dense_voxel_num = out_shape_x_ * out_shape_y_ * out_shape_z_;
    size_t grids_out_size = batchSize * dense_voxel_num * sizeof(int);  // 41*1600*1408 -1
    size_t indice_pairs_size = batchSize * ksize0_ * ksize1_ * ksize2_ * MAX_VOXELS * 2 * sizeof(int);    // 27,2,16825 -1
    size_t indice_num_size = batchSize * ksize0_ * ksize1_ * ksize2_ * sizeof(unsigned int); // 27 ,0
    size_t valid_points_size = batchSize * MAX_VOXELS * 27*4 * sizeof(unsigned int); // 20000*256*4
    size_t valid_points_num_size = batchSize * MAX_VOXELS * sizeof(unsigned int);// 20000*1;

    // std::cout << out_shape_x_ << "," << out_shape_y_ << "," << out_shape_z_ << "," << ksize0_ << "," << max_voxels_ << "," << out_channel_ << std::endl;
    size_t workspaces[5];
    workspaces[0] = grids_out_size;
    workspaces[1] = indice_pairs_size;
    workspaces[2] = indice_num_size;
    workspaces[3] = valid_points_size;
    workspaces[4] = valid_points_num_size;

    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 5);
  
    int* grids_out = static_cast<int*>(workspace);
   
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
    unsigned int output_coords_size = batchSize * max_voxels_ * 4 * sizeof(unsigned int);
    unsigned int output_voxel_num_size = batchSize * sizeof(unsigned int);
    // unsigned int params_data_size = batchSize * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(output_voxel_features, 0, output_voxel_features_size, stream));
    checkCudaErrors(cudaMemsetAsync(output_coords, 0, output_coords_size, stream));
    checkCudaErrors(cudaMemsetAsync(output_voxel_num, 0, output_voxel_num_size, stream));

    // unsigned int voxel_num_cpu = 0;
    // float * voxel_features_cpu = (float*)malloc(sizeof(float)*20000*4);
    // unsigned int * coors_cpu = (unsigned int*)malloc(sizeof(unsigned int)*20000*4);
    // CHECK(cudaMemcpyAsync(&voxel_num_cpu, voxel_num, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(voxel_features_cpu, input_voxel_features, 1 * 20000 * 4* sizeof(float), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(coors_cpu, input_coords, 1 * 20000 * 4* sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));

    // string save_path = "./cuda_000000.txt";
    // std::ofstream out_txt_file;
    // out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // // out_txt_file << fixed;
    // out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    // int seconds = 1;
    // out_txt_file << seconds << std::endl;

    // for(int i=0;i < voxel_num_cpu; i++)
    // {
    //     for(int j=0;j<4;j++)
    //     {
    //         out_txt_file << *(voxel_features_cpu + i * 4+j) << ",";
    //     }
    //     out_txt_file << *(coors_cpu + i * 4) << "," << *(coors_cpu + i * 4+1) << "," 
    //         << *(coors_cpu + i * 4 + 2) << "," << *(coors_cpu+i*4+3) << std::endl;;

    //     // std::cout << "index : " << i << "   " << "voxel: " << *(voxel_feature + i * 4) << ","  << *(voxel_feature + i * 4+1) << ","
    //     //     << *(voxel_feature + i * 4+2) << "," << *(voxel_feature + i * 4+3) << ",||||"<< *(coors + i * 4) << "," << *(coors + i * 4+1) << "," 
    //     //     << *(coors + i * 4 + 2) << "," << *(coors+i*4+3)
    //     //     << std::endl;   
    // }
    // out_txt_file.close();
    // // return 0;
    // free(voxel_features_cpu);
    // free(coors_cpu);

    checkCudaErrors(getValidOutPosKernel_launch(input_coords, valid_points,valid_points_num,
        voxel_num,ksize0_,ksize1_,ksize2_,stride0_,stride1_,stride2_, padding0_,padding1_,padding2_, dilation_, out_shape_x_,out_shape_y_, out_shape_z_,stream));

    
    // // c++ fill grid
    // // create memory
    // // int * grids_out,unsigned int * valid_points, unsigned int * valid_points_num, unsigned int *voxel_num,int out_shape_x,
    // //                                      int out_shape_y, int out_shape_z, unsigned int *output_voxel_num
    // // cout << "voxel_num" << *(voxel_num) << std::endl;
    // unsigned int voxel_num_cpu_ = 0;
    // unsigned int cpu_out_voxel_num_ = 0;
    // // size_t grids_out_size = batchSize * dense_voxel_num * sizeof(int);  // 41*1600*1408 -1
    // // size_t valid_points_size = batchSize * MAX_VOXELS * 256*4 * sizeof(unsigned int); // 20000*256*4
    // // size_t valid_points_num_size = batchSize * MAX_VOXELS * sizeof(unsigned int);// 20000*1;
    // int *grids_out_cpu = (int*)malloc(grids_out_size);
    // unsigned int * valid_points_cpu = (unsigned int*)malloc(valid_points_size);
    // unsigned int * valid_points_num_cpu = (unsigned int*)malloc(valid_points_num_size);
    // // cout << "out_shape_x" << out_shape_x_ << std::endl;
    


    // // copy to cpu
    // CHECK(cudaMemcpyAsync(&voxel_num_cpu_, voxel_num, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // // CHECK(cudaMemcpyAsync(&cpu_out_voxel_num_, output_voxel_num, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(grids_out_cpu, grids_out, grids_out_size, cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(valid_points_cpu, valid_points, valid_points_size, cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(valid_points_num_cpu, valid_points_num, valid_points_num_size, cudaMemcpyDeviceToHost, stream));
    // int point_counter = 0;
    // int index = 0;
    // unsigned int index_z = 0;
    // unsigned int index_y = 0;
    // unsigned int index_x = 0;
    // unsigned int offset = 0;
    // for(int i=0;i<voxel_num_cpu_;i++)
    // {
    //     int point_counter = valid_points_num_cpu[i];
    //     for(int j=0;j<point_counter;j++)
    //     {
    //         index_z = *(valid_points_cpu + i*(256*4)+j*4+0);
    //         index_y = *(valid_points_cpu + i*(256*4)+j*4+1);
    //         index_x = *(valid_points_cpu + i*(256*4)+j*4+2);
    //         offset = *(valid_points_cpu + i*(256*4)+j*4+3);

    //         index = 0 * out_shape_x_ * out_shape_y_ * out_shape_z_ + index_z * out_shape_y_ * out_shape_x_ + index_y * out_shape_x_ +  index_x;
    //         if (grids_out_cpu[index] == -1)
    //         {

    //         grids_out_cpu[index] = cpu_out_voxel_num_;
    //         cpu_out_voxel_num_++;
    //         }
    //     }
        
    // }
    // std::cout << "cpu_out_voxel_num_" << cpu_out_voxel_num_ << std::endl;
    // // copy to gpu
    // // CHECK(cudaMemcpyAsync(&voxel_num_cpu_, voxel_num, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // CHECK(cudaMemcpyAsync(output_voxel_num, &cpu_out_voxel_num_, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    // CHECK(cudaMemcpyAsync(grids_out, grids_out_cpu, grids_out_size, cudaMemcpyHostToDevice, stream));
    
    // free(grids_out_cpu);
    // free(valid_points_cpu);
    // free(valid_points_num_cpu);


    checkCudaErrors(fillGridKernel_launch1(input_coords,grids_out,valid_points,valid_points_num, voxel_num,
                    out_shape_x_,out_shape_y_,out_shape_z_,stream));

    checkCudaErrors(fillGridKernel_launch2(grids_out,output_voxel_num,out_shape_x_,out_shape_y_,out_shape_z_,stream));
    
    checkCudaErrors(getSparseIndicePairsKernel_launch(input_coords,grids_out,indice_pairs,indice_num,valid_points,valid_points_num, voxel_num,
                    out_shape_x_,out_shape_y_,out_shape_z_,output_coords, output_voxel_num,stream));

    // //save input features
    // float* voxel_features_cpu = (float*)malloc(sizeof(float)*40000*16);
    // CHECK(cudaMemcpyAsync(voxel_features_cpu, input_voxel_features, 40000*16 * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // std::ofstream out_txt_file;
    // std::string save_path = "./000000_trt_input_feature.txt";
    // out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // // out_txt_file << fixed;
    // out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    // // int voxel_num_cpu = 16825;
    // int output_channel = 16; 
    // for(int i=0; i < voxel_num_cpu; i++)
    // {
    //     for(int j=0;j<output_channel;j++)
    //     {
    //         out_txt_file << *(voxel_features_cpu + i * output_channel+j) << ",";
    //     }
    //     out_txt_file << std::endl;

    //     // std::cout << "index : " << i << "   " << "voxel: " << *(voxel_feature + i * 4) << ","  << *(voxel_feature + i * 4+1) << ","
    //     //     << *(voxel_feature + i * 4+2) << "," << *(voxel_feature + i * 4+3) << ",||||"<< *(coors + i * 4) << "," << *(coors + i * 4+1) << "," 
    //     //     << *(coors + i * 4 + 2) << "," << *(coors+i*4+3)
    //     //     << std::endl;   
    // }
    // out_txt_file.close();


    int kernel_volume = ksize0_ * ksize1_ * ksize2_;
    for(int i = 0; i < kernel_volume; i++)
    {
        checkCudaErrors(sparseGatherConvScatter_launch(input_voxel_features,indice_pairs,indice_num,output_voxel_features,i,weights_dev_,
                                                        in_channel_,out_channel_,stream));
    }

    // std::cout << "enqueue  finished >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
    return 0;
}

nvinfer1::DataType SparseConv3dLayerPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 0)
    {
        // cout << "inputTypes[0]: " << inputTypes[0] <<  std::endl;
        return inputTypes[0];
        // return nvinfer1::DataType::kFLOAT;
    }
    
    return inputTypes[1];
    // return nvinfer1::DataType::kINT32;
    
}

const char* SparseConv3dLayerPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* SparseConv3dLayerPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int SparseConv3dLayerPlugin::getNbOutputs() const noexcept
{
    return 3;
}

int SparseConv3dLayerPlugin::initialize() noexcept
{  
    return 0;
}

void SparseConv3dLayerPlugin::terminate() noexcept
{
    // // cudafree
    cudaFree(weights_dev_);
    // //c free
    // free(weights_data_);
}

SparseConv3dLayerPlugin::~SparseConv3dLayerPlugin()
{
    terminate();
}

size_t SparseConv3dLayerPlugin::getSerializationSize() const noexcept
{
    return ksize0_*ksize1_*ksize2_*in_channel_*out_channel_ * sizeof(float) + 22 * sizeof(int);
}

void SparseConv3dLayerPlugin::serialize(void* buffer) const noexcept
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
    writeToBuffer<int>(d, ksize0_);
    writeToBuffer<int>(d, ksize1_);
    writeToBuffer<int>(d, ksize2_);
    writeToBuffer<int>(d, stride0_);
    writeToBuffer<int>(d, stride1_);
    writeToBuffer<int>(d, stride2_);
    writeToBuffer<int>(d, padding0_);
    writeToBuffer<int>(d, padding1_);
    writeToBuffer<int>(d, padding2_);
    writeToBuffer<int>(d, dilation_);
    writeToBuffer<int>(d, out_padding_);
    writeToBuffer<int>(d, weights_size_);
    const float * data = (const float*)weights_.values;
    for(int i=0; i < weights_size_; i++)
    {
        writeToBuffer<float>(d,data[i]);
    }
}

void SparseConv3dLayerPlugin::destroy() noexcept
{
    delete this;
}

void SparseConv3dLayerPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SparseConv3dLayerPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

SparseConv3dLayerPluginCreator::SparseConv3dLayerPluginCreator()
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

const char* SparseConv3dLayerPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* SparseConv3dLayerPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* SparseConv3dLayerPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* SparseConv3dLayerPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
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
    int ksize0 = 0;
    int ksize1 = 0;
    int ksize2 = 0;
    int stride0 = 0;
    int stride1 = 0;
    int stride2 = 0;
    int padding0 = 0;
    int padding1 = 0;
    int padding2 = 0;
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
            ksize0 = d[0];
            ksize1 = d[1];
            ksize2 = d[2];
        }
        else if (!strcmp(attr_name, "stride"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            stride0 = d[0];
            stride1 = d[1];
            stride2 = d[2];
        }
        else if (!strcmp(attr_name, "padding"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            padding0 = d[0];
            padding1 = d[1];
            padding2 = d[2];
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
    weights_size = in_channel * out_channel * ksize0 * ksize1 * ksize2;
    // for(int i=0;i<weights_size;i++)
        // std::cout << weight[3*3*3*4*16-1] << std::endl;
    Weights wt{DataType::kFLOAT, nullptr, 0};
    wt.count = weights_size;
    wt.values = weight;
    // std::cout << max_voxels << " " << in_channel << " " <<out_channel << " " << feature_num << " " << out_shape_x << " "
    // << out_shape_y << " "<< out_shape_z << " " << spatial_shape_x << " " << spatial_shape_y << " " << spatial_shape_z << " "
    // << ksize0 << " " << stride0 << " " << padding << " " << dilation << " " << out_padding << " " << std::endl;
    
    IPluginV2DynamicExt* plugin = new SparseConv3dLayerPlugin(in_channel, out_channel,max_voxels, feature_num, out_shape_x,out_shape_y,out_shape_z,
                                    spatial_shape_x, spatial_shape_y, spatial_shape_z,ksize0,ksize1,ksize2,
                                    stride0,stride1,stride2, padding0,padding1,padding2, dilation, out_padding, weights_size, wt);
    return plugin;
}

IPluginV2* SparseConv3dLayerPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new SparseConv3dLayerPlugin(serialData, serialLength);
}

void SparseConv3dLayerPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* SparseConv3dLayerPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

SparseConv3dLayerPluginCreator::~SparseConv3dLayerPluginCreator()
{

}