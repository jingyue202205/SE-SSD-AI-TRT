#include "sparse2Dense.h"

using namespace nvinfer1;
using nvinfer1::Sparse2DensePlugin;
using nvinfer1::Sparse2DensePluginCreator;
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
static const char* PLUGIN_NAME{"Sparse2DensePlugin"};

// Static class fields initialization
PluginFieldCollection Sparse2DensePluginCreator::mFC{};
std::vector<PluginField> Sparse2DensePluginCreator::mPluginAttributes;

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

// create the plugin at runtime from a byte stream
Sparse2DensePlugin::Sparse2DensePlugin(int feature_map_size_x, int feature_map_size_y, int feature_map_size_z,int feature_map_channel)
: feature_map_size_x_(feature_map_size_x), feature_map_size_y_(feature_map_size_y), feature_map_size_z_(feature_map_size_z),
    feature_map_channel_(feature_map_channel)
{
}

Sparse2DensePlugin::Sparse2DensePlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    feature_map_size_x_ = readFromBuffer<int>(d);
    feature_map_size_y_ = readFromBuffer<int>(d);
    feature_map_size_z_ = readFromBuffer<int>(d);
    feature_map_channel_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* Sparse2DensePlugin::clone() const noexcept
{
    auto* plugin = new Sparse2DensePlugin(feature_map_size_x_,feature_map_size_y_,feature_map_size_z_,feature_map_channel_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs Sparse2DensePlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
   
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 4;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(feature_map_size_z_*feature_map_channel_);
        dim0.d[2] = exprBuilder.constant(feature_map_size_y_); // 200
        dim0.d[3] = exprBuilder.constant(feature_map_size_x_);  //176
        return dim0; 
    }
}

bool Sparse2DensePlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
  
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // voxel_feature 
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // coor
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // voxel_num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // dense mat
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void Sparse2DensePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t Sparse2DensePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}
   

__global__ void sparse2dense_kernel(float *voxel_feature, unsigned int *coords, unsigned int * voxel_num, float * dense_mat, int feature_map_size_x,
                                int feature_map_size_y, int feature_map_size_z, int feature_map_channel)
{
    // printf("point_size:%d\n",*points_size);
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxel_idx >= *voxel_num) return;

    uint4 coord = ((uint4*)coords)[voxel_idx];
    unsigned int batch_size_idx = coord.x;
    unsigned int z_index = coord.y;
    unsigned int y_index = coord.z;
    unsigned int x_index = coord.w;
    // for(int i=0;i<feature_map_channel;i++)
    // {
    //     dense_mat[batch_size_idx*feature_map_channel*feature_map_size_x*feature_map_size_y*feature_map_size_z+
    //     ((z_index*feature_map_channel)+i)*feature_map_size_x*feature_map_size_y+y_index*feature_map_size_x+x_index] = 
    //     voxel_feature[voxel_idx*feature_map_channel+i];
    // }
    // 穿插排列(64*2)
    for(int i=0;i<feature_map_channel;i++)
    {
        dense_mat[batch_size_idx*feature_map_channel*feature_map_size_x*feature_map_size_y*feature_map_size_z+
        ((i*feature_map_size_z)+z_index)*feature_map_size_x*feature_map_size_y+y_index*feature_map_size_x+x_index] = 
        voxel_feature[voxel_idx*feature_map_channel+i];
    }
}

  
cudaError_t sparse2dense_launch(float *voxel_features,unsigned int *coords,unsigned int* voxel_num,float *dense_mat,
                    int feature_map_size_x,int feature_map_size_y,int feature_map_size_z,int feature_map_channel,cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  sparse2dense_kernel<<<blocks, threads, 0, stream>>>
       (voxel_features,coords,voxel_num,dense_mat,feature_map_size_x,feature_map_size_y,feature_map_size_z,feature_map_channel);
  cudaError_t err = cudaGetLastError();
  return err;
}


int Sparse2DensePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    float * voxel_features = const_cast<float *>((const float *)inputs[0]);
    unsigned int* coords = const_cast<unsigned int *>((const unsigned int *)inputs[1]);
    unsigned int* voxel_num = const_cast<unsigned int*>((const unsigned int *)inputs[2]);
    //TRT-output
    float *dense_mat = (float *)(outputs[0]);
   

    // init output
    unsigned int dense_mat_size = batchSize * feature_map_size_z_ * feature_map_size_y_  * feature_map_size_x_ * feature_map_channel_ * sizeof(float);
    
    checkCudaErrors(cudaMemsetAsync(dense_mat, 0, dense_mat_size, stream));
   
   
    checkCudaErrors(sparse2dense_launch(
          voxel_features,coords,voxel_num,dense_mat,feature_map_size_x_,feature_map_size_y_,feature_map_size_z_,feature_map_channel_, stream));
    // cout << "sparse2dense finished" << std::endl;
    return 0;
}


nvinfer1::DataType Sparse2DensePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* Sparse2DensePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* Sparse2DensePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int Sparse2DensePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int Sparse2DensePlugin::initialize() noexcept
{
    return 0;
}

void Sparse2DensePlugin::terminate() noexcept
{
}


size_t Sparse2DensePlugin::getSerializationSize() const noexcept
{
    return  4 * sizeof(int);
}

void Sparse2DensePlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, feature_map_size_x_);
    writeToBuffer<int>(d, feature_map_size_y_);
    writeToBuffer<int>(d, feature_map_size_z_);
    writeToBuffer<int>(d, feature_map_channel_);
}

void Sparse2DensePlugin::destroy() noexcept
{
    delete this;
}

void Sparse2DensePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* Sparse2DensePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



Sparse2DensePluginCreator::Sparse2DensePluginCreator()
{


    mPluginAttributes.clear();


    mPluginAttributes.emplace_back(PluginField("feature_map_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("feature_map_channel", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* Sparse2DensePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* Sparse2DensePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* Sparse2DensePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* Sparse2DensePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int feature_map_size_x = 0;
    int feature_map_size_y = 0;
    int feature_map_size_z = 0;
    int feature_map_channel = 0;
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "feature_map_size"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_map_size_x = d[0];
            feature_map_size_y = d[1];
            feature_map_size_z = d[2];
        }
        else if (!strcmp(attr_name, "feature_map_channel"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_map_channel = d[0];
        }
    }
    // std::cout << "fsfsfsfsfsfsfsfsfsfsfsfsf    " << feature_map_channel  << "  " << feature_map_size_x << " " <<  feature_map_size_y << " " << feature_map_size_z << " " << nbFields<< std::endl;
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    IPluginV2DynamicExt* plugin = new Sparse2DensePlugin(feature_map_size_x,feature_map_size_y,feature_map_size_z,feature_map_channel);
    return plugin;
}

IPluginV2* Sparse2DensePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new Sparse2DensePlugin(serialData, serialLength);
}

void Sparse2DensePluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* Sparse2DensePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

Sparse2DensePluginCreator::~Sparse2DensePluginCreator()
{
   
}