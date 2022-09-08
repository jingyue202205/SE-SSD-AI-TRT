#include "zeroPad2d.h"

using namespace nvinfer1;
using nvinfer1::ZeroPad2dPlugin;
using nvinfer1::ZeroPad2dPluginCreator;
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
static const char* PLUGIN_NAME{"ZeroPad2dPlugin"};

// Static class fields initialization
PluginFieldCollection ZeroPad2dPluginCreator::mFC{};
std::vector<PluginField> ZeroPad2dPluginCreator::mPluginAttributes;

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
ZeroPad2dPlugin::ZeroPad2dPlugin(int zero_padding_2d_top, int zero_padding_2d_bottom, int zero_padding_2d_left,int zero_padding_2d_right)
: zero_padding_2d_top_(zero_padding_2d_top), zero_padding_2d_bottom_(zero_padding_2d_bottom), zero_padding_2d_left_(zero_padding_2d_left),
    zero_padding_2d_right_(zero_padding_2d_right)
{
}

ZeroPad2dPlugin::ZeroPad2dPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    zero_padding_2d_top_ = readFromBuffer<int>(d);
    zero_padding_2d_bottom_ = readFromBuffer<int>(d);
    zero_padding_2d_left_ = readFromBuffer<int>(d);
    zero_padding_2d_right_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* ZeroPad2dPlugin::clone() const noexcept
{
    auto* plugin = new ZeroPad2dPlugin(zero_padding_2d_top_,zero_padding_2d_bottom_,zero_padding_2d_left_,zero_padding_2d_right_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs ZeroPad2dPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    auto channel_num = inputs[0].d[1];
    auto height = inputs[0].d[2];
    auto width = inputs[0].d[3];

    int height_ = height->getConstantValue();
    int width_ = width->getConstantValue();

    // std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
    // std::cout << channel_num << " " << height_ << " " << width_ << std::endl; 
   
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 4;
        dim0.d[0] = batch_size;
        dim0.d[1] = channel_num;
        dim0.d[2] = exprBuilder.constant(height_+zero_padding_2d_top_+zero_padding_2d_bottom_); // 200
        dim0.d[3] = exprBuilder.constant(width_+zero_padding_2d_left_+zero_padding_2d_right_);  //176
        return dim0; // voxels 1 20000 4
    }
}

bool ZeroPad2dPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
  
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // voxel_feature 
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // dense mat
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void ZeroPad2dPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t ZeroPad2dPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}
   

__global__ void zero_padding2d_kernel(float *feature_map,float* output_feature_map,int channel_num,int height,int width,int zero_padding_2d_top,        
                            int zero_padding_2d_left,int output_height, int output_width)
{
    // printf("point_size:%d\n",*points_size);
    int voxel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int height_index = voxel_idx / width;
    int width_index = voxel_idx % width;

    int output_height_index = height_index + zero_padding_2d_top;
    int output_width_index = width_index + zero_padding_2d_left;

    for(int i=0;i < channel_num;i++)
    {
        output_feature_map[i*output_height*output_width+output_height_index*output_width+output_width_index] = 
        feature_map[i*height*width + height_index*width + width_index];
    }
}

  
cudaError_t zero_padding2d_launch(float *feature_map,float* output_feature_map,int channel_num,int height,int width,int zero_padding_2d_top,        
                            int zero_padding_2d_left,int output_height,int output_width, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((width*height+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  zero_padding2d_kernel<<<blocks, threads, 0, stream>>>
       (feature_map,output_feature_map,channel_num,height,width,zero_padding_2d_top,zero_padding_2d_left,output_height,output_width);
  cudaError_t err = cudaGetLastError();
  return err;
}


int ZeroPad2dPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    int channel_num = inputDesc[0].dims.d[1];
    int height = inputDesc[0].dims.d[2];
    int width = inputDesc[0].dims.d[3];
    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    float * feature_map = const_cast<float *>((const float *)inputs[0]);
    //TRT-output
    float *output_feature_map = (float *)(outputs[0]);
   

    // init output
    unsigned int output_feature_map_size = batchSize * channel_num * (zero_padding_2d_top_+zero_padding_2d_bottom_+height)  * (zero_padding_2d_left_+zero_padding_2d_right_+width) * sizeof(float);
    
    checkCudaErrors(cudaMemsetAsync(output_feature_map, 0, output_feature_map_size, stream));
    
    int output_height = height + zero_padding_2d_top_ + zero_padding_2d_bottom_;
    int output_width = width + zero_padding_2d_left_  + zero_padding_2d_right_;
   
    checkCudaErrors(zero_padding2d_launch(
          feature_map,output_feature_map,channel_num,height,width,zero_padding_2d_top_,zero_padding_2d_left_,output_height,output_width, stream));
    // cout << "zero_padding2d finished" << std::endl;
    return 0;
}


nvinfer1::DataType ZeroPad2dPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* ZeroPad2dPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* ZeroPad2dPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int ZeroPad2dPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int ZeroPad2dPlugin::initialize() noexcept
{
    return 0;
}

void ZeroPad2dPlugin::terminate() noexcept
{
}


size_t ZeroPad2dPlugin::getSerializationSize() const noexcept
{
    return  4 * sizeof(int);
}

void ZeroPad2dPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, zero_padding_2d_top_);
    writeToBuffer<int>(d, zero_padding_2d_bottom_);
    writeToBuffer<int>(d, zero_padding_2d_left_);
    writeToBuffer<int>(d, zero_padding_2d_right_);
}

void ZeroPad2dPlugin::destroy() noexcept
{
    delete this;
}

void ZeroPad2dPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ZeroPad2dPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



ZeroPad2dPluginCreator::ZeroPad2dPluginCreator()
{


    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("zero_padding_2d_size", nullptr, PluginFieldType::kINT32, 1));
    // mPluginAttributes.emplace_back(PluginField("feature_map_channel", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* ZeroPad2dPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* ZeroPad2dPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* ZeroPad2dPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* ZeroPad2dPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int zero_padding_2d_top = 0;
    int zero_padding_2d_bottom = 0;
    int zero_padding_2d_left = 0;
    int zero_padding_2d_right = 0;
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "zero_padding_2d_size"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            zero_padding_2d_top = d[0];
            zero_padding_2d_bottom = d[1];
            zero_padding_2d_left = d[2];
            zero_padding_2d_right = d[3];
        }
    }
    // std::cout << "fsfsfsfsfsfsfsfsfsfsfsfsf    " << zero_padding_2d_top  << "  " << zero_padding_2d_bottom << " " <<  zero_padding_2d_left << " " << zero_padding_2d_right << " " << nbFields<< std::endl;
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    IPluginV2DynamicExt* plugin = new ZeroPad2dPlugin(zero_padding_2d_top,zero_padding_2d_bottom,zero_padding_2d_left,zero_padding_2d_right);
    return plugin;
}

IPluginV2* ZeroPad2dPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new ZeroPad2dPlugin(serialData, serialLength);
}

void ZeroPad2dPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* ZeroPad2dPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

ZeroPad2dPluginCreator::~ZeroPad2dPluginCreator()
{
   
}