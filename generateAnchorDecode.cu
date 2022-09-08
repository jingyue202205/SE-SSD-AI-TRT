#include "generateAnchorDecode.h"

using namespace nvinfer1;
using nvinfer1::GenerateAnchorDecodePlugin;
using nvinfer1::GenerateAnchorDecodePluginCreator;
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
static const char* PLUGIN_NAME{"GenerateAnchorDecodePlugin"};


// Static class fields initialization
PluginFieldCollection GenerateAnchorDecodePluginCreator::mFC{};
std::vector<PluginField> GenerateAnchorDecodePluginCreator::mPluginAttributes;

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
GenerateAnchorDecodePlugin::GenerateAnchorDecodePlugin(float min_x_range,float max_x_range,float min_y_range,float max_y_range,
                    float min_z_range,float max_z_range, int feature_map_height, int feature_map_width, float car_length,
                    float car_width, float car_height, float direction_angle_0, float direction_angle_1,int direction_angle_num)
: min_x_range_(min_x_range), max_x_range_(max_x_range), min_y_range_(min_y_range),
    max_y_range_(max_y_range),min_z_range_(min_z_range),max_z_range_(max_z_range),feature_map_height_(feature_map_height),
    feature_map_width_(feature_map_width),car_length_(car_length),car_width_(car_width),car_height_(car_height),
    direction_angle_0_(direction_angle_0),direction_angle_1_(direction_angle_1),direction_angle_num_(direction_angle_num)
{   
   
}

GenerateAnchorDecodePlugin::GenerateAnchorDecodePlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    min_x_range_ = readFromBuffer<float>(d);
    max_x_range_ = readFromBuffer<float>(d);
    min_y_range_ = readFromBuffer<float>(d);
    max_y_range_ = readFromBuffer<float>(d);
    min_z_range_ = readFromBuffer<float>(d);
    max_z_range_ = readFromBuffer<float>(d);

    feature_map_height_ = readFromBuffer<int>(d);
    feature_map_width_ = readFromBuffer<int>(d);

    car_length_ = readFromBuffer<float>(d);
    car_width_ = readFromBuffer<float>(d);
    car_height_ = readFromBuffer<float>(d);
    direction_angle_0_ = readFromBuffer<float>(d);
    direction_angle_1_ = readFromBuffer<float>(d);


    direction_angle_num_ = readFromBuffer<int>(d);

}

IPluginV2DynamicExt* GenerateAnchorDecodePlugin::clone() const noexcept
{
    auto* plugin = new GenerateAnchorDecodePlugin(min_x_range_,max_x_range_,min_y_range_,max_y_range_,
                min_z_range_,max_z_range_,feature_map_height_,feature_map_width_,car_length_,car_width_,car_height_,
                direction_angle_0_,direction_angle_1_,direction_angle_num_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs GenerateAnchorDecodePlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    auto line_num = inputs[0].d[1];
    auto dim_num = inputs[0].d[2];

    // std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
    // std::cout << batch_size->getConstantValue() << " " << line_num->getConstantValue() << " " 
    //         << dim_num->getConstantValue() << std::endl; 
   
    if (outputIndex == 0)
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = line_num;
        dim0.d[2] = dim_num;
        return dim0; 
    }
}

bool GenerateAnchorDecodePlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)      
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void GenerateAnchorDecodePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t GenerateAnchorDecodePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}
   

__global__ void generate_anchor_kernel(float* output_features,float min_x_range, float max_x_range, float min_y_range,
                float max_y_range, float min_z_range, float max_z_range, int feature_map_height, int feature_map_width,
                float car_length, float car_width, float car_height, int direction_angle_num, float direction_angle_0,
                float direction_angle_1)
{
    // printf("point_size:%d\n",*points_size);
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;

    float stride_x = (max_x_range-min_x_range) / feature_map_width;
    float stride_y = (max_y_range-min_y_range) / feature_map_height;

    float x_start = min_x_range + stride_x/2;
    float y_start = min_y_range + stride_y/2;

    int pos_0_value_index = (line_idx%(feature_map_width*direction_angle_num))/direction_angle_num; //index%352/2
    float pos_0_value = x_start + pos_0_value_index * stride_x;

    int pos_1_value_index = line_idx / (feature_map_width*direction_angle_num); // index/352
    float pos_1_value = y_start + pos_1_value_index * stride_y;
    
    float pos_2_value = -1.0;

    float pos_3_value = car_width; // 1.60; // w
    float pos_4_value = car_length; // 3.90; // l
    float pos_5_value = car_height; // 1.56; // h

    float pos_6_value = 0.0;
    int pos_6_value_index = line_idx % direction_angle_num;
    if(pos_6_value_index == 0)
    {
        pos_6_value = direction_angle_0;
    }
    if(pos_6_value_index == 1)
    {
        pos_6_value = direction_angle_1;
        // printf("direction_range_1: %f\n",direction_angle_1);
    }

    *(output_features+line_idx*7+0) = pos_0_value;
    *(output_features+line_idx*7+1) = pos_1_value;
    *(output_features+line_idx*7+2) = pos_2_value;
    *(output_features+line_idx*7+3) = pos_3_value;
    *(output_features+line_idx*7+4) = pos_4_value;
    *(output_features+line_idx*7+5) = pos_5_value;
    *(output_features+line_idx*7+6) = pos_6_value;

}


cudaError_t generate_anchor_launch(float* output_features,float min_x_range, float max_x_range, float min_y_range,
                float max_y_range, float min_z_range, float max_z_range, int feature_map_height, int feature_map_width,
                float car_length, float car_width, float car_height, int direction_angle_num, float direction_angle_0,
                float direction_angle_1, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((feature_map_height*feature_map_width*direction_angle_num+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generate_anchor_kernel<<<blocks, threads, 0, stream>>>
       (output_features,min_x_range,max_x_range,min_y_range,max_y_range,min_z_range,max_z_range,
          feature_map_height,feature_map_width,car_length,car_width,car_height,direction_angle_num,
          direction_angle_0,direction_angle_1);
  cudaError_t err = cudaGetLastError();
  return err;
}


__global__ void decode_kernel(float* features, float* output_features)
{
    // printf("point_size:%d\n",*points_size);
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // output_features  == anchors
    // features == box_preds
    // xa, ya, za, wa, la, ha, ra
    float anchor_x = *(output_features+line_idx*7+0);
    float anchor_y = *(output_features+line_idx*7+1);
    float anchor_z = *(output_features+line_idx*7+2);
    float anchor_w = *(output_features+line_idx*7+3);
    float anchor_l = *(output_features+line_idx*7+4);
    float anchor_h = *(output_features+line_idx*7+5);
    float anchor_r = *(output_features+line_idx*7+6);

    float box_x = *(features+line_idx*7+0);
    float box_y = *(features+line_idx*7+1);
    float box_z = *(features+line_idx*7+2);
    float box_w = *(features+line_idx*7+3);
    float box_l = *(features+line_idx*7+4);
    float box_h = *(features+line_idx*7+5);
    float box_r = *(features+line_idx*7+6);

    // diagonal = torch.sqrt(la ** 2 + wa ** 2)
    // xg = xt * diagonal + xa
    // yg = yt * diagonal + ya
    // zg = zt * ha + za

    float diagonal = sqrt(anchor_l * anchor_l + anchor_w * anchor_w);

    *(output_features+line_idx*7+0) = box_x * diagonal + anchor_x;
    *(output_features+line_idx*7+1) = box_y * diagonal + anchor_y;
    *(output_features+line_idx*7+2) = box_z * anchor_h + anchor_z;

    // lg = torch.exp(lt) * la
    // wg = torch.exp(wt) * wa
    // hg = torch.exp(ht) * ha
    // ret.extend([wg, lg, hg])

    *(output_features+line_idx*7+3) = exp(box_w) * anchor_w;
    *(output_features+line_idx*7+4) = exp(box_l) * anchor_l;
    *(output_features+line_idx*7+5) = exp(box_h) * anchor_h;

    // rg = rt + ra

    *(output_features+line_idx*7+6) = box_r + anchor_r;

}

cudaError_t decode_launch(float * features, float* output_features, int feature_map_height, int feature_map_width,
                int direction_angle_num,  cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((feature_map_height*feature_map_width*direction_angle_num+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  decode_kernel<<<blocks, threads, 0, stream>>>
       (features,output_features);
  cudaError_t err = cudaGetLastError();
  return err;
}


int GenerateAnchorDecodePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];

    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    float * features = const_cast<float *>((const float *)inputs[0]);
    //TRT-output
    float *output_features = (float *)(outputs[0]);
   

    // init output
    unsigned int output_features_size = batchSize * feature_map_height_*feature_map_width_*
                                    direction_angle_num_ * 7 * sizeof(float);
    
    checkCudaErrors(cudaMemsetAsync(output_features, 0, output_features_size, stream));
    
    checkCudaErrors(generate_anchor_launch(
          output_features,min_x_range_,max_x_range_,min_y_range_,max_y_range_,min_z_range_,max_z_range_,
          feature_map_height_,feature_map_width_,car_length_,car_width_,car_height_,direction_angle_num_,
          direction_angle_0_,direction_angle_1_, stream));

    checkCudaErrors(decode_launch(
          features,output_features,feature_map_height_,feature_map_width_,direction_angle_num_, stream));
    // cout << "generate anchor finished" << std::endl;
    return 0;
}


nvinfer1::DataType GenerateAnchorDecodePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    return inputTypes[0];
}

const char* GenerateAnchorDecodePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* GenerateAnchorDecodePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int GenerateAnchorDecodePlugin::getNbOutputs() const noexcept
{
    return 1;
}

int GenerateAnchorDecodePlugin::initialize() noexcept
{
    return 0;
}

void GenerateAnchorDecodePlugin::terminate() noexcept
{
}


size_t GenerateAnchorDecodePlugin::getSerializationSize() const noexcept
{
    return  3 * sizeof(int)+11*sizeof(float);
}

void GenerateAnchorDecodePlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<float>(d, min_x_range_);
    writeToBuffer<float>(d, max_x_range_);
    writeToBuffer<float>(d, min_y_range_);
    writeToBuffer<float>(d, max_y_range_);
    writeToBuffer<float>(d, min_z_range_);
    writeToBuffer<float>(d, max_y_range_);

    writeToBuffer<int>(d, feature_map_height_);
    writeToBuffer<int>(d, feature_map_width_);

    writeToBuffer<float>(d, car_length_);
    writeToBuffer<float>(d, car_width_);
    writeToBuffer<float>(d, car_height_);
    writeToBuffer<float>(d, direction_angle_0_);
    writeToBuffer<float>(d, direction_angle_1_);

    writeToBuffer<int>(d, direction_angle_num_);
}

void GenerateAnchorDecodePlugin::destroy() noexcept
{
    delete this;
}

void GenerateAnchorDecodePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GenerateAnchorDecodePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



GenerateAnchorDecodePluginCreator::GenerateAnchorDecodePluginCreator()
{

    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("point_cloud_range", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("feature_map_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("car_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("direction_angle", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("direction_angle_num", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* GenerateAnchorDecodePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* GenerateAnchorDecodePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* GenerateAnchorDecodePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* GenerateAnchorDecodePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    float min_x_range = 0.0;
    float max_x_range = 1000.0;
    float min_y_range = 0.0;
    float max_y_range = 1000.0;
    float min_z_range = 0.0;
    float max_z_range = 1000.0;

    int feature_map_height = 0;
    int feature_map_width = 0;

    float car_length = 0.0;
    float car_width  = 0.0;
    float car_height = 0.0;

    float direction_angle_0 = 0.0;
    float direction_angle_1 = 0.0;

    int direction_angle_num = 0;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "point_cloud_range"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            min_x_range = d[0];
            min_y_range = d[1];
            min_z_range = d[2];
            max_x_range = d[3];
            max_y_range = d[4];
            max_z_range = d[5];
        }
        else if(!strcmp(attr_name, "feature_map_size"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_map_height = d[0];
            feature_map_width = d[1];
        }
        else if (!strcmp(attr_name, "car_size"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            car_length = d[0];
            car_width = d[1];
            car_height = d[2];
        }
          else if (!strcmp(attr_name, "direction_angle"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            direction_angle_0 = d[0];
            direction_angle_1 = d[1];
        }
        else if(!strcmp(attr_name, "direction_angle_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            direction_angle_num = d[0];
        }
    }
    
    IPluginV2DynamicExt* plugin = new GenerateAnchorDecodePlugin(min_x_range,max_x_range,min_y_range,max_y_range,
                min_z_range,max_z_range,feature_map_height,feature_map_width,car_length,car_width,car_height,
                direction_angle_0,direction_angle_1,direction_angle_num);
    return plugin;
}

IPluginV2* GenerateAnchorDecodePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new GenerateAnchorDecodePlugin(serialData, serialLength);
}

void GenerateAnchorDecodePluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* GenerateAnchorDecodePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

GenerateAnchorDecodePluginCreator::~GenerateAnchorDecodePluginCreator()
{
   
}