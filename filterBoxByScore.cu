#include "filterBoxByScore.h"

using namespace nvinfer1;
using nvinfer1::FilterBoxByScorePlugin;
using nvinfer1::FilterBoxByScorePluginCreator;
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
static const char* PLUGIN_NAME{"FilterBoxByScorePlugin"};


// Static class fields initialization
PluginFieldCollection FilterBoxByScorePluginCreator::mFC{};
std::vector<PluginField> FilterBoxByScorePluginCreator::mPluginAttributes;

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

__device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

// create the plugin at runtime from a byte stream
FilterBoxByScorePlugin::FilterBoxByScorePlugin(int feature_map_height, int feature_map_width, int direction_angle_num,
                        int max_box_num, float score_threshold,float direction_offset)
: feature_map_height_(feature_map_height), feature_map_width_(feature_map_width), direction_angle_num_(direction_angle_num),
    max_box_num_(max_box_num),score_threshold_(score_threshold),direction_offset_(direction_offset)
{   
}

FilterBoxByScorePlugin::FilterBoxByScorePlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    
    feature_map_height_ = readFromBuffer<int>(d);
    feature_map_width_ = readFromBuffer<int>(d);

    direction_angle_num_ = readFromBuffer<int>(d);

    max_box_num_ = readFromBuffer<int>(d);
    score_threshold_ = readFromBuffer<float>(d);
    direction_offset_ = readFromBuffer<float>(d);

}

IPluginV2DynamicExt* FilterBoxByScorePlugin::clone() const noexcept
{
    auto* plugin = new FilterBoxByScorePlugin(feature_map_height_,feature_map_width_,direction_angle_num_,
                        max_box_num_,score_threshold_,direction_offset_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs FilterBoxByScorePlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    auto line_num = inputs[0].d[1];
    auto dim_num_0 = inputs[0].d[2];
    auto dim_num_1 = inputs[1].d[2];
    auto dim_num_2 = inputs[2].d[2];
    auto dim_num_3 = inputs[3].d[2];

    // std::cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" << std::endl;
    // std::cout << batch_size->getConstantValue() << " " << line_num->getConstantValue() << " " 
    //         << dim_num_0->getConstantValue() << " " << dim_num_1->getConstantValue() << " "
    //         << dim_num_2->getConstantValue() << " " 
    //         << dim_num_3->getConstantValue() << std::endl; 
   
    if (outputIndex == 0)  // box_preds
    {
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(max_box_num_);
        dim0.d[2] = exprBuilder.constant(dim_num_0->getConstantValue()+2); // 9
        return dim0; 
    }

      if (outputIndex == 1) // cls_preds
    {
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 2;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(max_box_num_);
       
        return dim1; 
    }

      if (outputIndex == 2) // iou_preds
    {
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 3;
        dim2.d[0] = batch_size;
        dim2.d[1] = exprBuilder.constant(max_box_num_);
        dim2.d[2] = dim_num_2;
        return dim2; 
    }
      if (outputIndex == 3) // dir_preds
    {
        nvinfer1::DimsExprs dim3{};
        dim3.nbDims = 3;
        dim3.d[0] = batch_size;
        dim3.d[1] = exprBuilder.constant(max_box_num_);
        dim3.d[2] = exprBuilder.constant(1);
        return dim3; 
    }
     if (outputIndex == 4) // valid_line_num
    {
        nvinfer1::DimsExprs dim4{};
        dim4.nbDims = 2;
        dim4.d[0] = batch_size;
        dim4.d[1] = exprBuilder.constant(1);
        return dim4; 
    }
}

bool FilterBoxByScorePlugin::supportsFormatCombination(
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
    if (pos == 2)       
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)      
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)       
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 5)      
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 6)       
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 7)      
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 8)      
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void FilterBoxByScorePlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
}

size_t FilterBoxByScorePlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return 0;
}
   

__global__ void filter_box_by_score_kernel(float *box_preds,float *cls_preds,float *iou_preds,float *dir_preds,
          float* output_box_preds,float *output_cls_preds,float* output_iou_preds,float *output_dir_preds,
          unsigned int *valid_line_num,
          int direction_angle_num,
          float score_threshold,float direction_offset)
{
    // printf("point_size:%d\n",*points_size);
    int line_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // first get score
    float score = Logist(*(cls_preds+line_idx));
    

    if(score >= score_threshold)
    {
        int index = atomicAdd(valid_line_num,1);

    
        // iou_preds = (iou_preds.squeeze() + 1) * 0.5
        //         top_scores *= torch.pow(iou_preds.masked_select(top_scores_keep), 4)

        float iou_preds_value = (*(iou_preds+line_idx) + 1) * 0.5;

        *(output_iou_preds+index) = iou_preds_value;

        *(output_cls_preds+index) = score*iou_preds_value*iou_preds_value
                                         *iou_preds_value*iou_preds_value;


        // torch.max 
        int max_index =0;
        float max_value = -1000000.0;
        for(int i =0;i < direction_angle_num;i++)
        {
            if (*(dir_preds+line_idx*2+i) >= max_value)
            {
                max_value = *(dir_preds+line_idx*2+i);
                max_index = i;
            }
        }
        *(output_dir_preds+index) = max_index;

        *(output_box_preds+index*9+0) = *(box_preds+line_idx*7+0);
        *(output_box_preds+index*9+1) = *(box_preds+line_idx*7+1);
        *(output_box_preds+index*9+2) = *(box_preds+line_idx*7+2);
        *(output_box_preds+index*9+3) = *(box_preds+line_idx*7+3);
        *(output_box_preds+index*9+4) = *(box_preds+line_idx*7+4);
        *(output_box_preds+index*9+5) = *(box_preds+line_idx*7+5);

    
        if(((max_index == 1) && (*(box_preds+line_idx*7+6)-direction_offset<0)) ||
                ((max_index == 0) && (*(box_preds+line_idx*7+6)-direction_offset>0)))
        // if((max_index == 1) && (*(box_preds+line_idx*7+6)-direction_offset<0))
        {
            *(output_box_preds+index*9+6) = *(box_preds+line_idx*7+6) + 3.1415926f;
        }
        else{
            *(output_box_preds+index*9+6) = *(box_preds+line_idx*7+6);
        }

        *(output_box_preds+index*9+7) = 0.0;
        *(output_box_preds+index*9+8) = *(output_cls_preds+index);

    }
}


cudaError_t filter_box_by_score_launch(float* box_preds,float *cls_preds,float *iou_preds,float *dir_preds,
          float *output_box_preds,float *output_cls_preds,float *output_iou_preds,float *output_dir_preds,
          unsigned int* valid_line_num,
          int feature_map_height,int feature_map_width,int direction_angle_num,
          float score_threshold,float direction_offset, cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((feature_map_height*feature_map_width*direction_angle_num+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  filter_box_by_score_kernel<<<blocks, threads, 0, stream>>>
       (box_preds,cls_preds,iou_preds,dir_preds,
          output_box_preds,output_cls_preds,output_iou_preds,output_dir_preds,
          valid_line_num,
          direction_angle_num,
          score_threshold,direction_offset);
  cudaError_t err = cudaGetLastError();
  return err;
}

int FilterBoxByScorePlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];

    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    float * box_preds = const_cast<float *>((const float *)inputs[0]);
    float * cls_preds = const_cast<float *>((const float *)inputs[1]);
    float * iou_preds = const_cast<float *>((const float *)inputs[2]);
    float * dir_preds = const_cast<float *>((const float *)inputs[3]);
    //TRT-output
    float *output_box_preds = (float *)(outputs[0]);
    float *output_cls_preds = (float *)(outputs[1]);
    float *output_iou_preds = (float *)(outputs[2]);
    float *output_dir_preds = (float *)(outputs[3]);
    unsigned int *valid_line_num = (unsigned int*)(outputs[4]);

    // init output
    unsigned int output_box_preds_data_size = batchSize * max_box_num_ * 9 * sizeof(float);
    checkCudaErrors(cudaMemsetAsync(output_box_preds, 0, output_box_preds_data_size, stream));

    unsigned int output_cls_preds_data_size = batchSize * max_box_num_ * 1 * sizeof(float);
    checkCudaErrors(cudaMemsetAsync(output_cls_preds, 0, output_cls_preds_data_size, stream));

    unsigned int output_iou_preds_data_size = batchSize * max_box_num_ * 1 * sizeof(float);
    checkCudaErrors(cudaMemsetAsync(output_iou_preds, 0, output_iou_preds_data_size, stream));

    unsigned int output_dir_preds_data_size = batchSize * max_box_num_ * 1 * sizeof(float);
    checkCudaErrors(cudaMemsetAsync(output_dir_preds, 0, output_dir_preds_data_size, stream));

    unsigned int valid_line_num_data_size = batchSize * 1 * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(valid_line_num, 0, valid_line_num_data_size, stream));
    
    
    checkCudaErrors(filter_box_by_score_launch(
          box_preds,cls_preds,iou_preds,dir_preds,
          output_box_preds,output_cls_preds,output_iou_preds,output_dir_preds,valid_line_num,
          feature_map_height_,feature_map_width_,direction_angle_num_,
          score_threshold_,direction_offset_, stream));

    // cout << "filter box by score finished" << std::endl;
    return 0;
}


nvinfer1::DataType FilterBoxByScorePlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{   
    if (index == 4)
        return nvinfer1::DataType::kINT32;
    return inputTypes[0];
}

const char* FilterBoxByScorePlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* FilterBoxByScorePlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int FilterBoxByScorePlugin::getNbOutputs() const noexcept
{
    return 5;
}

int FilterBoxByScorePlugin::initialize() noexcept
{
    return 0;
}

void FilterBoxByScorePlugin::terminate() noexcept
{
}


size_t FilterBoxByScorePlugin::getSerializationSize() const noexcept
{
    return  4 * sizeof(int)+2*sizeof(float);
}

void FilterBoxByScorePlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, feature_map_height_);
    writeToBuffer<int>(d, feature_map_width_);
    writeToBuffer<int>(d, direction_angle_num_);
    writeToBuffer<int>(d, max_box_num_);
    writeToBuffer<float>(d, score_threshold_);
    writeToBuffer<float>(d, direction_offset_);

}

void FilterBoxByScorePlugin::destroy() noexcept
{
    delete this;
}

void FilterBoxByScorePlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* FilterBoxByScorePlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}



FilterBoxByScorePluginCreator::FilterBoxByScorePluginCreator()
{

    mPluginAttributes.clear();

    mPluginAttributes.emplace_back(PluginField("feature_map_size", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("direction_angle_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_box_num", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("score_threshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("direction_offset", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* FilterBoxByScorePluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* FilterBoxByScorePluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* FilterBoxByScorePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* FilterBoxByScorePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;

    int feature_map_height = 0;
    int feature_map_width = 0;

    int direction_angle_num = 0;

    int max_box_num = 0;

    float score_threshold = 0.0;

    float direction_offset = 0.0;


    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if(!strcmp(attr_name, "feature_map_size"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            feature_map_height = d[0];
            feature_map_width = d[1];
        }
       
        else if(!strcmp(attr_name, "direction_angle_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            direction_angle_num = d[0];
        }
         else if(!strcmp(attr_name, "max_box_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_box_num = d[0];
        }

         else if (!strcmp(attr_name, "score_threshold"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            score_threshold = d[0];
        }
          else if (!strcmp(attr_name, "direction_offset"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            direction_offset = d[0];
        }
    }
    // std::cout << "filter box by score    " <<feature_map_height << " " << feature_map_width << " "
    //  << direction_angle_num << " " << max_box_num << " " << score_threshold << " " << direction_offset << " "
    //  << nbFields << std::endl;
    IPluginV2DynamicExt* plugin = new FilterBoxByScorePlugin(feature_map_height,feature_map_width,direction_angle_num,
                        max_box_num,score_threshold,direction_offset);
    return plugin;
}

IPluginV2* FilterBoxByScorePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new FilterBoxByScorePlugin(serialData, serialLength);
}

void FilterBoxByScorePluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* FilterBoxByScorePluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

FilterBoxByScorePluginCreator::~FilterBoxByScorePluginCreator()
{
   
}