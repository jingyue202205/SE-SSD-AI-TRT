#include "voxelGenerator.h"

using namespace nvinfer1;
using nvinfer1::VoxelGeneratorPlugin;
using nvinfer1::VoxelGeneratorPluginCreator;
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
static const char* PLUGIN_NAME{"VoxelGeneratorPlugin"};

// Static class fields initialization
PluginFieldCollection VoxelGeneratorPluginCreator::mFC{};
std::vector<PluginField> VoxelGeneratorPluginCreator::mPluginAttributes;

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
VoxelGeneratorPlugin::VoxelGeneratorPlugin(
int max_voxels, int max_points, int voxel_features, float x_min,
float x_max, float y_min, float y_max, float z_min, float z_max,
float voxel_x, float voxel_y, float voxel_z
) : voxelNum_(max_voxels), pointNum_(max_points), featureNum_(voxel_features),
    min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    voxel_x_size_(voxel_x), voxel_y_size_(voxel_y),
    voxel_z_size_(voxel_z)
{
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(
    int max_voxels, int max_points, int voxel_features, float x_min,
    float x_max, float y_min, float y_max, float z_min, float z_max,
    float voxel_x, float voxel_y, float voxel_z, int point_features,
    int grid_x, int grid_y, int grid_z
) : voxelNum_(max_voxels), pointNum_(max_points), featureNum_(voxel_features),
    min_x_range_(x_min), max_x_range_(x_max), min_y_range_(y_min),
    max_y_range_(y_max), min_z_range_(z_min), max_z_range_(z_max),
    voxel_x_size_(voxel_x), voxel_y_size_(voxel_y),
    voxel_z_size_(voxel_z), pointFeatureNum_(point_features),
    grid_x_size_(grid_x), grid_y_size_(grid_y), grid_z_size_(grid_z)
{
}

VoxelGeneratorPlugin::VoxelGeneratorPlugin(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    voxelNum_ = readFromBuffer<int>(d);
    pointNum_ = readFromBuffer<int>(d);
    featureNum_ = readFromBuffer<int>(d);
    min_x_range_ = readFromBuffer<float>(d);
    max_x_range_ = readFromBuffer<float>(d);
    min_y_range_ = readFromBuffer<float>(d);
    max_y_range_ = readFromBuffer<float>(d);
    min_z_range_ = readFromBuffer<float>(d);
    max_z_range_ = readFromBuffer<float>(d);
    voxel_x_size_ = readFromBuffer<float>(d);
    voxel_y_size_ = readFromBuffer<float>(d);
    voxel_z_size_ = readFromBuffer<float>(d);
    pointFeatureNum_ = readFromBuffer<int>(d);
    grid_x_size_ = readFromBuffer<int>(d);
    grid_y_size_ = readFromBuffer<int>(d);
    grid_z_size_ = readFromBuffer<int>(d);
}

IPluginV2DynamicExt* VoxelGeneratorPlugin::clone() const noexcept
{
    auto* plugin = new VoxelGeneratorPlugin(voxelNum_, pointNum_, featureNum_, min_x_range_, max_x_range_,
        min_y_range_, max_y_range_, min_z_range_, max_z_range_, voxel_x_size_, voxel_y_size_, voxel_z_size_,
        pointFeatureNum_, grid_x_size_, grid_y_size_, grid_z_size_);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

nvinfer1::DimsExprs VoxelGeneratorPlugin::getOutputDimensions(
int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // assert(outputIndex >= 0 && outputIndex < this->getNbOutputs());
    auto batch_size = inputs[0].d[0];
    // auto batch_size = 1;
    // std::cout  << inputs[0].nbDims << " " << inputs[0].d[0]->getConstantValue() << " " << inputs[0].d[1]->getConstantValue() << " " << inputs[0].d[2]->getConstantValue() << std::endl;
    // std::cout  << inputs[1].nbDims << " " << inputs[1].d[0]->getConstantValue() << std::endl;
    if (outputIndex == 0)
    {
        // std::cout << "batch_size: " << batch_size->getConstantValue() << " voxel_num: " << voxelNum_ << " featurennum_: " << featureNum_ << std::endl;
        nvinfer1::DimsExprs dim0{};
        dim0.nbDims = 3;
        dim0.d[0] = batch_size;
        dim0.d[1] = exprBuilder.constant(voxelNum_);
        dim0.d[2] = exprBuilder.constant(featureNum_);
        return dim0; // voxels 1 20000 4
    }
    if(outputIndex == 1){
        // std::cout << "batch_size: " << batch_size->getConstantValue() << "  voxel_num: " << voxelNum_ << " featurennum_: " << 4 << std::endl;
        nvinfer1::DimsExprs dim1{};
        dim1.nbDims = 3;
        dim1.d[0] = batch_size;
        dim1.d[1] = exprBuilder.constant(voxelNum_);
        dim1.d[2] = exprBuilder.constant(4);
        return dim1; // coors 1 20000 4
    }
    if(outputIndex == 2)
    {   
        // std::cout << "batch_size: " << batch_size->getConstantValue() << std::endl;
        nvinfer1::DimsExprs dim2{};
        dim2.nbDims = 1;
        dim2.d[0] = batch_size;
        return dim2;
    }
}

bool VoxelGeneratorPlugin::supportsFormatCombination(
int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    // PLUGIN_ASSERT(nbInputs == 2);
    // PLUGIN_ASSERT(nbOutputs == 2);
    const PluginTensorDesc& in = inOut[pos];
    if (pos == 0)       // PointCloud Array --- x, y, z, i   dim: 1  40000 4
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 1)       // Point Num
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 2)       // voxels, dim: 1 20000 4
    {
        return (in.type == nvinfer1::DataType::kFLOAT) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 3)       // voxelCoords, dim: 1 x 20000 x 4
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    if (pos == 4)    // voxel_num valid
    {
        return (in.type == nvinfer1::DataType::kINT32) && (in.format == TensorFormat::kLINEAR);
    }
    
    return false;
}

void VoxelGeneratorPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    pointFeatureNum_ = in[0].desc.dims.d[2];
    grid_x_size_ = np_round((max_x_range_ - min_x_range_) / voxel_x_size_);
    grid_y_size_ = np_round((max_y_range_ - min_y_range_) / voxel_y_size_);
    grid_z_size_ = np_round((max_z_range_ - min_z_range_) / voxel_z_size_);
}

size_t VoxelGeneratorPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    int batchSize = inputs[0].dims.d[0];
    size_t coor_to_voxelidx_size = batchSize * grid_z_size_ * grid_y_size_ * grid_x_size_ * 2 * sizeof(unsigned int);
    size_t num_points_per_voxel_size = batchSize * voxelNum_ * 1 * sizeof(unsigned int);
    // size_t grid_voxel_size = batchSize * grid_z_size_ * grid_y_size_ * grid_x_size_ * pointNum_ * pointFeatureNum_ * sizeof(float);
    size_t voxel_size = batchSize * voxelNum_ * pointNum_ * featureNum_ * sizeof(float);

    size_t workspaces[3];
    workspaces[0] = coor_to_voxelidx_size;
    workspaces[1] = num_points_per_voxel_size;
    workspaces[2] = voxel_size;
   
    return  calculateTotalWorkspaceSize(workspaces, 3);
}

// 1 N 5 4 ---> 1 N 4
__global__ void generateAverage_kernel(float *voxel_,
        float *voxel_features_data,unsigned int* num_points_per_voxel,unsigned int *voxel_num_data)
{
    int voxelidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(voxelidx >= *voxel_num_data) return;

    float4 point = ((float4*)voxel_)[voxelidx*MAX_POINTS_PER_VOXEL];
    // float4 point = ((float4*)points)[point_idx];
    int index_z = floorf((point.z - (-3.0)) / 0.1);
    int index_y = floorf((point.y - (-40.)) / 0.05);
    int index_x = floorf((point.x - 0.0)/0.05);

    float sum_x = 0.0;
    float sum_y = 0.0;
    float sum_z = 0.0;
    float sum_i = 0.0;

    int num_point = num_points_per_voxel[voxelidx];

    sum_x = (*(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+0*4) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+1*4) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+2*4) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+3*4) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+4*4))/num_point;

    sum_y = (*(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+0*4 + 1) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+1*4 + 1) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+2*4 + 1) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+3*4 + 1) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+4*4 + 1))/num_point;

    sum_z = (*(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+0*4 + 2) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+1*4 + 2) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+2*4 + 2) + 
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+3*4 + 2) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+4*4 + 2))/num_point;

    sum_i = (*(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+0*4 + 3) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+1*4 + 3) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+2*4 + 3) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+3*4 + 3) +
            *(voxel_+ voxelidx*MAX_POINTS_PER_VOXEL*4+4*4 + 3))/num_point;
    
    #if 0
     if (index_z == 32 && index_y == 731 && index_x == 95)
     {
         printf("sum: %f,%f,%f,%f\n",sum_x,sum_y,sum_z,sum_i);
     }
     #endif

    // float x = sum_x/num_point;
    // float y = sum_y/num_point;
    // float z = sum_z/num_point;
    // float inten = sum_i / num_point;

    float *address = voxel_features_data + voxelidx*4;
    atomicExch(address+0,sum_x);
    atomicExch(address+1,sum_y);
    atomicExch(address+2,sum_z);
    atomicExch(address+3,sum_i);
    #if 0
    if (index_z == 32 && index_y == 731 && index_x == 95)
     {
         printf("address_sum: %f,%f,%f,%f\n",*(address),*(address+1),*(address+2),*(address+3));
     }
    #endif
}


cudaError_t generateAverage_launch(float *voxel_,
        float *voxel_features_data,unsigned int* num_points_per_voxel,unsigned int* voxel_num_data,
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;
  dim3 blocks((MAX_VOXELS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateAverage_kernel<<<blocks, threads, 0, stream>>>
       (voxel_,voxel_features_data,num_points_per_voxel,voxel_num_data);
  cudaError_t err = cudaGetLastError();
  return err;
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


__global__ void generateVoxels_kernel(float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_x_size, int grid_y_size,int grid_z_size,
        unsigned int *coor_to_voxelidx,unsigned int *coords_data, float* voxel_,
        unsigned int *num_points_per_voxel,unsigned int *voxel_num_data)
{
    // printf("point_size:%d\n",*points_size);
    int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(point_idx >= *points_size) return;
    // printf("generatevoxel11111111");
    float4 point = ((float4*)points)[point_idx];
    
    if( !(point.x >= min_x_range && point.x < max_x_range
        && point.y >= min_y_range && point.y < max_y_range
        && point.z >= min_z_range && point.z < max_z_range) ) {
      return;
    }
    // printf("generatevoxel222222\n");
    int index_x = floorf((point.x - min_x_range) / voxel_x_size);
    int index_y = floorf((point.y - min_y_range) / voxel_y_size);
    int index_z = floorf((point.z - min_z_range) / voxel_z_size);
    // printf("index_x: %d,%f,%f,%f\n",index_x,point.x,min_x_range,voxel_x_size);
    // printf("index_i: %d,%d,%d\n",index_x,index_y,index_z);
    // bool failed = false;
    if ((index_x < 0) or (index_x >= grid_x_size) or (index_y < 0) or (index_y >= grid_y_size) or (index_z < 0) or (index_z >= grid_z_size))
        return;
    
    unsigned int voxel_index = index_z * (grid_y_size * grid_x_size) + index_y * grid_x_size + index_x;
                                // index_z * (grid_y_size * grid_x_size * 2) + index_y*grid_z_size *2 + index_x*2 + 
    unsigned int point_id = atomicAdd(coor_to_voxelidx+voxel_index*2,1);


    if(point_id >= MAX_POINTS_PER_VOXEL) return;

    unsigned int current_voxelid = 0;

    if (point_id == 0)
    {
        //保存coor and current_voxel_id
        current_voxelid = atomicAdd(voxel_num_data,1);
        #if 0
        if(current_voxelid == 0)
        {
            printf("current_voxelid:%d,%f,%f,%f,%d,%d,%d,%d\n",current_voxelid,point.x,point.y,point.z,index_x,index_y,index_z,voxel_index * 2 + 1);
        }
        #endif

        //save current_voxelid
        unsigned int *current_voxelid_address = coor_to_voxelidx + voxel_index * 2 + 1;
        atomicExch(current_voxelid_address,current_voxelid);


        //save coord
        uint4 coord = {0,index_z,index_y,index_x};
        ((uint4*)coords_data)[current_voxelid] = coord;

    }
    // nanosleep()
    else{
        // if(current_voxelid == 0 && )
        // __nanosleep(100);
        current_voxelid = coor_to_voxelidx[voxel_index*2+1];
        if(current_voxelid == 0)
        {
            cuda_sleep(300000); //10000000
            current_voxelid = coor_to_voxelidx[voxel_index*2+1];
        }
        #if 0
         if(current_voxelid == 0)
        {
            printf("current_voxelid:%d,%f,%f,%f,%d,%d,%d,%d,%d\n",current_voxelid,point.x,point.y,point.z,index_x,index_y,index_z,point_id,voxel_index*2+1);
        }
        #endif
    }
    //save point
    float *address = voxel_ + current_voxelid*MAX_POINTS_PER_VOXEL*4+point_id*4;
    atomicExch(address+0,point.x);
    atomicExch(address+1,point.y);
    atomicExch(address+2,point.z);
    atomicExch(address+3,point.w);

     //num_points_per_voxel ++
    unsigned int *num_points_per_voxel_address = num_points_per_voxel + current_voxelid;
    unsigned int num = *(coor_to_voxelidx+voxel_index*2);
    if(num > MAX_POINTS_PER_VOXEL)
        num = MAX_POINTS_PER_VOXEL;
    atomicExch(num_points_per_voxel_address,num);
    // [  0  27 843  72]
    #if 0
    if (index_z == 32 && index_y == 731 && index_x == 95)
    {
        printf("point: %f,%f,%f,%f %d\n",point.x,point.y,point.z,point.w,point_id);
        printf("adress: %f,%f,%f,%f\n",*address,*(address+1),*(address+2),*(address+3));
        printf("fsfsfsfsf current_voxelid: %d\n",current_voxelid);
        printf("coor_to_voxelidx[voxel_index*2]: %d\n",atomicAdd(coor_to_voxelidx+voxel_index*2,0));
        printf("point_id: %d\n",point_id);
        printf("num_points_per_voxel_address: %d\n",*num_points_per_voxel_address);
    }
    #endif

}

cudaError_t generateVoxels_launch(float *points, unsigned int* points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float voxel_x_size, float voxel_y_size, float voxel_z_size,
        int grid_x_size, int grid_y_size,int grid_z_size,
        unsigned int *coor_to_voxelidx,unsigned int *coords_data, float *voxel_,
        unsigned int *num_points_per_voxel,unsigned int * voxel_num_data, 
        cudaStream_t stream)
{
  int threadNum = THREADS_FOR_VOXEL;

  dim3 blocks((MAX_POINTS+threadNum-1)/threadNum);
  dim3 threads(threadNum);
  generateVoxels_kernel<<<blocks, threads, 0, stream>>>
       (points, points_size,
        min_x_range, max_x_range,
        min_y_range, max_y_range,
        min_z_range, max_z_range,
        voxel_x_size, voxel_y_size, voxel_z_size,
        grid_x_size, grid_y_size,grid_z_size,
        coor_to_voxelidx,coords_data,voxel_,num_points_per_voxel,voxel_num_data);
  cudaError_t err = cudaGetLastError();
  return err;
}


int VoxelGeneratorPlugin::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    int batchSize = inputDesc[0].dims.d[0];
    // int maxNumPoints = inputDesc[0].dims.d[1];
    //TRT-input
    // std::cout << "voxelgenerator batch_size: " << batchSize << std::endl;
    float * pointCloud = const_cast<float *>((const float *)inputs[0]);
    unsigned int* pointNum = const_cast<unsigned int *>((const unsigned int *)inputs[1]);
    //TRT-output
    float *voxel_features_data = (float *)(outputs[0]);
    unsigned int *coords_data = (unsigned int *)(outputs[1]);
    unsigned int *voxel_num_data = (unsigned int *)(outputs[2]);
    // unsigned int *params_data = (unsigned int *)(outputs[2]);
    int dense_voxel_num = grid_z_size_ * grid_y_size_ * grid_x_size_;
    size_t coor_to_voxelidx_size = batchSize * dense_voxel_num * 2* sizeof(unsigned int);
    size_t num_points_per_voxel_size = batchSize * voxelNum_ * 1 * sizeof(unsigned int);
    // size_t grid_voxel_size = batchSize * grid_z_size_ * grid_y_size_ * grid_x_size_ * pointNum_ * featureNum_ * sizeof(float);
    size_t voxel_size = batchSize * voxelNum_ * pointNum_ * featureNum_ * sizeof(float);
    size_t workspaces[3];
    workspaces[0] = coor_to_voxelidx_size;
    workspaces[1] = num_points_per_voxel_size;
    workspaces[2] = voxel_size;
    size_t total_workspace = calculateTotalWorkspaceSize(workspaces, 3);
    // std::cout << "enqueue11111" << std::endl;
    unsigned int* coor_to_voxelidx = static_cast<unsigned int*>(workspace);
    unsigned int* num_points_per_voxel = reinterpret_cast<unsigned int*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(coor_to_voxelidx), coor_to_voxelidx_size)
    );
    // float* grid_voxel_ = reinterpret_cast<float*>(
    //     nextWorkspacePtr(reinterpret_cast<int8_t*>(num_points_per_voxel), num_points_per_voxel_size)
    // );
    float* voxel_ = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(num_points_per_voxel), num_points_per_voxel_size)
    );
    // Initialize workspace memory
    checkCudaErrors(cudaMemsetAsync(coor_to_voxelidx, 0, total_workspace, stream)); // total_workspace
    // checkCudaErrors(cudaMemsetAsync(num_points_per_voxel,0,num_points_per_voxel_size,stream));
    // checkCudaErrors(cudaMemsetAsync(voxel_,0,voxel_size,stream));
    unsigned int voxel_features_data_size = batchSize * voxelNum_  * featureNum_ * sizeof(float);
    unsigned int coords_data_size = batchSize * voxelNum_ * 4 * sizeof(unsigned int);
    unsigned int voxel_num_data_size = batchSize * sizeof(unsigned int);
    // unsigned int params_data_size = batchSize * sizeof(unsigned int);
    checkCudaErrors(cudaMemsetAsync(voxel_features_data, 0, voxel_features_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(coords_data, 0, coords_data_size, stream));
    checkCudaErrors(cudaMemsetAsync(voxel_num_data, 0, voxel_num_data_size, stream));
    // checkCudaErrors(cudaMemsetAsync(params_data, 0, params_data_size, stream));

    checkCudaErrors(generateVoxels_launch(
          pointCloud, pointNum,
          min_x_range_, max_x_range_,
          min_y_range_, max_y_range_,
          min_z_range_, max_z_range_,
          voxel_x_size_, voxel_y_size_, voxel_z_size_,
          grid_x_size_, grid_y_size_, grid_z_size_,
          coor_to_voxelidx,coords_data, voxel_,num_points_per_voxel,voxel_num_data, stream));

    
    checkCudaErrors(generateAverage_launch(
        voxel_,voxel_features_data,num_points_per_voxel,voxel_num_data, stream));
    return 0;
}

nvinfer1::DataType VoxelGeneratorPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if(index == 0)
      return inputTypes[0];
    return inputTypes[1];
}

const char* VoxelGeneratorPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const char* VoxelGeneratorPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int VoxelGeneratorPlugin::getNbOutputs() const noexcept
{
    return 3;
}

int VoxelGeneratorPlugin::initialize() noexcept
{
    return 0;
}

void VoxelGeneratorPlugin::terminate() noexcept
{
}

size_t VoxelGeneratorPlugin::getSerializationSize() const noexcept
{
    return 9 * sizeof(float) + 7 * sizeof(int);
}

void VoxelGeneratorPlugin::serialize(void* buffer) const noexcept
{
    char* d = reinterpret_cast<char*>(buffer);
    writeToBuffer<int>(d, voxelNum_);
    writeToBuffer<int>(d, pointNum_);
    writeToBuffer<int>(d, featureNum_);
    writeToBuffer<float>(d, min_x_range_);
    writeToBuffer<float>(d, max_x_range_);
    writeToBuffer<float>(d, min_y_range_);
    writeToBuffer<float>(d, max_y_range_);
    writeToBuffer<float>(d, min_z_range_);
    writeToBuffer<float>(d, max_z_range_);
    writeToBuffer<float>(d, voxel_x_size_);
    writeToBuffer<float>(d, voxel_y_size_);
    writeToBuffer<float>(d, voxel_z_size_);
    writeToBuffer<int>(d, pointFeatureNum_);
    writeToBuffer<int>(d, grid_x_size_);
    writeToBuffer<int>(d, grid_y_size_);
    writeToBuffer<int>(d, grid_z_size_);
}

void VoxelGeneratorPlugin::destroy() noexcept
{
    delete this;
}

void VoxelGeneratorPlugin::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* VoxelGeneratorPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}


VoxelGeneratorPluginCreator::VoxelGeneratorPluginCreator()
{
    
    mPluginAttributes.clear();

    // std::cout <<  *max_num_points_per_voxel_ptr << std::endl;
    mPluginAttributes.emplace_back(PluginField("max_num_points_per_voxel", max_num_points_per_voxel_ptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("max_voxels", max_voxels_ptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("point_cloud_range", point_cloud_range_ptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_feature_num", voxel_feature_num_ptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("voxel_size", voxel_size_ptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* VoxelGeneratorPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const char* VoxelGeneratorPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const PluginFieldCollection* VoxelGeneratorPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2* VoxelGeneratorPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    const PluginField* fields = fc->fields;
    int nbFields = fc->nbFields;
    int max_points = 0;
    int max_voxels = 0;
    int voxel_feature_num = 0;
    float point_cloud_range[6] = {0.0f};
    float voxel_size[3] = {0.0f};
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        if (!strcmp(attr_name, "max_num_points_per_voxel"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_points = d[0];
        }
        else if (!strcmp(attr_name, "max_voxels"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            max_voxels = d[0];
        }
        else if (!strcmp(attr_name, "point_cloud_range"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            point_cloud_range[0] = d[0];
            point_cloud_range[1] = d[1];
            point_cloud_range[2] = d[2];
            point_cloud_range[3] = d[3];
            point_cloud_range[4] = d[4];
            point_cloud_range[5] = d[5];
        }
        else if (!strcmp(attr_name, "voxel_feature_num"))
        {
            const int* d = static_cast<const int*>(fields[i].data);
            voxel_feature_num = d[0];
        }
        else if (!strcmp(attr_name, "voxel_size"))
        {
            const float* d = static_cast<const float*>(fields[i].data);
            voxel_size[0] = d[0];
            voxel_size[1] = d[1];
            voxel_size[2] = d[2];
        }
    }
    // std::cout << max_voxels << " " << max_points << " " <<voxel_feature_num << " " << point_cloud_range[0] << " " << point_cloud_range[1] << " "
    // << point_cloud_range[2] << " "<< point_cloud_range[3] << " " << point_cloud_range[4] << " " << point_cloud_range[5] << " " << voxel_size[0] << " "
    // << voxel_size[1] << " " << voxel_size[2] << std::endl;
    IPluginV2DynamicExt* plugin = new VoxelGeneratorPlugin(max_voxels, max_points,voxel_feature_num, point_cloud_range[0],
        point_cloud_range[3], point_cloud_range[1], point_cloud_range[4], point_cloud_range[2],
        point_cloud_range[5], voxel_size[0], voxel_size[1], voxel_size[2]);
    return plugin;
}

IPluginV2* VoxelGeneratorPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    return new VoxelGeneratorPlugin(serialData, serialLength);
}

void VoxelGeneratorPluginCreator::setPluginNamespace(const char* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

const char* VoxelGeneratorPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

VoxelGeneratorPluginCreator::~VoxelGeneratorPluginCreator()
{
    delete max_num_points_per_voxel_ptr;
    delete max_voxels_ptr;
    delete [] voxel_size_ptr;
    delete [] point_cloud_range_ptr;
    delete voxel_feature_num_ptr;
}