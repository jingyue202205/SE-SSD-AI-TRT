#include <fstream>
#include <iostream>
#include <iomanip> //设置输出格式
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <chrono>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "params.h"
#include "submConv3dlayer.h"
#include "sparseConv3dlayer.h"
#include "voxelGenerator.h"
#include "sparse2Dense.h"
#include "zeroPad2d.h"
#include "generateAnchorDecode.h"
#include "filterBoxByScore.h"
#include <time.h>
#include <chrono>
#include <cmath>
#include <string>
#include <string.h>

using namespace nvinfer1;
using namespace std;
using namespace std::chrono;
using std::string;


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


// stuff we know about the network and the input/output blobs
const char* INPUT_POINTS = "points_data";
const char* INPUT_POINTS_SIZE = "points_size";
const char* OUTPUT_VOXELS = "voxels";
const char* OUTPUT_COORS = "coors";
const char* OUTPUT_VOXEL_NUM = "voxel_num";
const float ThresHold = 1e-8;
static Logger gLogger;


class RTLogger : public nvinfer1::ILogger {
  public:
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level message
        //if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kINFO ) {
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "trt_infer: " << msg << std::endl;
        }
    }
} rt_glogger;


//code for calculating rotated NMS come from  https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/src/postprocess.cpp
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};


inline float cross(const float2 p1, const float2 p2, const float2 p0) {
    return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

inline int check_box2d(const Bndbox box, const float2 p) {
    const float MARGIN = 1e-2;
    float center_x = box.x;
    float center_y = box.y;
    float angle_cos = cos(-box.rt);
    float angle_sin = sin(-box.rt);
    float rot_x = (p.x - center_x) * angle_cos + (p.y - center_y) * (-angle_sin);
    float rot_y = (p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos;

    return (fabs(rot_x) < box.w / 2 + MARGIN && fabs(rot_y) < box.l / 2 + MARGIN);
}

bool intersection(const float2 p1, const float2 p0, const float2 q1, const float2 q0, float2 &ans) {

    if (( std::min(p0.x, p1.x) <= std::max(q0.x, q1.x) &&
          std::min(q0.x, q1.x) <= std::max(p0.x, p1.x) &&
          std::min(p0.y, p1.y) <= std::max(q0.y, q1.y) &&
          std::min(q0.y, q1.y) <= std::max(p0.y, p1.y) ) == 0)
        return false;


    float s1 = cross(q0, p1, p0);
    float s2 = cross(p1, q1, p0);
    float s3 = cross(p0, q1, q0);
    float s4 = cross(q1, p1, q0);

    if (!(s1 * s2 > 0 && s3 * s4 > 0))
        return false;

    float s5 = cross(q1, p1, p0);
    if (fabs(s5 - s1) > ThresHold) {
        ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
        ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

    } else {
        float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
        float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
        float D = a0 * b1 - a1 * b0;

        ans.x = (b0 * c1 - b1 * c0) / D;
        ans.y = (a1 * c0 - a0 * c1) / D;
    }

    return true;
}

inline void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p) {
    float new_x = (p.x - center.x) * angle_cos + (p.y - center.y) * (-angle_sin) + center.x;
    float new_y = (p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
    p = float2 {new_x, new_y};
    return;
}

inline float box_overlap(const Bndbox &box_a, const Bndbox &box_b) {
    float a_angle = box_a.rt, b_angle = box_b.rt;
    float a_dx_half = box_a.w / 2, b_dx_half = box_b.w / 2, a_dy_half = box_a.l / 2, b_dy_half = box_b.l / 2;
    float a_x1 = box_a.x - a_dx_half, a_y1 = box_a.y - a_dy_half;
    float a_x2 = box_a.x + a_dx_half, a_y2 = box_a.y + a_dy_half;
    float b_x1 = box_b.x - b_dx_half, b_y1 = box_b.y - b_dy_half;
    float b_x2 = box_b.x + b_dx_half, b_y2 = box_b.y + b_dy_half;
    float2 box_a_corners[5];
    float2 box_b_corners[5];

    float2 center_a = float2 {box_a.x, box_a.y};
    float2 center_b = float2 {box_b.x, box_b.y};

    float2 cross_points[16];
    float2 poly_center =  {0, 0};
    int cnt = 0;
    bool flag = false;

    box_a_corners[0] = float2 {a_x1, a_y1};
    box_a_corners[1] = float2 {a_x2, a_y1};
    box_a_corners[2] = float2 {a_x2, a_y2};
    box_a_corners[3] = float2 {a_x1, a_y2};

    box_b_corners[0] = float2 {b_x1, b_y1};
    box_b_corners[1] = float2 {b_x2, b_y1};
    box_b_corners[2] = float2 {b_x2, b_y2};
    box_b_corners[3] = float2 {b_x1, b_y2};

    float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
    float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

    for (int k = 0; k < 4; k++) {
        rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
        rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
    }

    box_a_corners[4] = box_a_corners[0];
    box_b_corners[4] = box_b_corners[0];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                                box_b_corners[j + 1], box_b_corners[j],
                                cross_points[cnt]);
            if (flag) {
                poly_center = {poly_center.x + cross_points[cnt].x, poly_center.y + cross_points[cnt].y};
                cnt++;
            }
        }
    }

    for (int k = 0; k < 4; k++) {
        if (check_box2d(box_a, box_b_corners[k])) {
            poly_center = {poly_center.x + box_b_corners[k].x, poly_center.y + box_b_corners[k].y};
            cross_points[cnt] = box_b_corners[k];
            cnt++;
        }
        if (check_box2d(box_b, box_a_corners[k])) {
            poly_center = {poly_center.x + box_a_corners[k].x, poly_center.y + box_a_corners[k].y};
            cross_points[cnt] = box_a_corners[k];
            cnt++;
        }
    }

    poly_center.x /= cnt;
    poly_center.y /= cnt;

    float2 temp;
    for (int j = 0; j < cnt - 1; j++) {
        for (int i = 0; i < cnt - j - 1; i++) {
            if (atan2(cross_points[i].y - poly_center.y, cross_points[i].x - poly_center.x) >
                atan2(cross_points[i+1].y - poly_center.y, cross_points[i+1].x - poly_center.x)
                ) {
                temp = cross_points[i];
                cross_points[i] = cross_points[i + 1];
                cross_points[i + 1] = temp;
            }
        }
    }

    float area = 0;
    for (int k = 0; k < cnt - 1; k++) {
        float2 a = {cross_points[k].x - cross_points[0].x,
                    cross_points[k].y - cross_points[0].y};
        float2 b = {cross_points[k + 1].x - cross_points[0].x,
                    cross_points[k + 1].y - cross_points[0].y};
        area += (a.x * b.y - a.y * b.x);
    }
    return fabs(area) / 2.0;
}

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh, std::vector<Bndbox> &nms_pred)
{
    std::sort(bndboxes.begin(), bndboxes.end(),
              [](Bndbox boxes1, Bndbox boxes2) { return boxes1.score > boxes2.score; });
    std::vector<int> suppressed(bndboxes.size(), 0);
    for (size_t i = 0; i < bndboxes.size(); i++) {
        if (suppressed[i] == 1) {
            continue;
        }
        nms_pred.emplace_back(bndboxes[i]);
        for (size_t j = i + 1; j < bndboxes.size(); j++) {
            if (suppressed[j] == 1) {
                continue;
            }

            float sa = bndboxes[i].w * bndboxes[i].l;
            float sb = bndboxes[j].w * bndboxes[j].l;
            float s_overlap = box_overlap(bndboxes[i], bndboxes[j]);
            float iou = s_overlap / fmaxf(sa + sb - s_overlap, ThresHold);

            if (iou >= nms_thresh) {
                suppressed[j] = 1;
            }
        }
    }
    return 0;
}


// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

IPluginV2Layer* add_voxel_generator(INetworkDefinition *network,ITensor * point_data, ITensor* point_size,int max_num_points_per_voxel,
                                    int max_voxels, float voxel_size_x,float voxel_size_y,float voxel_size_z,float x_min,float x_max,
                                    float y_min,float y_max,float z_min, float z_max,int voxel_feature_num)
{

    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    float *voxel_size = (float*)malloc(3*sizeof(float));
    float *point_cloud_range = (float*)malloc(6*sizeof(float));
    voxel_size[0] = voxel_size_x;
    voxel_size[1] = voxel_size_y;
    voxel_size[2] = voxel_size_z;

    point_cloud_range[0] = x_min;
    point_cloud_range[1] = y_min;
    point_cloud_range[2] = z_min;
    point_cloud_range[3] = x_max;
    point_cloud_range[4] = y_max;
    point_cloud_range[5] = z_max;

    auto voxelGeneratorcreator = getPluginRegistry()->getPluginCreator("VoxelGeneratorPlugin", "1");
    const PluginFieldCollection* voxelGeneratorpluginData = voxelGeneratorcreator->getFieldNames();

    const PluginField* fields = voxelGeneratorpluginData->fields;
    int nbFields = voxelGeneratorpluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "max_num_points_per_voxel"))
        {
            
            new_pluginData_list.emplace_back(PluginField("max_num_points_per_voxel",  &(max_num_points_per_voxel), PluginFieldType::kINT32, 1));
          
            
        }
        else if (!strcmp(attr_name, "max_voxels"))
        {
           
            new_pluginData_list.emplace_back(PluginField("max_voxels",  &(max_voxels), PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "point_cloud_range"))
        {
           
            new_pluginData_list.emplace_back(PluginField("point_cloud_range",  point_cloud_range, PluginFieldType::kFLOAT32, 1));  
           
        }
          else if (!strcmp(attr_name, "voxel_feature_num"))
        {
            
            new_pluginData_list.emplace_back(PluginField("voxel_feature_num",  &(voxel_feature_num), PluginFieldType::kINT32, 1));  
           
        }

        else if (!strcmp(attr_name, "voxel_size"))
        {
            

            new_pluginData_list.emplace_back(PluginField("voxel_size",  voxel_size, PluginFieldType::kFLOAT32, 1));
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_voxelGenerator = voxelGeneratorcreator->createPlugin("voxelGeneratorlayer", newPluginFieldCollection);
    ITensor* inputTensors_voxelgenerator[] = {point_data,point_size};
    auto voxelGenerator = network->addPluginV2(inputTensors_voxelgenerator, 2, *pluginObj_voxelGenerator);
    pluginObj_voxelGenerator->destroy();
    free(voxel_size);
    free(point_cloud_range);
    return voxelGenerator;
}


IPluginV2Layer* add_generate_anchor_decoder(INetworkDefinition *network,ITensor * features,
                                    float min_x_range,float max_x_range,
                                    float min_y_range,float max_y_range,float min_z_range,
                                    float max_z_range, int feature_map_height, int feature_map_width, float car_length,
                                    float car_width, float car_height, float direction_angle_0, float direction_angle_1,
                                    int direction_angle_num)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    float *point_cloud_range = (float*)malloc(6*sizeof(float));
    int *feature_map_size = (int*)malloc(2*sizeof(int));
    float *car_size = (float*)malloc(3*sizeof(float));
    float *direction_angle = (float*)malloc(2*sizeof(float));

    point_cloud_range[0] = min_x_range;
    point_cloud_range[1] = min_y_range;
    point_cloud_range[2] = min_z_range;
    point_cloud_range[3] = max_x_range;
    point_cloud_range[4] = max_y_range;
    point_cloud_range[5] = max_z_range;
    
    feature_map_size[0] = feature_map_height;
    feature_map_size[1] = feature_map_width;

    car_size[0] = car_length;
    car_size[1] = car_width;
    car_size[2] = car_height;

    direction_angle[0] = direction_angle_0;
    direction_angle[1] = direction_angle_1;


    auto generateAnchorDecoderCreater = getPluginRegistry()->getPluginCreator("GenerateAnchorDecodePlugin", "1");
    const PluginFieldCollection* generateAnchorDecodepluginData = generateAnchorDecoderCreater->getFieldNames();

    const PluginField* fields = generateAnchorDecodepluginData->fields;
    int nbFields = generateAnchorDecodepluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "point_cloud_range"))
        {
          
            new_pluginData_list.emplace_back(PluginField("point_cloud_range",  point_cloud_range, PluginFieldType::kFLOAT32, 1));

        }
        else if (!strcmp(attr_name, "feature_map_size"))
        {
          
            new_pluginData_list.emplace_back(PluginField("feature_map_size",  feature_map_size, PluginFieldType::kINT32, 1)); 
             
        }
        else if (!strcmp(attr_name, "car_size"))
        {
            new_pluginData_list.emplace_back(PluginField("car_size",  car_size, PluginFieldType::kFLOAT32, 1));  
     
        }
        else if (!strcmp(attr_name, "direction_angle"))
        {
        
            new_pluginData_list.emplace_back(PluginField("direction_angle",  direction_angle, PluginFieldType::kFLOAT32, 1));   
        }

        else if (!strcmp(attr_name, "direction_angle_num"))
        {
 
            new_pluginData_list.emplace_back(PluginField("direction_angle_num",  &(direction_angle_num), PluginFieldType::kINT32, 1));
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_generateAnchorDecoder = generateAnchorDecoderCreater->createPlugin("generateAnchorDecodelayer", newPluginFieldCollection);
    ITensor* inputTensors_generateAnchorDecoder[] = {features,};
    auto generateAnchorDecoder = network->addPluginV2(inputTensors_generateAnchorDecoder, 1, *pluginObj_generateAnchorDecoder);
    pluginObj_generateAnchorDecoder->destroy();
    free(point_cloud_range);
    free(feature_map_size);
    free(car_size);
    free(direction_angle);
    return generateAnchorDecoder;
}

IPluginV2Layer* add_filter_box_by_score_layer(INetworkDefinition *network,ITensor * box_preds,ITensor * cls_preds,
                                    ITensor *iou_preds, ITensor *dir_preds,
                                    int feature_map_height, int feature_map_width, int direction_angle_num,
                                    int max_box_num, float score_threshold,float direction_offset)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;


    int *feature_map_size = (int*)malloc(2*sizeof(int));
    
    feature_map_size[0] = feature_map_height;
    feature_map_size[1] = feature_map_width;


    auto filterBoxByScoreCreater = getPluginRegistry()->getPluginCreator("FilterBoxByScorePlugin", "1");
    const PluginFieldCollection* filterBoxByScorepluginData = filterBoxByScoreCreater->getFieldNames();

    const PluginField* fields = filterBoxByScorepluginData->fields;
    int nbFields = filterBoxByScorepluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "feature_map_size"))
        {
          
            new_pluginData_list.emplace_back(PluginField("feature_map_size",  feature_map_size, PluginFieldType::kINT32, 1)); 
             
        }
         else if (!strcmp(attr_name, "direction_angle_num"))
        {
 
            new_pluginData_list.emplace_back(PluginField("direction_angle_num",  &(direction_angle_num), PluginFieldType::kINT32, 1));
            
        }
         else if (!strcmp(attr_name, "max_box_num"))
        {
 
            new_pluginData_list.emplace_back(PluginField("max_box_num",  &(max_box_num), PluginFieldType::kINT32, 1));
            
        }
        else if (!strcmp(attr_name, "score_threshold"))
        {
            new_pluginData_list.emplace_back(PluginField("score_threshold",  &(score_threshold), PluginFieldType::kFLOAT32, 1));  
     
        }
        else if (!strcmp(attr_name, "direction_offset"))
        {
        
            new_pluginData_list.emplace_back(PluginField("direction_offset",  &(direction_offset), PluginFieldType::kFLOAT32, 1));   
        }

    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_filterBoxByScore = filterBoxByScoreCreater->createPlugin("filterBoxByScorelayer", newPluginFieldCollection);
    ITensor* inputTensors_filterBoxByScore[] = {box_preds,cls_preds,iou_preds,dir_preds};
    auto filterBoxByScore = network->addPluginV2(inputTensors_filterBoxByScore, 4, *pluginObj_filterBoxByScore);
    pluginObj_filterBoxByScore->destroy();
    free(feature_map_size);
    return filterBoxByScore;
}


IPluginV2Layer* add_sparse2dense_layer(INetworkDefinition *network,ITensor * voxel_feature, ITensor* coords,ITensor* voxel_num,int feature_map_size_x,
                                        int feature_map_size_y, int feature_map_size_z,int feature_map_channel)
{

    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    int *feature_map_size = (int*)malloc(3*sizeof(int));
    feature_map_size[0] = feature_map_size_x;
    feature_map_size[1] = feature_map_size_y;
    feature_map_size[2] = feature_map_size_z;

    auto sparse2DenseCreator = getPluginRegistry()->getPluginCreator("Sparse2DensePlugin", "1");
    const PluginFieldCollection* sparse2DensepluginData = sparse2DenseCreator->getFieldNames();

    const PluginField* fields = sparse2DensepluginData->fields;
    int nbFields = sparse2DensepluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "feature_map_size"))
        {
            
            new_pluginData_list.emplace_back(PluginField("feature_map_size", feature_map_size, PluginFieldType::kINT32, 1));
            
        }
        else if (!strcmp(attr_name, "feature_map_channel"))
        {
          
            new_pluginData_list.emplace_back(PluginField("feature_map_channel",  &(feature_map_channel), PluginFieldType::kINT32, 1)); 
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_sparse2Dense = sparse2DenseCreator->createPlugin("sparse2denselayer", newPluginFieldCollection);
    ITensor* inputTensors_sparse2Dense[] = {voxel_feature,coords,voxel_num};
    auto sparse2dense = network->addPluginV2(inputTensors_sparse2Dense, 3, *pluginObj_sparse2Dense);
    pluginObj_sparse2Dense->destroy();
    free(feature_map_size);
    return sparse2dense;
}

IPluginV2Layer* add_zero_pad2d_layer(INetworkDefinition *network, ITensor * feature_map,int zero_padding_2d_top,
                                        int zero_padding_2d_bottom, int zero_padding_2d_left,int zero_padding_2d_right)
{

    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    int *zero_padding_2d_size = (int*)malloc(4*sizeof(int));
    zero_padding_2d_size[0] = zero_padding_2d_top;
    zero_padding_2d_size[1] = zero_padding_2d_bottom;
    zero_padding_2d_size[2] = zero_padding_2d_left;
    zero_padding_2d_size[3] = zero_padding_2d_right;

    auto zeroPad2dCreator = getPluginRegistry()->getPluginCreator("ZeroPad2dPlugin", "1");
    const PluginFieldCollection* zeroPad2dpluginData = zeroPad2dCreator->getFieldNames();

    const PluginField* fields = zeroPad2dpluginData->fields;
    int nbFields = zeroPad2dpluginData->nbFields;

    
    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "zero_padding_2d_size"))
        {
            new_pluginData_list.emplace_back(PluginField("zero_padding_2d_size", zero_padding_2d_size, PluginFieldType::kINT32, 1));
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_zeroPad2d = zeroPad2dCreator->createPlugin("zero_pad2dlayer", newPluginFieldCollection);
    ITensor* inputTensors_zeroPad2d[] = {feature_map,};
    auto zeroPad2d = network->addPluginV2(inputTensors_zeroPad2d, 1, *pluginObj_zeroPad2d);
    pluginObj_zeroPad2d->destroy();
    free(zero_padding_2d_size);
    return zeroPad2d;
}

IPluginV2Layer* add_element_wise_multiply_layer(INetworkDefinition *network, ITensor * feature_map,ITensor * weights, int index)
{

    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;


    auto elementWiseMultiplyCreator = getPluginRegistry()->getPluginCreator("ElementWiseMultiplyPlugin", "1");
    const PluginFieldCollection* elementWiseMultiplypluginData = elementWiseMultiplyCreator->getFieldNames();

    const PluginField* fields = elementWiseMultiplypluginData->fields;
    int nbFields = elementWiseMultiplypluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
    {
        const char* attr_name = fields[i].name;
        std::cout << attr_name << std::endl;
        if (!strcmp(attr_name, "index"))
        {
            new_pluginData_list.emplace_back(PluginField("index", &index, PluginFieldType::kINT32, 1));
            
        }
    }
    newPluginFieldCollection->fields = new_pluginData_list.data();
    newPluginFieldCollection->nbFields = new_pluginData_list.size();

    
    IPluginV2 *pluginObj_elementWiseMultiply = elementWiseMultiplyCreator->createPlugin("elementWiseMultiplylayer", newPluginFieldCollection);
    ITensor* inputTensors_elementWiseMultiply[] = {feature_map,weights};
    auto elementWiseMultiply = network->addPluginV2(inputTensors_elementWiseMultiply, 2, *pluginObj_elementWiseMultiply);
    pluginObj_elementWiseMultiply->destroy();
    return elementWiseMultiply;
}


IPluginV2Layer* add_subm_conv3d_layer(INetworkDefinition *network,ITensor* voxel_features,ITensor* coors,ITensor* voxel_num, nvinfer1::Weights const& weights,
                                    int max_voxels, int in_channel, int out_channel, int out_shape_x,
                                    int out_shape_y, int out_shape_z)
{
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
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

    auto submConv3dLayercreator = getPluginRegistry()->getPluginCreator("SubmConv3dLayerPlugin", "1");
    const PluginFieldCollection* submConv3dLayerpluginData = submConv3dLayercreator->getFieldNames();

    const PluginField* fields = submConv3dLayerpluginData->fields;
    int nbFields = submConv3dLayerpluginData->nbFields;
    
    for (int i = 0; i < nbFields; ++i)
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

   

    IPluginV2 *pluginObj_submConv3dLayer = submConv3dLayercreator->createPlugin("submConv3dLayer", newPluginFieldCollection);
    ITensor* inputTensors_submConv3dLayer[] = {voxel_features,coors,voxel_num};
    auto submConv3dLayer = network->addPluginV2(inputTensors_submConv3dLayer, 3, *pluginObj_submConv3dLayer);
    pluginObj_submConv3dLayer->destroy();
    free(out_shape);
    free(newPluginFieldCollection);
    return submConv3dLayer;
}

ILayer* add_batchNorm1d_relu(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;
   
    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    IShuffleLayer* shuffle1 = network->addShuffle(input);
    auto dim1 = input.getDimensions();
    dim1.d[0] = dim1.d[1];
    dim1.d[1] = dim1.d[2];
    dim1.d[2] = 1;
    dim1.d[3] = 1;
    dim1.nbDims = 4;
    shuffle1->setReshapeDimensions(dim1);
    assert(shuffle1);

    IScaleLayer* scale_1 = network->addScale(*shuffle1->getOutput(0), ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    IShuffleLayer* shuffle2 = network->addShuffle(*scale_1->getOutput(0));
    auto dim2 = scale_1->getOutput(0)->getDimensions();
    dim2.d[2] = dim2.d[1];
    dim2.d[1] = dim2.d[0];
    dim2.d[0] = 1;
    dim2.nbDims = 3;
    shuffle2->setReshapeDimensions(dim2);
    assert(shuffle2);
    auto lr = network->addActivation(*shuffle2->getOutput(0), ActivationType::kRELU);
    lr->setAlpha(1e-8);
    return lr;
}


IPluginV2Layer* add_sparse_conv3d_layer(INetworkDefinition *network,ITensor* voxel_features,ITensor* coors,ITensor* voxel_num, nvinfer1::Weights const& weights,
                                    int max_voxels, int in_channel, int out_channel, int out_shape_x,
                                    int out_shape_y, int out_shape_z,
                                    int spatial_shape_x,int spatial_shape_y, int spatial_shape_z,int ksize0,int ksize1, int ksize2,
                                    int stride0, int stride1, int stride2, int padding0, int padding1, int padding2)
{   
    // cout << "add_sparse_conv3d_layer:               " << weights.count << std::endl;
    PluginFieldCollection * newPluginFieldCollection = (PluginFieldCollection *)malloc(sizeof(PluginFieldCollection));
    newPluginFieldCollection->fields = nullptr;
    newPluginFieldCollection->nbFields = 0;
    std::vector<PluginField> new_pluginData_list;

    int *out_shape = (int*)malloc(3*sizeof(int));
    out_shape[0] = out_shape_x;
    out_shape[1] = out_shape_y;
    out_shape[2] = out_shape_z;

    int *spatial_shape = (int*)malloc(3*sizeof(int));
    spatial_shape[0] = spatial_shape_x;
    spatial_shape[1] = spatial_shape_y;
    spatial_shape[2] = spatial_shape_z;


    int dilation = 1;
    int out_padding = 0;

    // std::cout << "ksize0" << ksize0 << std::endl;

    int *ksize = (int*)malloc(3*sizeof(int));
    ksize[0] = ksize0;
    ksize[1] = ksize1;
    ksize[2] = ksize2;

    int *stride = (int*)malloc(3*sizeof(int));
    stride[0] = stride0;
    stride[1] = stride1;
    stride[2] = stride2;

    int *padding = (int*)malloc(3*sizeof(int));
    padding[0] = padding0;
    padding[1] = padding1;
    padding[2] = padding2;



    // std::cout << "max_voxels: " << max_voxels << "in_channel: " << in_channel << "out_channel: " << out_channel << "out_shape_x: " << out_shape_x << 
    //             "out_shape_y: " << out_shape_y << "out_shape_z: " << out_shape_z << "spatial_shape_x: " << spatial_shape_x << "spatial_shape_y: " << spatial_shape_y << 
    //             "spatial_shape_z" << spatial_shape_z << "ksize: " << ksize << "stride: " << stride << "padding " << padding << "dilation: " << dilation << 
    //             "out_padding: " << out_padding << std::endl;

    
    
    auto sparseConv3dLayercreator = getPluginRegistry()->getPluginCreator("SparseConv3dLayerPlugin", "1");
    const PluginFieldCollection* sparseConv3dLayerpluginData = sparseConv3dLayercreator->getFieldNames();
    const PluginField* fields = sparseConv3dLayerpluginData->fields;
    int nbFields = sparseConv3dLayerpluginData->nbFields;

    for (int i = 0; i < nbFields; ++i)
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
           
            new_pluginData_list.emplace_back(PluginField("spatial_shape",  spatial_shape, PluginFieldType::kINT32, 1));
          
        }
        else if (!strcmp(attr_name, "ksize"))
        {
            
            new_pluginData_list.emplace_back(PluginField("ksize",  ksize, PluginFieldType::kINT32, 1)); 
           
        }
        else if (!strcmp(attr_name, "stride"))
        {
           
            new_pluginData_list.emplace_back(PluginField("stride",  stride, PluginFieldType::kINT32, 1)); 
            
        }
        else if (!strcmp(attr_name, "padding"))
        {
            
            new_pluginData_list.emplace_back(PluginField("padding",  padding, PluginFieldType::kINT32, 1)); 
           
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


    IPluginV2 *pluginObj_sparseConv3dLayer = sparseConv3dLayercreator->createPlugin("sparseConv3dLayer", newPluginFieldCollection);
    ITensor* inputTensors_sparseConv3dLayer[] = {voxel_features,coors,voxel_num};
    auto sparseConv3dLayer = network->addPluginV2(inputTensors_sparseConv3dLayer, 3, *pluginObj_sparseConv3dLayer);
    pluginObj_sparseConv3dLayer->destroy();
    free(out_shape);
    free(spatial_shape);
    free(ksize);
    free(stride);
    free(padding);
    free(newPluginFieldCollection);
    return sparseConv3dLayer;
}


IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLELU(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, 
                    std::string conv2d_prefix, std::string batchnorm_prefix) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), batchnorm_prefix, 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    lr->setAlpha(0.1);

    return lr;
}

ILayer* convBn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, 
                    std::string conv2d_prefix, std::string batchnorm_prefix) 
{
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), batchnorm_prefix, 1e-5);


    return bn1;
}

ILayer* deconvBnLELU(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int output_padding,
                    std::string conv2d_prefix, std::string batchnorm_prefix) 
{
    
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IDeconvolutionLayer* conv1 = network->addDeconvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap[conv2d_prefix + ".weight"], emptywts);
    assert(conv1);
    // conv1->setStrideNd(DimsHW{s, s});
    conv1->setStride(DimsHW{s, s});
    // conv1->setPaddingNd(DimsHW{p, p});
    conv1->setDilationNd(DimsHW{1,1});
    conv1->setPrePadding(DimsHW{p, p}); // pytorch padding
    // conv1->setPostPadding(DimsHW(output_padding+1,output_padding+1)); // pytorch output_padding
  
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), batchnorm_prefix, 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    lr->setAlpha(0.1);

    return lr;
}


ILayer* add_bottom_up_block_conv(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, 
                          int conv_0_in_channel,  int conv_0_out_channel, int conv_0_ksize, int conv_0_stride, int conv_0_padding,
                          std::string conv_0_prefix, int batchnorm2d_0_num_features, std::string batchnorm2d_0_prefix,
                          int conv_1_in_channel,  int conv_1_out_channel, int conv_1_ksize, int conv_1_stride, int conv_1_padding,
                          std::string conv_1_prefix, int batchnorm2d_1_num_features, std::string batchnorm2d_1_prefix,
                          int conv_2_in_channel,  int conv_2_out_channel, int conv_2_ksize, int conv_2_stride, int conv_2_padding,
                          std::string conv_2_prefix, int batchnorm2d_2_num_features, std::string batchnorm2d_2_prefix
                          )
{
    auto bottom_up_conv_0_bn2d_relu = convBnLELU(network,weightMap,input,conv_0_out_channel,conv_0_ksize,conv_0_stride,conv_0_padding,
                        conv_0_prefix,batchnorm2d_0_prefix);
    auto bottom_up_conv_1_bn2d_relu = convBnLELU(network,weightMap,*bottom_up_conv_0_bn2d_relu->getOutput(0),conv_1_out_channel,conv_1_ksize,conv_1_stride,conv_1_padding,
                        conv_1_prefix,batchnorm2d_1_prefix);
    auto bottom_up_conv_2_bn2d_relu = convBnLELU(network,weightMap,*bottom_up_conv_1_bn2d_relu->getOutput(0),conv_2_out_channel,conv_2_ksize,conv_2_stride,conv_2_padding,
                        conv_2_prefix,batchnorm2d_2_prefix);
    return bottom_up_conv_2_bn2d_relu;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config) {
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);

    ITensor* point_data = network->addInput(INPUT_POINTS, DataType::kFLOAT, Dims3{1, MAX_POINTS,4});
    Dims dims1;
    dims1.d[0] = 1;
    dims1.nbDims = 1;
    ITensor* point_size = network->addInput(INPUT_POINTS_SIZE,DataType::kINT32,dims1);
    assert(point_data);
    assert(point_size);
    // return;

    std::map<std::string, Weights> weightMap = loadWeights("../se-ssd.wts");
    std::cout << "load weights finished" << std::endl;
   

    auto voxelGenerator = add_voxel_generator(network,point_data,point_size,MAX_NUM_POINTS_PER_VOXEL,MAX_VOXELS,VOXEL_SIZE_X,VOXEL_SIZE_Y,VOXEL_SIZE_Z,
                                        X_MIN,X_MAX,Y_MIN,Y_MAX,Z_MIN,Z_MAX,VOXEL_FEATURE_NUM);
    

    /*
            backbone <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    */
  

    auto submConv3dLayer0 = add_subm_conv3d_layer(network,voxelGenerator->getOutput(0),voxelGenerator->getOutput(1),voxelGenerator->getOutput(2),
                                    weightMap["backbone.middle_conv.0.weight"],
                                SUBM_0_MAX_VOXELS,SUBM_0_IN_CHANNEL,SUBM_0_OUT_CHANNEL,SUBM_0_OUT_SHAPE_X,SUBM_0_OUT_SHAPE_Y,SUBM_0_OUT_SHAPE_Z);
    
    auto batch1d_1_2 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer0->getOutput(0), "backbone.middle_conv.1", 1e-5);


    auto submConv3dLayer3 = add_subm_conv3d_layer(network,batch1d_1_2->getOutput(0),voxelGenerator->getOutput(1),voxelGenerator->getOutput(2),
                                    weightMap["backbone.middle_conv.3.weight"],
                                SUBM_3_MAX_VOXELS,SUBM_3_IN_CHANNEL,SUBM_3_OUT_CHANNEL,SUBM_3_OUT_SHAPE_X,SUBM_3_OUT_SHAPE_Y,SUBM_3_OUT_SHAPE_Z);

    auto batch1d_4_5 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer3->getOutput(0), "backbone.middle_conv.4", 1e-5);

   
    auto sparseConv3dLayer6 = add_sparse_conv3d_layer(network,batch1d_4_5->getOutput(0),voxelGenerator->getOutput(1),voxelGenerator->getOutput(2),
                                    weightMap["backbone.middle_conv.6.weight"],
                                SPARSE_6_MAX_VOXELS,SPARSE_6_IN_CHANNEL,SPARSE_6_OUT_CHANNEL,
                                SPARSE_6_OUT_SHAPE_X,SPARSE_6_OUT_SHAPE_Y,SPARSE_6_OUT_SHAPE_Z,
                                SPARSE_6_SPATIAL_SHAPE_X,SPARSE_6_SPATIAL_SHAPE_Y,SPARSE_6_SPATIAL_SHAPE_Z,
                                SPARSE_6_KSIZE_0,SPARSE_6_KSIZE_1,SPARSE_6_KSIZE_2,
                                SPARSE_6_STRIDE_0,SPARSE_6_STRIDE_1,SPARSE_6_STRIDE_2,
                                SPARSE_6_PADDING_0,SPARSE_6_PADDING_1,SPARSE_6_PADDING_2);

    auto batch1d_7_8 = add_batchNorm1d_relu(network, weightMap, *sparseConv3dLayer6->getOutput(0), "backbone.middle_conv.7", 1e-5);


    auto submConv3dLayer9 = add_subm_conv3d_layer(network,batch1d_7_8->getOutput(0),sparseConv3dLayer6->getOutput(1),sparseConv3dLayer6->getOutput(2),
                                    weightMap["backbone.middle_conv.9.weight"],
                                SUBM_9_MAX_VOXELS,SUBM_9_IN_CHANNEL,SUBM_9_OUT_CHANNEL,SUBM_9_OUT_SHAPE_X,SUBM_9_OUT_SHAPE_Y,SUBM_9_OUT_SHAPE_Z);

    auto batch1d_10_11 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer9->getOutput(0), "backbone.middle_conv.10", 1e-5);

    auto submConv3dLayer12 = add_subm_conv3d_layer(network,batch1d_10_11->getOutput(0),sparseConv3dLayer6->getOutput(1),sparseConv3dLayer6->getOutput(2),
                                    weightMap["backbone.middle_conv.12.weight"],
                                SUBM_12_MAX_VOXELS,SUBM_12_IN_CHANNEL,SUBM_12_OUT_CHANNEL,SUBM_12_OUT_SHAPE_X,SUBM_12_OUT_SHAPE_Y,SUBM_12_OUT_SHAPE_Z);

    auto batch1d_13_14 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer12->getOutput(0), "backbone.middle_conv.13", 1e-5);
    
    auto sparseConv3dLayer15 = add_sparse_conv3d_layer(network,batch1d_13_14->getOutput(0),sparseConv3dLayer6->getOutput(1),sparseConv3dLayer6->getOutput(2),
                                    weightMap["backbone.middle_conv.15.weight"],
                                SPARSE_15_MAX_VOXELS,SPARSE_15_IN_CHANNEL,SPARSE_15_OUT_CHANNEL,
                                SPARSE_15_OUT_SHAPE_X,SPARSE_15_OUT_SHAPE_Y,SPARSE_15_OUT_SHAPE_Z,
                                SPARSE_15_SPATIAL_SHAPE_X,SPARSE_15_SPATIAL_SHAPE_Y,SPARSE_15_SPATIAL_SHAPE_Z,
                                SPARSE_15_KSIZE_0,SPARSE_15_KSIZE_1,SPARSE_15_KSIZE_2,
                                SPARSE_15_STRIDE_0,SPARSE_15_STRIDE_1,SPARSE_15_STRIDE_2,
                                SPARSE_15_PADDING_0,SPARSE_15_PADDING_1,SPARSE_15_PADDING_2);

    auto batch1d_16_17 = add_batchNorm1d_relu(network, weightMap, *sparseConv3dLayer15->getOutput(0), "backbone.middle_conv.16", 1e-5);

    auto submConv3dLayer18 = add_subm_conv3d_layer(network,batch1d_16_17->getOutput(0),sparseConv3dLayer15->getOutput(1),sparseConv3dLayer15->getOutput(2),
                                    weightMap["backbone.middle_conv.18.weight"],
                                SUBM_18_MAX_VOXELS,SUBM_18_IN_CHANNEL,SUBM_18_OUT_CHANNEL,SUBM_18_OUT_SHAPE_X,SUBM_18_OUT_SHAPE_Y,SUBM_18_OUT_SHAPE_Z);

    auto batch1d_19_20 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer18->getOutput(0), "backbone.middle_conv.19", 1e-5);

    auto submConv3dLayer21 = add_subm_conv3d_layer(network,batch1d_19_20->getOutput(0),sparseConv3dLayer15->getOutput(1),sparseConv3dLayer15->getOutput(2),
                                    weightMap["backbone.middle_conv.21.weight"],
                                SUBM_21_MAX_VOXELS,SUBM_21_IN_CHANNEL,SUBM_21_OUT_CHANNEL,SUBM_21_OUT_SHAPE_X,SUBM_21_OUT_SHAPE_Y,SUBM_21_OUT_SHAPE_Z);

    auto batch1d_22_23 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer21->getOutput(0), "backbone.middle_conv.22", 1e-5);

    auto submConv3dLayer24 = add_subm_conv3d_layer(network,batch1d_22_23->getOutput(0),sparseConv3dLayer15->getOutput(1),sparseConv3dLayer15->getOutput(2),
                                    weightMap["backbone.middle_conv.24.weight"],
                                SUBM_24_MAX_VOXELS,SUBM_24_IN_CHANNEL,SUBM_24_OUT_CHANNEL,SUBM_24_OUT_SHAPE_X,SUBM_24_OUT_SHAPE_Y,SUBM_24_OUT_SHAPE_Z);

    auto batch1d_25_26 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer24->getOutput(0), "backbone.middle_conv.25", 1e-5);

    auto sparseConv3dLayer27 = add_sparse_conv3d_layer(network,batch1d_25_26->getOutput(0),sparseConv3dLayer15->getOutput(1),sparseConv3dLayer15->getOutput(2),
                                    weightMap["backbone.middle_conv.27.weight"],
                                SPARSE_27_MAX_VOXELS,SPARSE_27_IN_CHANNEL,SPARSE_27_OUT_CHANNEL,
                                SPARSE_27_OUT_SHAPE_X,SPARSE_27_OUT_SHAPE_Y,SPARSE_27_OUT_SHAPE_Z,
                                SPARSE_27_SPATIAL_SHAPE_X,SPARSE_27_SPATIAL_SHAPE_Y,SPARSE_27_SPATIAL_SHAPE_Z,
                                SPARSE_27_KSIZE_0,SPARSE_27_KSIZE_1,SPARSE_27_KSIZE_2,
                                SPARSE_27_STRIDE_0,SPARSE_27_STRIDE_1,SPARSE_27_STRIDE_2,
                                SPARSE_27_PADDING_0,SPARSE_27_PADDING_1,SPARSE_27_PADDING_2);

    auto batch1d_28_29 = add_batchNorm1d_relu(network, weightMap, *sparseConv3dLayer27->getOutput(0), "backbone.middle_conv.28", 1e-5);


    auto submConv3dLayer30 = add_subm_conv3d_layer(network,batch1d_28_29->getOutput(0),sparseConv3dLayer27->getOutput(1),sparseConv3dLayer27->getOutput(2),
                                    weightMap["backbone.middle_conv.30.weight"],
                                SUBM_30_MAX_VOXELS,SUBM_30_IN_CHANNEL,SUBM_30_OUT_CHANNEL,SUBM_30_OUT_SHAPE_X,SUBM_30_OUT_SHAPE_Y,SUBM_30_OUT_SHAPE_Z);
    auto batch1d_31_32 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer30->getOutput(0), "backbone.middle_conv.31", 1e-5);

    auto submConv3dLayer33 = add_subm_conv3d_layer(network,batch1d_31_32->getOutput(0),sparseConv3dLayer27->getOutput(1),sparseConv3dLayer27->getOutput(2),
                                    weightMap["backbone.middle_conv.33.weight"],
                                SUBM_33_MAX_VOXELS,SUBM_33_IN_CHANNEL,SUBM_33_OUT_CHANNEL,SUBM_33_OUT_SHAPE_X,SUBM_33_OUT_SHAPE_Y,SUBM_33_OUT_SHAPE_Z);
    auto batch1d_34_35 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer33->getOutput(0), "backbone.middle_conv.34", 1e-5);

    auto submConv3dLayer36 = add_subm_conv3d_layer(network,batch1d_34_35->getOutput(0),sparseConv3dLayer27->getOutput(1),sparseConv3dLayer27->getOutput(2),
                                    weightMap["backbone.middle_conv.36.weight"],
                                SUBM_36_MAX_VOXELS,SUBM_36_IN_CHANNEL,SUBM_36_OUT_CHANNEL,SUBM_36_OUT_SHAPE_X,SUBM_36_OUT_SHAPE_Y,SUBM_36_OUT_SHAPE_Z);
    auto batch1d_37_38 = add_batchNorm1d_relu(network, weightMap, *submConv3dLayer36->getOutput(0), "backbone.middle_conv.37", 1e-5);

    auto sparseConv3dLayer39 = add_sparse_conv3d_layer(network,batch1d_37_38->getOutput(0),sparseConv3dLayer27->getOutput(1),sparseConv3dLayer27->getOutput(2),
                                    weightMap["backbone.middle_conv.39.weight"],
                                SPARSE_39_MAX_VOXELS,SPARSE_39_IN_CHANNEL,SPARSE_39_OUT_CHANNEL,
                                SPARSE_39_OUT_SHAPE_X,SPARSE_39_OUT_SHAPE_Y,SPARSE_39_OUT_SHAPE_Z,
                                SPARSE_39_SPATIAL_SHAPE_X,SPARSE_39_SPATIAL_SHAPE_Y,SPARSE_39_SPATIAL_SHAPE_Z,
                                SPARSE_39_KSIZE_0,SPARSE_39_KSIZE_1,SPARSE_39_KSIZE_2,
                                SPARSE_39_STRIDE_0,SPARSE_39_STRIDE_1,SPARSE_39_STRIDE_2,
                                SPARSE_39_PADDING_0,SPARSE_39_PADDING_1,SPARSE_39_PADDING_2);
    auto batch1d_40_41 = add_batchNorm1d_relu(network, weightMap, *sparseConv3dLayer39->getOutput(0), "backbone.middle_conv.40", 1e-5);

    auto sparse2denseLayer = add_sparse2dense_layer(network,batch1d_40_41->getOutput(0),sparseConv3dLayer39->getOutput(1),sparseConv3dLayer39->getOutput(2),
                            BACKBONE_FEATURE_MAP_SIZE_X,BACKBONE_FEATURE_MAP_SIZE_Y,BACKBONE_FEATURE_MAP_SIZE_Z,BACKBONE_FEATURE_MAP_CHANNEL);
    auto zeroPad2dLayer = add_zero_pad2d_layer(network,sparse2denseLayer->getOutput(0),ZERO_PADDING_2D_TOP,ZERO_PADDING_2D_BOTTOM,ZERO_PADDING_2D_LEFT,ZERO_PADDING_2D_RIGHT);




     /*
            neck <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    */
  
    auto bottom_up_block_0_conv = add_bottom_up_block_conv(network,weightMap,*zeroPad2dLayer->getOutput(0),BOTTOM_UP_BLOCK_0_CONV_0_IN_CHANNEL,
                                       BOTTOM_UP_BLOCK_0_CONV_0_OUT_CHANNEL,BOTTOM_UP_BLOCK_0_CONV_0_KSIZE,BOTTOM_UP_BLOCK_0_CONV_0_STRIDE,
                                       BOTTOM_UP_BLOCK_0_CONV_0_PADDING,"neck.bottom_up_block_0.1",BOTTOM_UP_BLOCK_0_BATCHNORM2D_0_NUM_FEATURES,
                                       "neck.bottom_up_block_0.2",BOTTOM_UP_BLOCK_0_CONV_1_IN_CHANNEL,BOTTOM_UP_BLOCK_0_CONV_1_OUT_CHANNEL,
                                        BOTTOM_UP_BLOCK_0_CONV_1_KSIZE,BOTTOM_UP_BLOCK_0_CONV_1_STRIDE,BOTTOM_UP_BLOCK_0_CONV_1_PADDING,
                                        "neck.bottom_up_block_0.4",BOTTOM_UP_BLOCK_0_BATCHNORM2D_1_NUM_FEATURES,"neck.bottom_up_block_0.5",
                                        BOTTOM_UP_BLOCK_0_CONV_2_IN_CHANNEL,BOTTOM_UP_BLOCK_0_CONV_2_OUT_CHANNEL,BOTTOM_UP_BLOCK_0_CONV_2_KSIZE,
                                        BOTTOM_UP_BLOCK_0_CONV_2_STRIDE,BOTTOM_UP_BLOCK_0_CONV_2_PADDING,"neck.bottom_up_block_0.7",
                                        BOTTOM_UP_BLOCK_0_BATCHNORM2D_2_NUM_FEATURES,"neck.bottom_up_block_0.8"
                                       );
    auto bottom_up_block_1_conv = add_bottom_up_block_conv(network,weightMap,*bottom_up_block_0_conv->getOutput(0),BOTTOM_UP_BLOCK_1_CONV_0_IN_CHANNEL,
                                    BOTTOM_UP_BLOCK_1_CONV_0_OUT_CHANNEL,BOTTOM_UP_BLOCK_1_CONV_0_KSIZE,BOTTOM_UP_BLOCK_1_CONV_0_STRIDE,
                                    BOTTOM_UP_BLOCK_1_CONV_0_PADDING,"neck.bottom_up_block_1.0",BOTTOM_UP_BLOCK_1_BATCHNORM2D_0_NUM_FEATURES,
                                    "neck.bottom_up_block_1.1",BOTTOM_UP_BLOCK_1_CONV_1_IN_CHANNEL,BOTTOM_UP_BLOCK_1_CONV_1_OUT_CHANNEL,
                                    BOTTOM_UP_BLOCK_1_CONV_1_KSIZE,BOTTOM_UP_BLOCK_1_CONV_1_STRIDE,BOTTOM_UP_BLOCK_1_CONV_1_PADDING,
                                    "neck.bottom_up_block_1.3",BOTTOM_UP_BLOCK_1_BATCHNORM2D_1_NUM_FEATURES,"neck.bottom_up_block_1.4",
                                    BOTTOM_UP_BLOCK_1_CONV_2_IN_CHANNEL,BOTTOM_UP_BLOCK_1_CONV_2_OUT_CHANNEL,BOTTOM_UP_BLOCK_1_CONV_2_KSIZE,
                                    BOTTOM_UP_BLOCK_1_CONV_2_STRIDE,BOTTOM_UP_BLOCK_1_CONV_2_PADDING,"neck.bottom_up_block_1.6",
                                    BOTTOM_UP_BLOCK_1_BATCHNORM2D_2_NUM_FEATURES,"neck.bottom_up_block_1.7"
                                    );
    
    auto trans_0 = convBnLELU(network,weightMap,*bottom_up_block_0_conv->getOutput(0),TRANS_0_CONV_OUT_CHANNEL,TRANS_0_CONV_KSIZE,TRANS_0_CONV_STRIDE,
                            TRANS_0_CONV_PADDING,"neck.trans_0.0","neck.trans_0.1");
    
    auto trans_1 = convBnLELU(network,weightMap,*bottom_up_block_1_conv->getOutput(0),TRANS_1_CONV_OUT_CHANNEL,TRANS_1_CONV_KSIZE,TRANS_1_CONV_STRIDE,
                            TRANS_1_CONV_PADDING,"neck.trans_1.0","neck.trans_1.1");

    auto deconv_block_0 = deconvBnLELU(network,weightMap,*trans_1->getOutput(0),DECONV_BLOCK_0_CONV_OUT_CHANNEL,DECONV_BLOCK_0_CONV_KSIZE,DECONV_BLOCK_0_CONV_STRIDE,
                                        DECONV_BLOCK_0_CONV_PADDING,DECONV_BLOCK_0_CONV_OUT_PADDING,"neck.deconv_block_0.0","neck.deconv_block_0.1");

    auto elementwise = network->addElementWise(*deconv_block_0->getOutput(0),*trans_0->getOutput(0),ElementWiseOperation::kSUM);
    
                
    auto deconv_block_1 = deconvBnLELU(network,weightMap,*trans_1->getOutput(0),DECONV_BLOCK_1_CONV_OUT_CHANNEL,DECONV_BLOCK_1_CONV_KSIZE,DECONV_BLOCK_1_CONV_STRIDE,
                                        DECONV_BLOCK_1_CONV_PADDING,DECONV_BLOCK_1_CONV_OUT_PADDING,"neck.deconv_block_1.0","neck.deconv_block_1.1");
    
    auto conv_0 = convBnLELU(network,weightMap,*elementwise->getOutput(0),CONV_0_CONV_OUT_CHANNEL,CONV_0_CONV_KSIZE,CONV_0_CONV_STRIDE,CONV_0_CONV_PADDING,
                        "neck.conv_0.0","neck.conv_0.1");
    auto conv_1 = convBnLELU(network,weightMap,*deconv_block_1->getOutput(0),CONV_1_CONV_OUT_CHANNEL,CONV_1_CONV_KSIZE,CONV_1_CONV_STRIDE,CONV_1_CONV_PADDING,
                        "neck.conv_1.0","neck.conv_1.1");

    auto w_0 = convBn(network,weightMap,*conv_0->getOutput(0),W_0_CONV_OUT_CHANNEL,W_0_CONV_KSIZE,W_0_CONV_STRIDE,W_0_CONV_PADDING,
                        "neck.w_0.0","neck.w_0.1");
    auto w_1 = convBn(network,weightMap,*conv_1->getOutput(0),W_1_CONV_OUT_CHANNEL,W_1_CONV_KSIZE,W_1_CONV_STRIDE,W_1_CONV_PADDING,
                        "neck.w_1.0","neck.w_1.1");

    ITensor* inputTensors[] = {w_0->getOutput(0), w_1->getOutput(0)};
    auto cat_tensor = network->addConcatenation(inputTensors, 2);
    cat_tensor->setAxis(1);

    auto softmax_weight = network->addSoftMax(*cat_tensor->getOutput(0));
    softmax_weight->setAxes(1<<1);


    nvinfer1::Dims start_0{ 4, 0, 0, 0, 0 };
    nvinfer1::Dims size_0{ 4, 1, 1, 200, 176};  
    nvinfer1::Dims stride_0{ 4, 1, 1, 1, 1 };

    auto slice_0 = network->addSlice(*softmax_weight->getOutput(0),start_0,size_0,stride_0);

    nvinfer1::Dims start_1{ 4, 0, 1, 0, 0};
    nvinfer1::Dims size_1{ 4, 1, 1, 200, 176};  
    nvinfer1::Dims stride_1{ 4, 1, 1, 1, 1 };

    auto slice_1 = network->addSlice(*softmax_weight->getOutput(0),start_1,size_1,stride_1);

    auto weight_element_wise_multiply_0 = network->addElementWise(*conv_0->getOutput(0),*slice_0->getOutput(0),ElementWiseOperation::kPROD);

    auto weight_element_wise_multiply_1 = network->addElementWise(*conv_1->getOutput(0),*slice_1->getOutput(0),ElementWiseOperation::kPROD);
    
    auto elementwise_sum_1 = network->addElementWise(*weight_element_wise_multiply_0->getOutput(0),*weight_element_wise_multiply_1->getOutput(0),
                                ElementWiseOperation::kSUM);
    
    
    /*
    head <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    */


    auto conv_box = network->addConvolutionNd(*elementwise_sum_1->getOutput(0), CONV_BOX_OUT_CHANNEL,
                 DimsHW{CONV_BOX_KSIZE, CONV_BOX_KSIZE}, weightMap["bbox_head.tasks.0.conv_box.weight"], weightMap["bbox_head.tasks.0.conv_box.bias"]);
    assert(conv_box);
    conv_box->setStrideNd(DimsHW{CONV_BOX_STRIDE, CONV_BOX_STRIDE});
    conv_box->setPaddingNd(DimsHW{CONV_BOX_PADDING, CONV_BOX_PADDING});
    auto conv_box_sfl = network->addShuffle(*conv_box->getOutput(0));
    conv_box_sfl->setFirstTranspose(Permutation{0,2,3,1});


    auto conv_cls = network->addConvolutionNd(*elementwise_sum_1->getOutput(0), CONV_CLS_OUT_CHANNEL,
                 DimsHW{CONV_CLS_KSIZE, CONV_CLS_KSIZE}, weightMap["bbox_head.tasks.0.conv_cls.weight"], weightMap["bbox_head.tasks.0.conv_cls.bias"]);
    assert(conv_cls);
    conv_cls->setStrideNd(DimsHW{CONV_CLS_STRIDE, CONV_CLS_STRIDE});
    conv_cls->setPaddingNd(DimsHW{CONV_CLS_PADDING, CONV_CLS_PADDING});
    auto conv_cls_sfl = network->addShuffle(*conv_cls->getOutput(0));
    conv_cls_sfl->setFirstTranspose(Permutation{0,2,3,1});


    auto conv_iou = network->addConvolutionNd(*elementwise_sum_1->getOutput(0), CONV_IOU_OUT_CHANNEL,
                 DimsHW{CONV_IOU_KSIZE, CONV_IOU_KSIZE}, weightMap["bbox_head.tasks.0.conv_iou.weight"], weightMap["bbox_head.tasks.0.conv_iou.bias"]);
    assert(conv_iou);
    conv_iou->setStrideNd(DimsHW{CONV_IOU_STRIDE, CONV_IOU_STRIDE});
    conv_iou->setPaddingNd(DimsHW{CONV_IOU_PADDING, CONV_IOU_PADDING});
    auto conv_iou_sfl = network->addShuffle(*conv_iou->getOutput(0));
    conv_iou_sfl->setFirstTranspose(Permutation{0,2,3,1});


    auto conv_dir = network->addConvolutionNd(*elementwise_sum_1->getOutput(0), CONV_DIR_OUT_CHANNEL,
                 DimsHW{CONV_DIR_KSIZE, CONV_DIR_KSIZE}, weightMap["bbox_head.tasks.0.conv_dir.weight"], weightMap["bbox_head.tasks.0.conv_dir.bias"]);
    assert(conv_dir);
    conv_dir->setStrideNd(DimsHW{CONV_DIR_STRIDE, CONV_DIR_STRIDE});
    conv_dir->setPaddingNd(DimsHW{CONV_DIR_PADDING, CONV_DIR_PADDING});
    auto conv_dir_sfl = network->addShuffle(*conv_dir->getOutput(0));
    conv_dir_sfl->setFirstTranspose(Permutation{0,2,3,1});


      /*
            postprocess   generate_anchor  decode   nms <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    */
    // reshape

    auto conv_box_preds = network->addShuffle(*conv_box_sfl->getOutput(0));
    auto box_preds_dim = conv_box_sfl->getOutput(0)->getDimensions();
    // int channel = dim1.d[2];
    box_preds_dim.d[1] = box_preds_dim.d[1]*box_preds_dim.d[2]*2;
    box_preds_dim.d[2] = CONV_BOX_DIM_NUM;
    box_preds_dim.nbDims = 3;
    conv_box_preds->setReshapeDimensions(box_preds_dim);
    assert(conv_box_preds);  // 1 70400 7


    auto conv_cls_preds = network->addShuffle(*conv_cls_sfl->getOutput(0));
    auto cls_preds_dim = conv_cls_sfl->getOutput(0)->getDimensions();
    // int channel = dim1.d[2];
    cls_preds_dim.d[1] = cls_preds_dim.d[1]*cls_preds_dim.d[2]*2;
    cls_preds_dim.d[2] = CONV_CLS_DIM_NUM;
    cls_preds_dim.nbDims = 3;
    conv_cls_preds->setReshapeDimensions(cls_preds_dim);
    assert(conv_cls_preds);  // 1 70400 1


    auto conv_iou_preds = network->addShuffle(*conv_iou_sfl->getOutput(0));
    auto iou_preds_dim = conv_iou_sfl->getOutput(0)->getDimensions();
    // int channel = dim1.d[2];
    iou_preds_dim.d[1] = iou_preds_dim.d[1]*iou_preds_dim.d[2]*2;
    iou_preds_dim.d[2] = CONV_IOU_DIM_NUM;
    iou_preds_dim.nbDims = 3;
    conv_iou_preds->setReshapeDimensions(iou_preds_dim);
    assert(conv_iou_preds);  // 1 70400 1


    auto conv_dir_preds = network->addShuffle(*conv_dir_sfl->getOutput(0));
    auto dir_preds_dim = conv_dir_sfl->getOutput(0)->getDimensions();
    // int channel = dim1.d[2];
    dir_preds_dim.d[1] = dir_preds_dim.d[1]*dir_preds_dim.d[2]*2;
    dir_preds_dim.d[2] = CONV_DIR_DIM_NUM;
    dir_preds_dim.nbDims = 3;
    conv_dir_preds->setReshapeDimensions(dir_preds_dim);
    assert(conv_dir_preds);  // 1 70400 2

    auto anchor_decoder = add_generate_anchor_decoder(network,conv_box_preds->getOutput(0),X_MIN,X_MAX,
                Y_MIN,Y_MAX,Z_MIN,Z_MAX,FEATURE_MAP_HEIGHT,FEATURE_MAP_WIDTH,CAR_LENGTH,CAR_WIDTH,CAR_HEIGHT,
                DIRECTION_ANGLE_0,DIRECTION_ANGLE_1,DIRECTION_ANGLE_NUM);
    /*
    //postprocess  filter box by score   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    */

   auto filter_box_by_score_layer = add_filter_box_by_score_layer(network,anchor_decoder->getOutput(0),conv_cls_preds->getOutput(0),
                                        conv_iou_preds->getOutput(0),conv_dir_preds->getOutput(0),FEATURE_MAP_HEIGHT,
                                        FEATURE_MAP_WIDTH,DIRECTION_ANGLE_NUM,MAX_BOX_NUM,SCORE_THRESHOLD,DIRECTION_OFFSET);






//    auto topk = network->addTopK(*filter_box_by_score_layer->getOutput(1),TopKOperation::kMAX,88,0x02);
// network->addTopK
//     std::cout << "create plugin finished" << std::endl;
    
    auto dim = filter_box_by_score_layer->getOutput(0)->getDimensions();
    std::cout << "output0 output shape: ";
    for (int i = 0; i < dim.nbDims; i++) {
        std::cout << dim.d[i] << " ";
    }
    std::cout << std::endl;

    auto dim1 = filter_box_by_score_layer->getOutput(1)->getDimensions();
    std::cout << "output1 output shape: ";
    for (int i = 0; i < dim1.nbDims; i++) {
        std::cout << dim1.d[i] << " ";
    }
    std::cout << std::endl;

    auto dim2 = filter_box_by_score_layer->getOutput(2)->getDimensions();
    std::cout << "output2 output shape: ";
    for (int i = 0; i < dim2.nbDims; i++) {
        std::cout << dim2.d[i] << " ";
    }
    std::cout << std::endl;

    filter_box_by_score_layer->getOutput(0)->setName(OUTPUT_VOXELS);
    network->markOutput(*filter_box_by_score_layer->getOutput(0));

    // sparseConv3dLayer39->getOutput(1)->setName(OUTPUT_COORS);
    // network->markOutput(*sparseConv3dLayer39->getOutput(1));

    filter_box_by_score_layer->getOutput(4)->setName(OUTPUT_VOXEL_NUM); 
    network->markOutput(*filter_box_by_score_layer->getOutput(4));

    // Build engine
    config->setMaxWorkspaceSize(1600 * (1 << 20));  // 16MB
#ifdef USE_FP16
    config->setFlag(BuilderFlag::kFP16);
#endif
    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();
    // pluginObj_voxelGenerator->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    // free params struct;
    // free(newSubmConv3dLayerpluginData);
    

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(rt_glogger); // rt_glogger
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config); // DataType::kFLOAT
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

int loadData(const char *file, void **data, unsigned int *length)
{
  std::fstream dataFile(file, std::ifstream::in);

  if (!dataFile.is_open())
  {
	  std::cout << "Can't open files: "<< file<<std::endl;
	  return -1;
  }

  //get length of file:
  unsigned int len = 0;
  dataFile.seekg (0, dataFile.end);
  len = dataFile.tellg();
  dataFile.seekg (0, dataFile.beg);

  //allocate memory:
  char *buffer = new char[MAX_POINTS*4*4];
  if(len>MAX_POINTS*4*4)
  {
      std::cout << "num of points: " << len << ">" << MAX_POINTS*4*4 << std::endl;
      delete [] buffer;
      dataFile.close();
      exit(-1);
  }
  // init for buffer
  for(int i=0; i<MAX_POINTS; i++)
  {
      buffer[i] = 0;
  }
  
  if(buffer==NULL) {
	  std::cout << "Can't malloc buffer."<<std::endl;
    dataFile.close();
	  exit(-1);
  }

  //read data as a block:
  dataFile.read(buffer, len);
  dataFile.close();

  *data = (void*)buffer;
  *length = len;
  return 0;  
}

// string split(使用字符串分割)
void stringsplit(const std::string& str, const std::string& splits, std::vector<std::string>& res)
{
    if(str == "") return;
    string strs = str + splits;
    int pos = strs.find(splits);
    int step = splits.size();

    while(pos != strs.npos)
    {
        std::string temp = strs.substr(0,pos);
        res.push_back(temp);
        strs = strs.substr(pos+step,strs.size());
        pos = strs.find(splits);
    }
}

int save_txt(float* voxel_feature,unsigned int* coors,int voxel_num,std::string save_path,float seconds,int output_channel)
{
    std::ofstream out_txt_file;
    out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // out_txt_file << fixed;
    out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    out_txt_file << seconds << std::endl;

    for(int i=0;i < voxel_num; i++)
    {
        for(int j=0;j<output_channel;j++)
        {
            out_txt_file << *(voxel_feature + i * output_channel+j) << ",";
        }
        out_txt_file << *(coors + i * 4) << "," << *(coors + i * 4+1) << "," 
            << *(coors + i * 4 + 2) << "," << *(coors+i*4+3) << std::endl;;

        // std::cout << "index : " << i << "   " << "voxel: " << *(voxel_feature + i * 4) << ","  << *(voxel_feature + i * 4+1) << ","
        //     << *(voxel_feature + i * 4+2) << "," << *(voxel_feature + i * 4+3) << ",||||"<< *(coors + i * 4) << "," << *(coors + i * 4+1) << "," 
        //     << *(coors + i * 4 + 2) << "," << *(coors+i*4+3)
        //     << std::endl;   
    }
    out_txt_file.close();
    return 0;
}

int save_txt2(float* voxel_feature,std::string save_path,float seconds,int output_channel,int height, int width)
{
    std::ofstream out_txt_file;
    out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // out_txt_file << fixed;
    out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    out_txt_file << seconds << std::endl;
    float max_value = 0;
    for(int channel_i=0;channel_i < output_channel; channel_i++)
    {
        for(int height_i=0;height_i < height; height_i++)
        {
            for(int width_i=0;width_i<width;width_i++)
            {
                out_txt_file << *(voxel_feature + channel_i*height*width+height_i*width+width_i) << ",";
                if (*(voxel_feature + channel_i*height*width+height_i*width+width_i)>max_value)
                {
                    max_value = *(voxel_feature + channel_i*height*width+height_i*width+width_i);
                }
            }
            
        }
        out_txt_file << std::endl;
    }
    std::cout << "max_value: " << max_value << std::endl;
    out_txt_file.close();
    return 0;
}

int save_txt3(float* voxel_feature,std::string save_path,float seconds,int output_channel,int height)
{
    std::ofstream out_txt_file;
    out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // out_txt_file << fixed;
    out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    out_txt_file << seconds << std::endl;
    float max_value = 0;
    for(int i=0;i < height*output_channel; i++)
    {
   
        out_txt_file << *(voxel_feature + i) << ",";
        if ( *(voxel_feature + i) >max_value)
        {
            max_value =  *(voxel_feature + i) ;
        }
    
    }
    std::cout << "max_value: " << max_value << std::endl;
    out_txt_file.close();
    return 0;
}

int save_txt4(float* box_preds,int valid_line_num, std::string save_path,float seconds,int output_channel)
{
    std::ofstream out_txt_file;
    out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // out_txt_file << fixed;
    out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    out_txt_file << seconds << std::endl;
    float max_value = 0;
    for(int i=0;i < valid_line_num; i++)
    {
        // std::cout << "height_i: " << height_i << std::endl;
        for(int channel_i=0;channel_i < output_channel; channel_i++)
        {
        
            out_txt_file << *(box_preds + i*output_channel+channel_i) << ",";

        }
        out_txt_file << std::endl;
    }
    // std::cout << "max_value: " << max_value << std::endl;
    out_txt_file.close();
    return 0;
}

int save_txt5(std::vector<Bndbox> &nms_pred, std::string save_path,float seconds)
{
    std::ofstream out_txt_file;
    out_txt_file.open(save_path.c_str(),ios::out | ios::trunc);
    // out_txt_file << fixed;
    out_txt_file << setiosflags(ios::fixed) << setprecision(6);
    out_txt_file << seconds << std::endl;
    float max_value = 0;
    for(int i=0;i < nms_pred.size(); i++)
    {
        // std::cout << "height_i: " << height_i << std::endl;
        
            
        out_txt_file << nms_pred[i].x << ",";
        out_txt_file << nms_pred[i].y << ",";
        out_txt_file << nms_pred[i].z << ",";
        out_txt_file << nms_pred[i].w << ",";
        out_txt_file << nms_pred[i].l << ",";
        out_txt_file << nms_pred[i].h << ",";
        out_txt_file << nms_pred[i].rt << ",";
        out_txt_file << nms_pred[i].id << ",";
        out_txt_file << nms_pred[i].score << std::endl;

    }
    // std::cout << "max_value: " << max_value << std::endl;
    out_txt_file.close();
    return 0;
}

void save_result(std::vector<Bndbox> &res_,float *output,int voxel_num)
{
    
    for (int i = 0; i < voxel_num; i++) {
    auto Bb = Bndbox(output[i * 9],
                    output[i * 9 + 1], output[i * 9 + 2], output[i * 9 + 3],
                    output[i * 9 + 4], output[i * 9 + 5], output[i * 9 + 6],
                    static_cast<int>(output[i * 9 + 7]),
                    output[i * 9 + 8]);
    res_.push_back(Bb);
  }
}

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (argc == 2 && std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        std::ofstream p("se-ssd-spp.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "-d") {
        std::ifstream file("se-ssd-spp.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./se-ssd-spp -s  // serialize model to plan file" << std::endl;
        std::cerr << "./se-ssd-spp -d// deserialize plan file and run inference" << std::endl;
        return -1;
    }
    std::cout << "detection start   " << std::endl;
    IRuntime* runtime = createInferRuntime(rt_glogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    context->setOptimizationProfile(0);

    // int feature_map_size_x = BACKBONE_FEATURE_MAP_SIZE_X;
    // int feature_map_size_y = BACKBONE_FEATURE_MAP_SIZE_Y;
    // int feature_map_size_z = BACKBONE_FEATURE_MAP_SIZE_Z;
    // int feature_map_channel = BACKBONE_FEATURE_MAP_CHANNEL;


    int line_num = MAX_BOX_NUM;
    int feature_map_channel = LAST_DIMS;
    unsigned int voxel_feature_byte_size = 1 * line_num * feature_map_channel * sizeof(float);


    const ICudaEngine& work_engine = context->getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(work_engine.getNbBindings() == 4);
    void* buffers[4];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex1 = work_engine.getBindingIndex(INPUT_POINTS);
    const int inputIndex2 = work_engine.getBindingIndex(INPUT_POINTS_SIZE);
    const int outputIndex1 = work_engine.getBindingIndex(OUTPUT_VOXELS);
    // const int outputIndex2 = work_engine.getBindingIndex(OUTPUT_COORS);
    const int outputIndex3 = work_engine.getBindingIndex(OUTPUT_VOXEL_NUM);

    context->setBindingDimensions(inputIndex1, Dims3{1, MAX_POINTS,4});
    Dims dims1;
    dims1.d[0] = 1;
    dims1.nbDims = 1;
    context->setBindingDimensions(inputIndex2,dims1);
  
    // Create GPU buffers on device
    checkCudaErrors(cudaMalloc(&buffers[inputIndex1], 1 * MAX_POINTS * 4* sizeof(float)));
    checkCudaErrors(cudaMalloc(&buffers[inputIndex2], 1 * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex1],voxel_feature_byte_size));
    // checkCudaErrors(cudaMalloc(&buffers[outputIndex2],1 * output_max_voxel * 4 * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc(&buffers[outputIndex3],1 * sizeof(unsigned int)));

    // Create stream
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));


   
    float* voxel_feature = (float*)malloc(voxel_feature_byte_size);
    // unsigned int coors_byte_size = 1 * output_max_voxel * 4 * sizeof(unsigned int);
    // unsigned int * coors = (unsigned int*)malloc(coors_byte_size);
    // unsigned int voxel_num = 0;

    std::string Data_File = "../data/kitti_training_velodyne_reduced/";
    std::string save_root = "../data/outputs/";

    std::vector<Bndbox> nms_pred;
    nms_pred.reserve(100);
    std::vector<Bndbox> res_;

    for (int i = 0; i <301; i++) //7481 
    {
        std::cout << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%" << std::endl;
        std::string dataFile = Data_File;
        std::stringstream ss;

        ss<< i;

        int n_zero = 6;
        std::string old_string(ss.str());
        std::string new_path = std::string(n_zero - old_string.length(),'0') + old_string;
        dataFile += new_path;
        dataFile += ".bin";

        std::cout << "<<<<<<<<<<<" <<std::endl;
        std::cout << "load file: "<< dataFile <<std::endl;

        //load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        loadData(dataFile.data(), &data, &length);
        buffer.reset((char *)data);

        float* points = (float*)buffer.get();
        unsigned int points_size = length/sizeof(float)/4;

        std::cout << "first point:  " << points[0] << "," << points[1] << "," << points[2] << "," << points[3] << std::endl; 

        std::cout << "find points num: "<< points_size <<std::endl;
       

            
        // auto start = std::chrono::system_clock::now();
        const clock_t begin_time = clock();
        // auto st = system_clock::now();
        
        // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
        // checkCudaErrors(cudaMemcpyAsync(buffers[inputIndex1], points, 1 * MAX_POINTS * 4* sizeof(float), cudaMemcpyHostToDevice, stream));
        // checkCudaErrors(cudaMemcpyAsync(buffers[inputIndex2], &points_size, 1 * sizeof(unsigned int),cudaMemcpyHostToDevice, stream));

        checkCudaErrors(cudaMemcpy(buffers[inputIndex1], points, 1 * MAX_POINTS * 4* sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(buffers[inputIndex2], &points_size, 1 * sizeof(unsigned int),cudaMemcpyHostToDevice));

        context->enqueueV2(buffers, stream, nullptr);
       
        

        // checkCudaErrors(cudaMemcpyAsync(voxel_feature, buffers[outputIndex1], 
        //         1 * feature_map_size_x * feature_map_size_y * feature_map_size_z * feature_map_channel * sizeof(float), cudaMemcpyDeviceToHost, stream));
        

        checkCudaErrors(cudaMemcpy(voxel_feature, buffers[outputIndex1], 
                voxel_feature_byte_size, cudaMemcpyDeviceToHost));
        int voxel_num = 0;
        checkCudaErrors(cudaMemcpy(&voxel_num, buffers[outputIndex3], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::cout << "voxel_num: " << voxel_num << std::endl;

        // checkCudaErrors(cudaMemcpyAsync(coors, buffers[outputIndex2], 1 * output_max_voxel * 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        // std::cout << "doinference_44444" << std::endl;

        // checkCudaErrors(cudaMemcpyAsync(&voxel_num, buffers[outputIndex3], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        // std::cout << "voxel_num: " << voxel_num << std::endl;

    
        cudaStreamSynchronize(stream);

        save_result(res_,voxel_feature,voxel_num);
        nms_cpu(res_,NMS_THRESH,nms_pred);
        
        float seconds = float(clock() - begin_time) / 1000;
        // duration<double> diff = system_clock::now() - st;
        std::cout << "doinference cost time: " << seconds <<  "ms" << std::endl;
        // std::cout << "耗时:" << diff.count() << "s" << std::endl;

        //save to txt
        std::vector<std::string> strlist;
        std::string split_name("/");
        stringsplit(dataFile,split_name,strlist);
        std::string save_path = strlist.back();
        save_path = save_path.replace(save_path.find("b"),3,"txt");
        save_path  =  save_root + save_path;
        std::cout << save_path << std::endl;
        save_txt5(nms_pred,save_path,seconds);
        
        nms_pred.clear(); 
        res_.clear();
    }

    free(voxel_feature);
    // free(coors);
    // Release stream and buffers
    cudaStreamDestroy(stream);
    
    checkCudaErrors(cudaFree(buffers[inputIndex1]));
    checkCudaErrors(cudaFree(buffers[inputIndex2]));
    checkCudaErrors(cudaFree(buffers[outputIndex1]));
    // checkCudaErrors(cudaFree(buffers[outputIndex2]));
    // checkCudaErrors(cudaFree(buffers[outputIndex3]));
    
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
