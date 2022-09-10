# SE-SDD-AI-TRT

SE-SSD-AI-TRT(**SE-SSD ALL IN TensorRT**,NMS not implemented in TensorRT,implemented in c++) 

SE-SSD consists of six parts:
- preprocess: generate voxel, it is implemented in voxelGenerator.cu,it is a TensorRT plugin
- 3D backbone: 3D backbone include 3D sparse Convolution and 3D Submanifold Convolution. sparseConv3dlayer.cu is a TensorRT plugin for 3D sparse Convolution, and submConv3dlayer.cu is a TensorRT plugin for 3D Submanifold Convolution.
- neck: this part is mainy implemented by TensorRT aip, because they are all general modules. the function of sparse2Dense.cu is  from sparse tensor to dense tensor
- head: this part is mainy implemented by TensorRT aip.
- postprocess: it includes anchorGenerate and decoder, they are implemented by generateAnchorDecode.cu, it is also a plugin.
- 3D NMS: it comes from  https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars/blob/main/src/postprocess.cpp

## Config

- all config in params.h
- FP16/FP32 can be selected by USE_FP16 in params.h
- GPU id can be selected by DEVICE in params.h
- NMS thresh can be modified by NMS_THRESH in params.h

## How to Run

1. **build SE-SSD-AI-TRT and run**

```
firstly, install TensorRT,my environment is ubuntu 18.04, cuda 10.2,cudnn8.2.
I installed TensorRT with TensorRT-8.0.1.6.Linux.x86_64-gnu.cuda-10.2.cudnn8.2.tar.gz.

after that, modify CMakeLists.txt
include_directories(/home/xxx/softwares/nvidia/TensorRT-8.0.1.6/include)
link_directories(/home/xxx/softwares/nvidia/TensorRT-8.0.1.6/lib)
Change these two lines to your own path

cd SE-SSD-AI-TRT
mkdir build
cd build
cmake ..
make
sudo ./se-ssd-ai-trt -s             // serialize model to plan file i.e. 'se-ssd-ai-trt.engine'
sudo ./se-ssd-ai-trt -d    // deserialize plan file and run inference, lidar points will be processed.
predicted outputs saved in SE-SSD-AI-TRT/data/outputs folder

```
**one frame takes about 1-2 seconds on my laptop with Intel(R) Core(TM) i5-7300HQ and NVIDIA GeForce GTX 1050 Mobile(1050ti), it is very slow, needs to be optimized in  the future.**

2. **show predicted 3D boxes in the lidar frame** 

```
fristly install python moudles by tools/requirements.txt. for show boxes in points, just install mayavi
cd tools
python show_box_in_points.py
warning: do not close current Mayavi Scene window, type c in running terminal, 
it will show next lidar frame with predited 3d boxes in current Mayavi Scene window. 

```
![Image text](https://raw.githubusercontent.com/jingyue202205/SE-SSD-AI-TRT/master/pics/000010.png)
![Image text](https://raw.githubusercontent.com/jingyue202205/SE-SSD-AI-TRT/master/pics/snapshot.png)



## More Information

Reference code:

[SE-SSD](https://github.com/Vegeta2020/SE-SSD)  

[spconv](https://github.com/poodarchu/spconv)

[tensorrtx](https://github.com/wang-xinyu/tensorrtx) 

[tensorrt_plugin](https://github.com/NVIDIA/TensorRT/tree/main/plugin)

[CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)

[frustum_pointnets_pytorch](https://github.com/simon3dv/frustum_pointnets_pytorch)





