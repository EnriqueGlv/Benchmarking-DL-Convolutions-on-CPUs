# Benchmarking-DL-Convolutions-on-CPUs

This repository contains material to reproduce the experiments of the article [Benchmarking Deep Learning Convolutions on Energy-constrained CPUs](https://hal.science/hal-05285542), accepted for presentation at [HiPEAC 2026](https://www.hipeac.net/2026/krakow/#/) conference, in the context of [DASIP](https://dasip-2026.github.io/) workshop.

It provides several implementations for computing the forward pass through convolutionnal layers of a CNN. 

To cite this work, please refer to the paper:

```bibtex
@unpublished{galvez:hal-05285542,
  TITLE = {{Benchmarking Deep Learning Convolutions on Energy-constrained CPUs}},
  AUTHOR = {Galvez, Enrique and Cassagne, Adrien and Munier, Alix and Bouyer, Manuel},
  URL = {https://hal.science/hal-05285542},
  NOTE = {working paper or preprint},
  YEAR = {2025},
  MONTH = Dec,
  KEYWORDS = {Convolution algorithms ; Benchmarking ; Energy-constrained CPUs ; High performance ; Edge AI},
  PDF = {https://hal.science/hal-05285542v2/file/main.pdf},
  HAL_ID = {hal-05285542},
  HAL_VERSION = {v2},
}
```

(Note that this citation refer to the preprint, and will be updated.)

> [!NOTE]
> This project is a fork of [OneDNN version 3.4](https://uxlfoundation.github.io/oneDNN/v3.4/index.html). Useful information about OneDNN can be found in the [Official README](README-ONEDNN.md) or in the [documentation of OneDNNv3.4](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_build.html#build-the-library).

## Project structure

This repository extends OneDNN with the following implementations for deep learning convolutions:
- `direct`: Improves the naive algotirhm by implementing optimization techniques such as loop reordering and cache blocking. [[Zhang et al., 2018]](https://proceedings.mlr.press/v80/zhang18d.html?ref=https://)
- `im2row`: Transforms the input tensor into a matrix (with data duplication) to compute the convolution as a GEMM. Take [NHWC](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_understanding_memory_formats.html) as input and produce [NHWC](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_understanding_memory_formats.html) output.
- `im2col`: Same as im2row but takes [NCHW](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_understanding_memory_formats.html) as input and produces [NCHW](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_understanding_memory_formats.html) ad output.
- `MEC`: Based on im2row but uses slicing to reduce the size of the input buffer. [[Cho et al., 2017]](https://proceedings.mlr.press/v70/cho17a/cho17a.pdf)
- `wino`: Uses Winograd's method to reduce the number of multiplications, only implemented for 3x3 and 1-strided convolutions. [[Lavin et al., 2015]](https://openaccess.thecvf.com/content_cvpr_2016/papers/Lavin_Fast_Algorithms_for_CVPR_2016_paper.pdf)
- `pointwise`: Uses a GEMM to efficiently compute 1x1 non-strided convolutions.

These implementations are located in `src/cpu/lip6`. The files specific to this project are the following:

```
src
├── cpu
│   ├── lip6                                // contains our implementations
|   |   ├── lip6_convolution_direct.cpp
|   |   ├── lip6_convolution_direct.hpp
|   |   ├── lip6_convolution_im2col.cpp
|   |   ├── lip6_convolution_im2col.hpp
|   |   ├── lip6_convolution_im2row.cpp
|   |   ├── lip6_convolution_im2row.hpp
|   |   ├── lip6_convolution_MEC.cpp
|   |   ├── lip6_convolution_MEC.hpp
|   |   ├── lip6_convolution_pointwise.cpp
|   |   ├── lip6_convolution_pointwise.hpp
|   |   ├── lip6_convolution_utils.cpp
|   |   ├── lip6_convolution_utils.hpp
|   |   ├── lip6_convolution_winograd.cpp
|   |   └── lip6_convolution_winograd.hpp
│   ├── cpu_convolution_list.cpp            // manage implementation choice 
│   ├── gemm_convolution.hpp                // code for OneDNN's gemm_ref, fallback for our implementations 
│   └── CMakeLists.txt
```

According to [OneDNN's programming model](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_basic_concepts.html), each implementation resides in 2 files:
- `implem.hpp`: contains the primitive descriptor (OneDNN object which describes the problem parameters supported by the implementation) 
- `implem.cpp`: contains the code for the forward pass

## Build the project

This project is a fork of [OneDNN version 3.4](https://uxlfoundation.github.io/oneDNN/v3.4/index.html).

To build OneDNN, the following commands can be used:

```
cmake -B build
cmake --build build -j [nproc]
```

Additionnal instructions to build the project can be found in the [Official README](README-ONEDNN.md) or in the [documentation of OneDNNv3.4](https://uxlfoundation.github.io/oneDNN/v3.4/dev_guide_build.html#build-the-library).

## Environment variables

In order to get more control over the implementations, one can use the following environment variables in order to select a specific implementation or repeating the operator.

| variable                 | functions                                            |
|--------------------------|------------------------------------------------------|
| `LIP6_DISABLE_WINO`      | Disable `wino` implementation                        |
| `LIP6_DISABLE_POINTWISE` | Disable `pointwise` implementation                   |
| `LIP6_DISABLE_MEC`       | Disable `mec` implementation                         |
| `LIP6_DISABLE_IM2ROW`    | Disable `im2row` implementation                      |
| `LIP6_DISABLE_IM2COL`    | Disable `im2col` implementation                      |
| `LIP6_DISABLE_DIRECT`    | Disable `direct` implementation                      |
| `LIP6_NB_REPS`           | Set the number of repetitions inside a forward pass. |


> [!TIP]
> In this list, implementations are sorted by order of priority (defined in [cpu_convolution_list.cpp](src/cpu/cpu_convolution_list.cpp)).
> To run a specific implementation, all that is needed is to disable implementations with higher priority !

As an example, in order to run `direct` implementation 100 times, one can setup the following environment:

```
export LIP6_DISABLE_WINO=1
export LIP6_DISABLE_POINTWISE=1
export LIP6_DISABLE_MEC=1
export LIP6_DISABLE_IM2ROW=1
export LIP6_DISABLE_IM2COL=1

export LIP6_NB_REPS=100
```

## Run experiments using benchdnn

OneDNN comes with a tool called `benchdnn` which allow us to evaluate both the correctness and the performance of implemented primitives.

To check the correctness of `mec` implementation for `mb1ic256ih14oc256oh14kh3ph1` convolution: 

```
export LIP6_DISABLE_WINO=1
export LIP6_DISABLE_POINTWISE=1
build/tests/benchdnn/benchdnn --mode=C --repeats-per-prb=1 --conv mb1ic256ih14oc256oh14kh3ph1
```

To measure the performance of 100 repetitions of `im2row` implementation for `mb1ic256ih14oc256oh14kh3ph1` convolution:

```
export LIP6_DISABLE_WINO=1
export LIP6_DISABLE_POINTWISE=1
export LIP6_DISABLE_MEC=1
export LIP6_NB_REPS=100
build/tests/benchdnn/benchdnn --mode=P --repeats-per-prb=1 --conv mb1ic256ih14oc256oh14kh3ph1
```

> [!NOTE]
> Additionnal benchmarking options are described in [BenchDNN README](tests/benchdnn/doc/driver_conv.md).

## Run experiments using ONNX Runtime

In order to benchmark the convolution implementations of this project in a real-life scenario, it is possible to build ONNX Runtime from source using this project as execution provider. 

First, build and install this custom OneDNN library to a path `[ONEDNN_INSTALL_PATH]`: 

```
cd build
cmake --build . --parallel
cmake --install . --prefix [ONEDNN_INSTALL_PATH]
```

To validate the installation of OneDNN:

```
ls [ONEDNN_INSTALL_PATH]/lib | grep dnnl
```

Then, clone ONNX Runtime's repository:

```
git clone --recursive https://github.com/Microsoft/onnxruntime.git
cd onnxruntime
```

Once it is done, build ONNX Runtime with your OneDNN library in your path:

```
export DNNLROOT=[ONEDNN_INSTALL_PATH]
export LD_LIBRARY_PATH=[ONEDNN_INSTALL_PATH]/lib:$LD_LIBRARY_PATH
./build.sh --config RelWithDebInfo --use_dnnl --build_wheel --parallel \
  --cmake_extra_defines "CMAKE_PREFIX_PATH=[ONEDNN_INSTALL_PATH]"
```

> [!NOTE]
> Additionnal information on the procedure to build ONNX Runtime from source can be found in the [official instructions](https://onnxruntime.ai/docs/build/).

To use this ONNX Runtime as a python library, we recommend to build the wheel in a virtual environment:

```
python -m venv ort.venv
source ort.venv/bin/activate
pip install build/Linux/RelWithDebInfo/dist/*.whl
```

Finally, the following script can be used to run the inference of a model (here ResNet50v1.5) for an image "example.jpeg", using our custom OneDNN as backend:

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
import time

# load model
MODEL_PATH="path/to/resnet50v1.5.onnx"
IMG_PATH="example.jpeg"

# initialize onnx runtime and disable kernel fusion for reproducibility of experiments
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# create inference session, using DNNL (OneDNN) as execution provider
EP_list = ['DnnlExecutionProvider', 'CPUExecutionProvider']
sess = ort.InferenceSession(MODEL_PATH, sess_options=sess_options, providers=EP_list)

# preprocessing of a 1x3x224x224 NCHW float32 image
im = Image.open(IMG_PATH).convert('RGB').resize((224,224))
x = np.array(im).astype(np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
x = (x - mean) / std
x = x.transpose(2,0,1)[None, ...]

# run the inference
input_name = sess.get_inputs()[0].name
start = time.time()
out = sess.run(None, {input_name: x})
end = time.time()

# print inference summary
print(f"Inference time: {end - start:.4f} seconds")
print("providers used:", sess.get_providers())
print("Graph optimization level:", sess_options.graph_optimization_level)
print("Execution mode:", sess_options.execution_mode)
print("--")
print("Top5 predictions: (label: score)")
scores = out[0][0]
top5_idx = np.argsort(scores)[-5:][::-1]
for i in top5_idx:
  print(f"- {i}: {scores[i]}")
```

> [!NOTE]
> The following packages may be needed in your `ort.venv` in order to run the script below:
> ```
> pip install "numpy<2" 
> pip install pillow 
> ```

This script can be run with the following commands (from the virtual environment where ONNX runtime is installed):

```
export LD_LIBRARY_PATH=[ONEDNN_INSTALL_PATH]/lib:$LD_LIBRARY_PATH
export ONEDNN_VERBOSE=1
python run_resnet_dnnl.py
```

> [!IMPORTANT]
> `ONEDNN_VERBOSE=1` allows you to check which convolution drivers are executed by ONNX Runtime. If the name of the driver is in the form `lip6:xxx`, it means that our implementations are called. 
