# Tutorial

This folder contains 2 kinds of tutorials:
- end to end examples of how to use the library
- onboard team members and contributors on this project.

## End to end examples

* [XNLI classification](./bert%20e2e.ipynb): classification with / without optimizations (`Roberta` + `XNLI` classification task)
* [text generation](./t5%20e2e.ipynb): text generation with / without optimizations (`T5`)

## Learning materials

Tutorials below will show you how to implement a GPU kernel.  
It requires basic knowledge of how GPU works, in particular its memory hierarchy.  
If you are not familiar with that, check [this article](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html) first.

Tutorials below are written in `Pytorch` in the style of `triton` (rewriting is trivial), to ease the learning.

* [tiled matmul](./1%20-%20tiled%20matmul.ipynb): matrix multiplication implementation in `CUDA` style
* [online softmax](./3%20-%20online%20softmax.ipynb): parallelized softmax computation, a key ingredient of Flash Attention
* [Flash Attention](./4%20-%20flash%20attention.ipynb): attention computation without saving attention matrix to global memory
* [matmul offsets](./2%20-%20matmul%20offsets.ipynb): detailed explanations related to a performance trick used in 
  [`triton` matmul tutorial](https://triton-lang.org/master/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py)

**Flash Attention** tutorial covers most of what you need to know.