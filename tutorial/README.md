# Tutorial

This folder contains tutorials to on-board team members and others on this project.
They are written in `Pytorch` in the style of `triton`, to ease the switch.

* [tiled matmul](./tutorial/1%20-%20tiled%20matmul.ipynb): matrix multiplication implementation in `CUDA` style
* [matmul offsets](./tutorial/2%20-%20matmul%20offsets.ipynb): detailed explanations related to a trick used in `triton` matmul tutorial
* [online softmax](./tutorial/3%20-%20online%20softmax.ipynb): parallelized softmax computation, a key ingredient of flash attention
* [flash attention](./tutorial/4%20-%20flash%20attention.ipynb): attention computation without saving attention matrix to global memory
* [test end to end](./tutorial/bert%20e2e.ipynb): classification with / without optimizations (`Roberta` + `XNLI` classification task)
