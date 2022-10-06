# Tutorial

This folder contains tutorials to on-board team members and others on this project.
They are written in `Pytorch` in the style of `triton`, to ease the switch.

* tiled matmul: matrix multiplication implementation in `CUDA` style;
* matmul offsets: detailed explanations related to a trick used in `triton` matmul tutorial;
* online softmax: parallelized softmax computation, a key ingredient of flash attention;
* flash attention: attention computation without saving attention matrix to global memory.