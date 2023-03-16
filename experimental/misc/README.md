# Random experiments

* dot_perf.py: performance comparison of different dot product implementations (related to case where QK^t with Q has 1 row, and K/V much more)
* transpose.py: perform a transposition of a matrix during the GM loading

This has been written with Triton 1.x, and is not compatible with Triton 2.x.