# LLama 2 optimization

The purpose of this experiment is to improve Llama v2 performance by fusing kernels together.  
Note that code should be run on top of Triton commit [69a806c](https://github.com/openai/triton/tree/69a806c745aa604fec6bd317628d3dc293aa1e46).
Main triton branch has some CPU overhead probably because of the add of AMD GPUs support (and some new mechanism to load the right backend).

We measured on consumer grade GPU 3090 RTX a speed up from 30 to 54 tokens/sec for 7B model.
The purpose is not to get extreme perf, there are many easy things to do to get even better performances.
Also, think about replacing the triton jit part by a lighter launcher if you want to push perf higher.


We tried to keep Llama model code as close as possible to the original one.  
In particular we removed all the multi GPU support code and replaced ut by classical local execution function (like Linear module).
It makes things more simple to run, less overhead for PyTorch benchmark, and at the end it is easier to understand.

More details about what is done and how it works are in our article [here]().
