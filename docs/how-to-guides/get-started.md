# Get started

## Install

To install Kernl library, you just have to pip install it:

``` { .bash }
python3 -m pip install install 'git+https://github.com/ELS-RD/kernl'
```

## Optimize a model

Then, in your progam, you have to import the optimization function and apply it to your model:

``` { .py }
from transformers import AutoModel
from kernl.model_optimization import optimize_model

model = AutoModel.from_pretrained(model_name).eval().cuda()
optimize_model(model)
```

That's it, you have your model with Kernl's optimizations !

Beware, Kernl works only on Ampere GPU and with python `3.9.*` for now.

Look at the [repository README](https://github.com/ELS-RD/kernl#readme) for more informations.
