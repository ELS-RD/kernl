# Hazyresearch Kernel implementation

Repo: https://github.com/HazyResearch/flash-attention

Taken from commit 557781933dbc13cc4e75626bb064b4784869426f

Backward is currently broken on 3090 RTX cards, only Fw is tested.

Most important difference with our implementations is that input shapes are different (axis not in the same order).
