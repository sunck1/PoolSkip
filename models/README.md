# Modeling

We provide an example demonstrating how to integrate the `PoolSkip` module into a model.
If you'd like to use a different base model, please download it from [weiaicunzai/pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100).

In this folder:

* `resnet.py` contains the original ResNet model.

* `resnet_poolskip.py` contains the modified version with the PoolSkip module integrated.

The different between two version is:

1. Import the `PoolSkip` module

```python
# Import the `PoolSkip` module
import sys
sys.path.append('..')
from pool_skip import pool_skip
```

2. In line 35, 40, 68, 73, 78, we plug the module into nn.Sequential.
```python
pool_skip(out_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_size=2),
```