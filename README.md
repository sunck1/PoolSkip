<div align="center">

# Beyond Skip Connection: Pooling and Unpooling Design for Elimination Singularities

**Authors**: Chengkun Sun, Jinqian Pan, Zhuoli Jin, Russell Stevens Terry, Jiang Bian, Jie Xu

[![arXiv](https://img.shields.io/badge/arXiv-2409.13154-red)](https://arxiv.org/abs/2409.13154)
[![Paper](https://img.shields.io/badge/AAAI2025-Paper%20Link-blue)](https://ojs.aaai.org/index.php/AAAI/article/view/34278)

</div>

<br>
<p align="center">
  <img src="./Supplementary materials/arch.drawio.png" width="1200">
</p>

## Installation
```bash
git clone git@github.com:sunck1/PoolSkip.git
cd PoolSkip
conda env create -f environment.yml
conda activate PoolSkip
```

## Training the model

```bash
python train.py -net resnet18_poolskip -gpu
```

> [!NOTE]
> We use CIFAR100 for practice.

## Integration

>[!TIP]
> To use our `PoolSkip` module directly, you can download [`pool_skip.py`](./pool_skip.py) and plug it into your architecture.  
>
> See the `models` folder for details.

To integrate PoolSkip, simply insert it after the convolutional layer, but before the activation function and batch normalization layer.
While effective with other convolution kernel sizes, the performance is best when applied after a 1x1 convolution.


## Effect on ResNet

This structure is particularly effective for addressing elimination singularity in convolutional neural networks. After applying PoolSkip, we observed a significant reduction in the number of shallow-layer 0 weights on ResNet architectures, improving model robustness and feature learning.

<div align="center"> <img src="./Supplementary materials/l2_l1.png" alt="ResNet Effect" width="1200"/> </div> <p align="left"> <small><i>Figure: $\frac{l_{2}}{l_{1}}$ value quantitative comparison in ResNet350 and ResNet420 on CIFAR10 and CIFAR100 Datasets. The $\frac{l_{2}}{l_{1}}$ values were computed based on the output sequence of the network, with and without the incorporation of the Pool Skip. The plot highlights a moderate alleviation of the network degradation issue in shallow layers upon the integration of Pool Skip. Note: The horizontal axis represents the layers of the network along the output direction, from left to right. The “Pool Skip S4” means
the size of Pool operation kernel is 4, “Pool Skip S2” does 2.</i><small> </p>

## Mathematical Derivation
For a complete explanation of the two compensation effects introduced by PoolSkip, including rigorous mathematical proofs, please refer to the provided **[Mathematical Derivation PDF](./Supplementary%20materials/Mathematical_proof.pdf)**.

## Acknowledgements
This project was based on (https://github.com/weiaicunzai/pytorch-cifar100).
Special thanks to the contributors for making development easier.  

## Citation
If this project contributes to your work, please cite the following paper:

```
@inproceedings{sun2025beyond,
  title={Beyond Skip Connection: Pooling and Unpooling Design for Elimination Singularities},
  author={Sun, Chengkun and Pan, Jinqian and Jin, Zhuoli and Terry, Russell Stevens and Bian, Jiang and Xu, Jie},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={19},
  pages={20672--20680},
  year={2025}
}
```
