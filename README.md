**AAAI2025**: Beyond Skip Connection: Pooling and Unpooling Design for Elimination Singularities  

[![Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34278)  

# Pytorch-cifar100

practice on cifar100 using pytorch

## Requirements

This is my experiment eviroument
- python3.6
- pytorch1.6.0+cu101


## Usage

### 1. enter directory
```bash
$ cd pytorch-cifar100
```

### 2. dataset
I will use cifar100 dataset from torchvision since it's more convenient, but I also
kept the sample code for writing your own dataset module in dataset folder, as an
example for people don't know how to write it.

### 3. train the model
You need to specify the net you want to train using arg -net

```bash
# use gpu to train vgg16
$ python train.py -net vgg16 -gpu
```
## Acknowledgements
This project was based on (https://github.com/weiaicunzai/pytorch-cifar100).
Special thanks to the contributors for making development easier.  


