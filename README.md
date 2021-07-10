# Set Transformer
An Implementation of [Set Transformer](https://github.com/juho-lee/set_transformer) that can work with variable sized sets. Out of the box, this function takes in sparse inputs in the format provided by Pytorch Geometric.

## Requirements
The code requires the following libraries:
1. Pytorch
2. Pytorch Geometric


## How to use
Import the modules from `modules.py` and run.

If you have your data in the form `x` and `ptr (or sizes)`, you can convert it to the form required by the module as
```python
>>> sizes = torch.tensor([3, 32, 18, 13, 90, 29])
>>> batch = torch.arange(sizes.size(0)).repeat_interleave(sizes)
```
