from modules import *

ptr = torch.tensor([3, 32, 21, 80, 90, 54, 32, 61])
batch = torch.arange(ptr.size()).repeat_interleave(ptr)
x = torch.randn(ptr.sum(), 3)

dim_in = 3
dim_head = 4
num_heads = 4
num_inds = 30
ln = False

isab = ISAB(
    dim_in,
    dim_head,
    num_heads,
    num_inds,
    ln=ln
)

out = isab(x, batch)