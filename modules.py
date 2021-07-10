import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch_geometric
import math

def _gen_mask(x_mask, y_mask):
    '''
        Generates mask of shape [B, 1, max_len_x, max_len_y]
    '''
    mask = (x_mask.unsqueeze(-1) & y_mask.unsqueeze(-2)).unsqueeze(1)
    return mask

class MAB(torch.nn.Module):
    def __init__(self, dim_q, dim_k, dim_head, num_heads, ln=False):
        super().__init__()
        
        self.dim_head = dim_head
        self.num_heads = num_heads
        dim_v = dim_head * num_heads

        self.dim_v = dim_v
        self.dim_k = dim_k
        self.dim_q = dim_q

        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)


        if ln:
            self.ln0 = nn.LayerNorm(dim_v)
            self.ln1 = nn.LayerNorm(dim_v)

        self.fc_o = nn.Linear(dim_v, dim_v)


    def forward(self, q, k, batch_q, batch_k):
        
        batch_size = batch_q.max() + 1

        # Normal MLP application
        q = self.fc_q(q) # [B * P, DV]
        k,v = self.fc_k(k), self.fc_v(k) # [B * P, DV]


        # Padding to [B, max_len_q/max_len_k, dim_v]
        q, mask_q = torch_geometric.utils.to_dense_batch(q, batch_q)
        k, mask_k = torch_geometric.utils.to_dense_batch(k, batch_k)
        v, mask_v = torch_geometric.utils.to_dense_batch(v, batch_k)

        # Now reshaping them to [B, max_len_q/max_len_k, num_heads, dim_heads]
        q = q.view(batch_size, -1, self.num_heads, self.dim_head) #[B, P, NH, DH]
        k = k.view(batch_size, -1, self.num_heads, self.dim_head) #[B, P, NH, DH]
        v = v.view(batch_size, -1, self.num_heads, self.dim_head) #[B, P, NH, DH]

        # Attention score with shape [B, num_heads, max_len_q, max_len_k]
        e = torch.einsum('bxhd,byhd->bhxy',q,k)
        # Normalize
        e = e / math.sqrt(self.dim_head)

        # Generate mask of shape [B, 1, max_len_q, max_len_k]
        mask = _gen_mask(mask_q, mask_k)
        e = e.masked_fill(mask == 0, float('-inf'))

        # Apply Softmax
        alpha = torch.softmax(e, dim=-1)
        alpha = alpha.masked_fill(mask == 0, 0.)


        # Sum of values weighted by alpha
        out = torch.einsum('bhxy,byhd->bxhd', alpha, v) # [B, max_len_q, num_heads, dim_heads]

        out = out + q #? Check this because the MAB thing has this but DGL doesn't (This seems correct)

        # Convert back to sparse representation
        out = out.contiguous().view(batch_size, -1, self.dim_v) # [B, max_len_q, DV] # Verify if contiguous is correct
        out = out[mask_q == 1].view(-1, self.dim_v) #[B * P, DV]

        
        out = out if getattr(self, 'ln0', None) is None else self.ln0(out)
        out = out + F.relu(self.fc_o(out))
        out = out if getattr(self, 'ln1', None) is None else self.ln1(out)

        return out


class SAB(torch.nn.Module):
    def __init__(self, dim_in, dim_head, num_heads, ln=False):
        super().__init__()
        self.mab = MAB( dim_in, dim_in, dim_head, num_heads, ln=ln)

    def forward(self, x, batch):
        return self.mab(x, x, batch, batch)


class ISAB(torch.nn.Module):
    def __init__(self, dim_in, dim_head, num_heads, num_inds, ln=False):
        super().__init__()
        
        self.num_inds = num_inds

        dim_out = dim_head * num_heads
        self.dim_out = dim_out

        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        
        self.mab0 = MAB(dim_out, dim_in, dim_head, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_head, num_heads, ln=ln)


    def forward(self, x, batch):
        batch_size = batch.max() + 1

        i = self.I.repeat(batch_size, 1, 1).view(-1, self.dim_out)
        batch_i = torch.arange(batch_size, device=i.device).unsqueeze(-1).repeat(1, self.num_inds).view(-1)

        H = self.mab0(i, x, batch_i, batch)
        return self.mab1(x, H, batch, batch_i)


class PMA(torch.nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super().__init__()

        self.dim = dim
        self.num_seeds = num_seeds
        
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        
        dim_head = dim // num_heads

        self.mab = MAB(dim, dim, dim_head, num_heads, ln=ln)

    def forward(self, x, batch):
        batch_size = batch.max() + 1

        s = self.S.repeat(batch_size, 1, 1).view(-1, self.dim)
        batch_s = torch.arange(batch_size, device=s.device).unsqueeze(-1).repeat(1, self.num_seeds).view(-1)
        
        return self.mab(s, x, batch_s, batch)
