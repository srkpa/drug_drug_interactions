import torch

def cosine_attn(x1, x2, eps=1e-8):
    """Compute attention using cosine similarity"""
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = x2.norm(p=2, dim=-1, keepdim=True)
    return torch.matmul(x1, x2.transpose(-2, -1)) / (w1 * w2.transpose(-2, -1)).clamp(min=eps)

def mul_attn(x1, x2, **kwargs):
    attn = torch.matmul(x1, x2.transpose(-2, -1)) # B, N, M 
    return attn

def upsample_to(vec, d):
    """Convert a N,F vector to N*d, F vector (for concatenation purpose
    Not sure if this is smarter than just using for loop. But fight me!"""
    vec = vec.view(-1, vec.shape[-1])
    return vec.unsqueeze(0).expand(d, vec.shape[0], vec.shape[1]).contiguous().view(-1, vec.shape[-1])

def gradient_penalty(y, x):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones_like(y)
    dydx = torch.autograd.grad(outputs=y,
                                 inputs=x,
                                 grad_outputs=weight,
                                 retain_graph=True,
                                 create_graph=True,
                                 only_inputs=True)[0]
    dydx = dydx.view(dydx.size(0), -1)
    return ((dydx.norm(2, dim=1) - 1) ** 2).mean()

