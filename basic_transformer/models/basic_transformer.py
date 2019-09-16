import numpy as np
import torch
import torch.nn as nn
import torch.functional as F


class BasicTransformer:
    """
    Basic Transformer
    """
    def __init__(self, dim, avg_out_seq=True):
        self.dim = dim
        self.avg_out_seq = avg_out_seq
        
        self.W_q = torch.rand(dim, dim, requires_grad=True)
        self.W_k = torch.rand(dim, dim, requires_grad=True)
        self.W_v = torch.rand(dim, dim, requires_grad=True)
    
    def forward(self, x):
        """
        x: 
            dim0: batch dimension
            cols: timesteps (or sequence)
            rows: dimensionality (dim)
        """
        q = torch.matmul(self.W_q, x)
        k = torch.matmul(self.W_k, x)
        v = torch.matmul(self.W_v, x)  # 32 x 8 x 17
        
        batch_size = x.shape[0]
        timesteps = x.shape[2]
        
        y = torch.empty(batch_size, self.dim, timesteps)
        
        for i in range(timesteps):
            q_i = q[:, :, i]
            # get weights
            weights = torch.matmul(q_i[:, np.newaxis, :], k).squeeze()
            # scale weights
            weights = weights / np.sqrt(self.dim)
            # softmax weights
            weights = torch.softmax(weights, dim=1)
            
            y[:, :, i] = torch.sum(weights[:, np.newaxis, :] * v, dim=2)
        
        if self.avg_out_seq:
            y = y.mean(dim=2)  # average out the sequence
            
        return y
    