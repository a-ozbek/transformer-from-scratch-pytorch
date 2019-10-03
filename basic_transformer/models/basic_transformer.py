import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicTransformer(nn.Module):
    """
    Basic Transformer
    """
    def __init__(self, dim, num_embeddings, embedding_dim, avg_out_seq=True):
        super(BasicTransformer, self).__init__()
        self.dim = dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.avg_out_seq = avg_out_seq
        
        self.embed_layer = nn.Embedding(num_embeddings=self.num_embeddings + 1, 
                                        embedding_dim=self.embedding_dim)
        
        self.W_q = torch.rand(dim, dim, requires_grad=True)
        self.W_k = torch.rand(dim, dim, requires_grad=True)
        self.W_v = torch.rand(dim, dim, requires_grad=True)
        
        self.W_q = self.W_q.cuda()
        self.W_k = self.W_k.cuda()
        self.W_v = self.W_v.cuda()
        
        
        
        self.linear = nn.Linear(dim, dim)
        self.linear_clf = nn.Linear(dim, 1)
    
    def forward(self, x):
        """
        x: (after embedding)
            dim0: batch dimension
            cols: timesteps (or sequence)
            rows: dimensionality (dim)
        """
        x = self.embed_layer(x)
        x = x.permute(0, 2, 1)
        
        q = torch.matmul(self.W_q, x)
        k = torch.matmul(self.W_k, x)
        v = torch.matmul(self.W_v, x)  # 32 x 8 x 17
        
        batch_size = x.shape[0]
        timesteps = x.shape[2]
        
        y = torch.empty(batch_size, self.dim, timesteps)
        y = y.cuda()
        
        for i in range(timesteps):
            q_i = q[:, :, i]
            # get weights
            weights = torch.matmul(q_i[:, np.newaxis, :], k)
            weights = weights.squeeze(dim=1)
            # scale weights
            weights = weights / np.sqrt(self.dim)
            # softmax weights
            weights = torch.softmax(weights, dim=1)
            
            y[:, :, i] = torch.sum(weights[:, np.newaxis, :] * v, dim=2)
            
        # apply linear layer to each timestep
        y = y.permute(0, 2, 1) # - permute y to be able to apply it
        y = self.linear(y)
        y = y.permute(0, 2, 1) # - permute back
        y = F.relu(y)
        
        if self.avg_out_seq:
            y = y.mean(dim=2)  # average out the sequence
        
        # last clf layer
        y = self.linear_clf(y)
        
        y = F.sigmoid(y)
            
        return y
    