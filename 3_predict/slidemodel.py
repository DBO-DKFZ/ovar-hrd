import torch
from torch import nn
from transformer import Transformer
from einops import repeat

class DynamicClassifier(nn.Module):
    def __init__(self, dim, num_patches, num_classes, depth, heads, dim_head, mlp_factor=4, dropout=0.,
                 emb_dropout=0., eps=1e-8, with_pos_emb=True):
        super().__init__()
        if with_pos_emb:
            self.num_emb = num_patches + num_classes + 3
        else:
            self.num_emb = num_classes + 3
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_emb, dim))
        k = dim ** -0.5
        self.weight_token = nn.Parameter(torch.empty(1, num_classes, dim).uniform_(-k, k))
        self.feat_token = nn.Parameter(torch.randn(1, 1, dim))
        self.scale_token = nn.Parameter(torch.ones(1, 1, dim))
        self.bias_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        mlp_dim = int(dim * mlp_factor)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_bias = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.to_scale = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
        self.to_feat = nn.LayerNorm(dim)

        self.register_buffer("eps", torch.tensor(eps))
        self.num_classes = num_classes

    def forward(self, x, static=False):
        b, n, _ = x.shape
        
        # Classifier weights
        weight_tokens = repeat(self.weight_token, '1 c d -> b c d', b = b)
        scale_tokens = repeat(self.scale_token, '1 1 d -> b 1 d', b = b)
        bias_tokens = repeat(self.bias_token, '1 1 d -> b 1 d', b = b)
        # Feature vector
        feat_tokens = repeat(self.feat_token, '1 1 d -> b 1 d', b = b)
        
        x = torch.cat((weight_tokens, scale_tokens, bias_tokens, feat_tokens, x), dim=1)
        x[:, :self.num_emb] += self.pos_embedding[:, :self.num_emb]
            
        x = self.dropout(x)
        
        if static:
            weight = x[:, :self.num_classes]
            scale, bias = torch.unbind(x[:, self.num_classes: self.num_classes+2], dim=1)
            x = x[:, self.num_classes + 2:]
            x = self.transformer(x)
            feat = x[:, 0]
            
        else:
            x = self.transformer(x)
            weight = x[:, :self.num_classes]
            scale, bias, feat = torch.unbind(x[:, self.num_classes: self.num_classes+3], dim=1)
        
        scale = self.to_scale(scale)
        bias = self.to_bias(bias)
        feat = self.to_feat(feat)
        
        # Weight normalization
        weight = scale.unsqueeze(2) * weight / (weight.norm(dim=2, keepdim=True) + self.eps)

        # Classifier
        out = torch.bmm(feat.unsqueeze(1), weight.transpose(1, 2)).squeeze(1) + bias

        return out
    
class StaticClassifier(nn.Module):
    def __init__(self, dim, num_patches, num_classes, depth, heads, dim_head, mlp_factor=4, dropout=0., emb_dropout=0., eps=1e-8, with_pos_emb=True):
        super().__init__()
        if with_pos_emb:
            self.num_emb = num_patches + 1
        else:
            self.num_emb = 1
            
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_emb, dim))
            
        k = dim ** -0.5
        
        self.feat_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        mlp_dim = int(dim * mlp_factor)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_feat = nn.LayerNorm(dim)
        self.num_classes = num_classes
        
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, x, static=False):
        b, n, _ = x.shape
        
        # Feature vector
        feat_tokens = repeat(self.feat_token, '1 1 d -> b 1 d', b = b)
        
        x = torch.cat((feat_tokens, x), dim=1)
        
        x[:, :self.num_emb] += self.pos_embedding[:, :self.num_emb]
            
        x = self.dropout(x)
        x = self.transformer(x)
        feat = x[:, 0]
        feat = self.to_feat(feat)
        out = self.classifier(feat)

        return out