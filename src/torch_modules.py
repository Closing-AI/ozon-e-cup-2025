from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.BatchNorm1d(out_features)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MultiModalAdapter(nn.Module):
    def __init__(self, input_dims, emb_dim, agg: str):
        """
        input_dims: list of input dims for each modality, e.g. [128, 512, 100, 300]
        emb_dim: output embedding dimension for all modalities
        agg: aggregation type ("concat" or "stack")
        """
        super().__init__()

        assert agg in ["concat", "stack"], "agg must be 'concat' or 'stack'"
        assert len(input_dims) == 4, "Expecting 4 input tensors"

        self.agg = agg
        self.proj_layers = nn.ModuleList([LinearProjection(in_dim, emb_dim) for in_dim in input_dims])

    def forward(self, x):
        outs = [proj(x) for proj, x in zip(self.proj_layers, x)]

        if self.agg == "concat":
            # Concatenate along the feature dimension
            seq = torch.cat(outs, dim=-1)
        else:
            # Stack as sequence: (batch, seq_len=5, emb_dim)
            seq = torch.stack(outs, dim=1)
        return seq


class MLPBinaryHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.norm = nn.RMSNorm(input_dim)
        self.layer1 = nn.Sequential(
            OrderedDict(
                [("linear1", nn.Linear(input_dim, 128)), ("act1", nn.LeakyReLU(0.1)), ("dropout", nn.Dropout(0.1))]
            )
        )
        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    ("linear2", nn.Linear(128, 32)),
                    ("norm2", nn.BatchNorm1d(32)),
                    ("act2", nn.LeakyReLU(0.1)),
                ]
            )
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.head(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.rms_norm = nn.RMSNorm(emb_dim)

    def forward(self, x):
        # x: (batch, seq_len, emb_dim)
        x_norm = self.rms_norm(x)
        attn_out, _ = self.mha(x_norm, x_norm, x_norm)
        out = x_norm + attn_out

        return out


class MLPBlock(nn.Module):
    def __init__(self, emb_dim, mlp_hidden_dim=None, dropout=0.1):
        super().__init__()

        self.rms_norm = nn.RMSNorm(emb_dim)
        hidden_dim = mlp_hidden_dim or emb_dim * 4
        self.fc1 = nn.Linear(emb_dim, hidden_dim * 2)  # Double for SwiGLU
        self.fc2 = nn.Linear(hidden_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.rms_norm(x)
        x_proj, x_gate = self.fc1(x_norm).chunk(2, dim=-1)
        x_act = F.silu(x_gate) * x_proj  # SwiGLU activation
        x_mlp = self.fc2(x_act)
        out = x + self.dropout(x_mlp)

        return out


class MMTransformerEncoder(nn.Module):
    def __init__(self, input_dims, emb_dim, num_heads, num_layers, mlp_hidden_dim=None):
        """
        input_dims: list of input dims for each modality
        emb_dim: embedding dimension for all modalities
        num_heads: number of attention heads
        num_layers: number of (Attention + MLP) blocks
        mlp_hidden_dim: hidden dim for MLP block (optional, default: emb_dim * 4)
        """
        super().__init__()

        self.projector = MultiModalAdapter(input_dims, emb_dim, agg="stack")
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([AttentionBlock(emb_dim, num_heads), MLPBlock(emb_dim, mlp_hidden_dim)])
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x):
        # Project and stack as sequence
        x = self.projector(x)  # (batch, seq_len, emb_dim)
        for attn, mlp in self.layers:
            x = attn(x)
            x = mlp(x)

        # Pooling: mean over sequence
        x_pooled = x.mean(dim=1)
        x_norm = self.final_norm(x_pooled)
        logits = self.classifier(x_norm).squeeze(-1)

        return logits
