import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalProjector(nn.Module):
    def __init__(self, input_dims, emb_dim):
        """
        input_dims: list of input dims for each modality, e.g. [128, 512, 50, 10, 300]
        emb_dim: output embedding dimension for all modalities
        """
        super().__init__()

        assert len(input_dims) == 5, "Expecting 5 input tensors"
        self.proj_layers = nn.ModuleList([nn.Linear(in_dim, emb_dim) for in_dim in input_dims])

    def forward(self, x1, x2, x3, x4, x5):
        outs = [proj(x) for proj, x in zip(self.proj_layers, [x1, x2, x3, x4, x5])]

        # Stack as sequence: (batch, seq_len=5, emb_dim)
        seq = torch.stack(outs, dim=1)
        return seq


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
    def __init__(self, emb_dim, mlp_hidden_dim=None):
        super().__init__()

        self.rms_norm = nn.RMSNorm(emb_dim)
        hidden_dim = mlp_hidden_dim or emb_dim * 4
        self.fc1 = nn.Linear(emb_dim, hidden_dim * 2)  # Double for SwiGLU
        self.fc2 = nn.Linear(hidden_dim, emb_dim)

    def forward(self, x):
        x_norm = self.rms_norm(x)
        x_proj, x_gate = self.fc1(x_norm).chunk(2, dim=-1)
        x_act = F.silu(x_gate) * x_proj  # SwiGLU activation
        x_mlp = self.fc2(x_act)
        out = x + x_mlp

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

        self.projector = MultiModalProjector(input_dims, emb_dim)
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([AttentionBlock(emb_dim, num_heads), MLPBlock(emb_dim, mlp_hidden_dim)])
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(emb_dim)
        self.classifier = nn.Linear(emb_dim, 1)

    def forward(self, x1, x2, x3, x4, x5):
        # Project and stack as sequence
        x = self.projector(x1, x2, x3, x4, x5)  # (batch, seq_len, emb_dim)
        for attn, mlp in self.layers:
            x = attn(x)
            x = mlp(x)

        # Pooling: mean over sequence
        x_pooled = x.mean(dim=1)
        x_norm = self.final_norm(x_pooled)
        logits = self.classifier(x_norm).squeeze(-1)

        return logits
