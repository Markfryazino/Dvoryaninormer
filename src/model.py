import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttentionWithShift(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = self.hidden_dim // self.num_heads

        self.key_linear = nn.Linear(hidden_dim, hidden_dim)
        self.query_linear = nn.Linear(hidden_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask, shift=None):
        q = self._reshape_to_heads(self.query_linear(q))
        k = self._reshape_to_heads(self.key_linear(k))
        v = self._reshape_to_heads(self.value_linear(v))

        attn_mask = self._reshape_mask(mask)

        output = self._attention_forward(q, k, v, attn_mask, shift)
        output = self.output_linear(self._reshape_from_heads(output))
        return output

    def _attention_forward(self, q, k, v, mask, shift=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if shift is not None:
            scores = scores + shift

        scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        return torch.matmul(scores, v)

    def _reshape_to_heads(self, x):
        batch_size, seq_len, hidden_dim = x.size()
        return x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)\
                .transpose(1, 2)\
                .reshape(batch_size * self.num_heads, seq_len, self.head_dim)

    def _reshape_mask(self, mask):
        seq_len = mask.size(1)

        reshaped_mask = mask.repeat([self.num_heads, 1])
        scorelike_mask_d1 = reshaped_mask.repeat_interleave(seq_len, 1).reshape(-1, seq_len, seq_len)
        return scorelike_mask_d1 * scorelike_mask_d1.transpose(1, 2)

    def _reshape_from_heads(self, x):
        batch_size, seq_len, head_dim = x.size()
        batch_size //= self.num_heads

        return x.reshape(batch_size, self.num_heads, seq_len, head_dim)\
                .transpose(1, 2)\
                .reshape(batch_size, seq_len, self.hidden_dim)


class Shifter(nn.Module):
    active = False

    def __init__(self, hidden_dim=64, num_heads=8):
        super().__init__()

        self.path_length_embedding = nn.Embedding(512, 1)
        self.path_idx_embedding = nn.Embedding(512, hidden_dim)
        self.num_heads = num_heads

    def forward(self, X, mask, path_lengths):
        if not Shifter.active:
            return None

        distance_mat = torch.zeros((path_lengths.size(0), path_lengths.size(1) + 1, 
                                    path_lengths.size(2) + 1)).to(X.device)
        distance_mat[:, 1:, 1:] = path_lengths
        distance_mat[:, 0, :] = 1
        distance_mat[:, :, 0] = 1

        result = self.path_length_embedding(distance_mat.long()).squeeze(-1)
        return result.repeat_interleave(self.num_heads, 0)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim=64, ff_dim=64, attn_num_heads=8, dropout_prob=0.0):
        super(EncoderBlock, self).__init__()

        self.attention = SelfAttentionWithShift(
            hidden_dim=embed_dim, num_heads=attn_num_heads
        )
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(ff_dim, embed_dim),
        )

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)

        self.shifter = Shifter()

    def forward(self, X, mask, path_lengths):
        shift = self.shifter(X, mask, path_lengths)

        X_normed = self.layer_norm_1(X)
        X_attn = self.attention(q=X_normed, k=X_normed, v=X_normed, mask=mask, shift=shift) + X
        X_attn_normed = self.layer_norm_2(X_attn)
        X_ff = self.feedforward(X_attn_normed)
        return X_ff + X_attn


class Dvoryaninormer(nn.Module):
    def __init__(self, node_dim, edge_dim, num_classes, embedding_dim, n_encoders, 
                add_centrality_embedding=False, add_path_length_embedding=False, **encoder_kwargs):
        super(Dvoryaninormer, self).__init__()
        
        self.encoders = nn.ModuleList([EncoderBlock(**encoder_kwargs) for i in range(n_encoders)])
        self.node_embedder = nn.Linear(node_dim, embedding_dim)
        self.edge_embedder = nn.Linear(edge_dim, embedding_dim)
        self.vnode_embedding = nn.Parameter(torch.randn(embedding_dim))
        self.projection = nn.Linear(embedding_dim, num_classes)

        self.add_centrality_embedding = add_centrality_embedding
        self.centrality_embedding = nn.Embedding(512, embedding_dim)

        self.add_path_length_embedding = add_path_length_embedding
        Shifter.active = add_path_length_embedding

        if add_centrality_embedding:
            print("Adding centrality embedding")

        if add_path_length_embedding:
            print("Adding path length embedding")

    def _add_vnode(self, node_features, mask):
        batch_size, n_nodes, embedding_dim = node_features.size()
        vnode_embedding = self.vnode_embedding.repeat(batch_size, 1)
        return torch.cat([vnode_embedding.unsqueeze(1), node_features], dim=1), \
               torch.cat([torch.ones(batch_size, 1).to(mask.device), mask], dim=1)

    def forward(self, node_features, mask, node_centralities=None, path_lengths=None):
        node_features = self.node_embedder(node_features)

        if node_centralities is not None and self.add_centrality_embedding:
            node_features += self.centrality_embedding(node_centralities)

        node_features, mask = self._add_vnode(node_features, mask)

        for encoder in self.encoders:
            node_features = encoder(node_features, mask, path_lengths)

        return self.projection(node_features[:, 0, :])
