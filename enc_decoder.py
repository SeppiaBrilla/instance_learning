import os
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# ==========================================
# 1. TOKENIZER TRAINING SETUP
# ==========================================
def train_and_get_tokenizer(vocab_size=10000) -> Tokenizer:
    # Dummy corpus file for illustration

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )
    corpus_folder = "flat/"
    tokenizer.train([os.path.join(corpus_folder, file) for file in os.listdir(corpus_folder)], trainer)
    return tokenizer

# ==========================================
# 2. ARCHITECTURE DEFINITIONS
# ==========================================

class GNNEncoder(nn.Module):
    """
    ~40M Parameter GNN Encoder using deep GIN layers.
    Maps a PyG Graph to sequence-like node embeddings.
    """
    def __init__(self, in_dim=128, hidden_dim=1024, num_layers=7):
        super().__init__()
        self.convs = nn.ModuleList()
        
        # First layer
        mlp1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(mlp1))
        
        # Hidden layers to scale up parameter count (~5.8M parameters per layer)
        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            
    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.relu(x)
        return x # Shape: [num_nodes, hidden_dim]


class TextDecoder(nn.Module):
    """
    ~60M Parameter Causal Transformer Decoder with Cross-Attention.
    """
    def __init__(self, vocab_size=8000, d_model=1024, nhead=8, num_layers=4, dim_feedforward=4096):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt, memory, tgt_mask=None):
        # tgt shape: [batch_size, tgt_len]
        # memory shape: [batch_size, num_nodes, d_model]
        tgt_emb = self.embedding(tgt) # [batch_size, tgt_len, d_model]
        
        out = self.transformer_decoder(
            tgt=tgt_emb, 
            memory=memory, 
            tgt_mask=tgt_mask
        )
        return self.fc_out(out)


class GraphToTextModel(nn.Module):
    """
    Combined Encoder-Decoder Pipeline.
    """
    def __init__(self, vocab_size=8000, d_model=2048):
        super().__init__()
        self.encoder = GNNEncoder(in_dim=128, hidden_dim=d_model)
        self.decoder = TextDecoder(vocab_size=vocab_size, d_model=d_model)
        
    def forward(self, x, edge_index, tgt, batch_indices=None, tgt_mask=None):
        # 1. Encode graph nodes
        node_embeddings = self.encoder(x, edge_index)
        
        # 2. Reshape/batch node embeddings for text cross-attention
        # For a single graph, unsqueeze to add batch dimension [1, num_nodes, d_model]
        if batch_indices is None:
            memory = node_embeddings.unsqueeze(0)
        else:
            # Multi-graph batch unpacking logic would go here if using PyG DataLoader
            pass
            
        # 3. Decode text tokens conditioned on graph memory
        logits = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return logits

# ==========================================
# 3. EXECUTION PIPELINE
# ==========================================

# Train Tokenizer
vocab_size = 10000
tokenizer = None
tokenizer_path = "./tokenizer"
if os.path.exists(tokenizer_path):
    tokenizer = Tokenizer.from_file(tokenizer_path)
else:
    tokenizer = train_and_get_tokenizer(vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)

print(tokenizer.encode("constraint int_lin_ne(X_INTRODUCED_16_,[X_INTRODUCED_0_,X_INTRODUCED_1_],0);").ids)

# Create Mock NetworkX Graph & Convert to PyG
nx_graph = nx.erdos_renyi_graph(n=50, p=0.1)
edges = torch.tensor(list(nx_graph.edges), dtype=torch.long).t().contiguous()
edge_index = torch.cat([edges, edges[[1, 0]]], dim=1) # undirected boundary
x = torch.randn(50, 128) # 50 nodes, 128-dim initial features

# Tokenize target text for the decoder
target_text = "[BOS] Graph neural networks encode structural topology. [EOS]"
encoded_text = tokenizer.encode(target_text)
tgt_tokens = torch.tensor([encoded_text.ids], dtype=torch.long) # Shape: [1, seq_len]

# Generate Causal Mask for Decoder
seq_len = tgt_tokens.size(1)
tgt_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)

# Initialize Model
model = GraphToTextModel(vocab_size=vocab_size, d_model=1024)

# Calculate Parameters
enc_params = sum(p.numel() for p in model.encoder.parameters())
dec_params = sum(p.numel() for p in model.decoder.parameters())

print(f"Encoder Parameters: {enc_params / 1e6:.2f}M")
print(f"Decoder Parameters: {dec_params / 1e6:.2f}M")

# Forward Pass
logits = model(x, edge_index, tgt_tokens, tgt_mask=tgt_mask)
print(f"Output logits shape: {logits.shape}")  # [Batch, Seq_len, Vocab_size]
