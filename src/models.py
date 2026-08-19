import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import BPE
import os

class GraphEncoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=256, out_dim=768):
        super().__init__()
        self.node_linear = nn.Linear(node_in_dim, hidden_dim)
        self.edge_linear = nn.Linear(edge_in_dim, hidden_dim)

        self.conv1 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv2 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv3 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)))
        self.conv4 = GINEConv(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)))
        
        # Project graph node embeddings to match the textual decoder hidden dimensions
        self.output_projection = nn.Linear(hidden_dim, out_dim)

    def forward(self, batch):
        x = F.gelu(self.node_linear(batch.x.float()))
        edge_attr = F.gelu(self.edge_linear(batch.edge_attr.float()))

        x = F.gelu(self.conv1(x, batch.edge_index, edge_attr))
        x = F.gelu(self.conv2(x, batch.edge_index, edge_attr))
        x = F.gelu(self.conv3(x, batch.edge_index, edge_attr))
        x = F.gelu(self.conv4(x, batch.edge_index, edge_attr))
        
        node_embeddings = F.tanh(self.output_projection(x))
        return node_embeddings


class GraphToTextConditionalGeneration(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, vocab_size, bos_token_id, eos_token_id, pad_token_id):
        super().__init__()
        # 1. Initialize custom small GPT configuration (~100M parameters)
        self.decoder_config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=2096,
            n_embd=640,          # Hidden dimensionality (640 / 10 = 64 head dim)
            n_layer=10,          # 10 decoder layers
            n_head=10,           # 10 attention heads
            is_decoder=True,
            add_cross_attention=True,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
        # Initialize randomly without downloading pretrained weights
        self.decoder = GPT2LMHeadModel(config=self.decoder_config)
        
        # 2. Initialize Graph Encoder
        decoder_hidden_dim = self.decoder_config.hidden_size
        self.graph_encoder = GraphEncoder(node_in_dim, edge_in_dim, hidden_dim=1024, out_dim=decoder_hidden_dim)

    def forward(self, graph_batch, decoder_input_ids, decoder_attention_mask=None, labels=None):
        # Extract per-node continuous representations
        node_features = self.graph_encoder(graph_batch)
        
        # Reconstruct the batch dimension for cross-attention
        batch_size = graph_batch.num_graphs
        device = node_features.device
        
        # Reconstruct padded tensor sequence manually for transformer encoder context input
        encoder_hidden_states = []
        encoder_attention_mask = []
        
        for batch_idx in range(batch_size):
            node_mask = (graph_batch.batch == batch_idx)
            current_nodes = node_features[node_mask]
            
            encoder_hidden_states.append(current_nodes)
            encoder_attention_mask.append(torch.ones(current_nodes.size(0), device=device))
            
        # Standardize sizes using padding
        encoder_hidden_states = nn.utils.rnn.pad_sequence(encoder_hidden_states, batch_first=True)
        encoder_attention_mask = nn.utils.rnn.pad_sequence(encoder_attention_mask, batch_first=True, padding_value=0)

        # Pass through Decoder with explicit Cross-Attention context injection
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels
        )
        return outputs

def train_and_get_tokenizer(vocab_size=10000, corpus_folders=None, tokenizer_path="./tokenizer", force_retrain=False) -> Tokenizer:
    if os.path.exists(tokenizer_path) and not force_retrain:
        return Tokenizer.from_file(tokenizer_path)

    if corpus_folders is None:
        corpus_folders = ["data/graph_coloring_flat", "data/knapsack_flat", "data/tsp_flat"]

    training_files = []
    for folder in corpus_folders:
        actual_folder = folder
        if not os.path.exists(actual_folder) and os.path.exists(os.path.join("data", folder)):
            actual_folder = os.path.join("data", folder)
        if os.path.exists(actual_folder):
            for file in os.listdir(actual_folder):
                if file.endswith(".fzn"):
                    training_files.append(os.path.join(actual_folder, file))

    if not training_files:
        raise ValueError(f"No .fzn files found in corpus folders: {corpus_folders}")

    # Initialize a BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size, 
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"], 
    )
    tokenizer.train(training_files, trainer)
    tokenizer.save(tokenizer_path)
    return tokenizer