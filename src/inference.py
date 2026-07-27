import os
import torch
from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from torch_geometric.data import Data

@torch.no_grad()
def generate_text_from_graph(model, tokenizer, graph1, graph2=None, max_new_tokens=50, device="cuda"):
    """
    Given two PyG graphs (or a single graph if graph2 is None), encodes them,
    combines their node encoder embeddings (which acts as random node-level feature noise perturbation
    because node ordering across different graphs is arbitrary and unaligned), and generates text output.
    """
    model.eval()
    model.to(device)
    
    # 1. Prepare graph batching and extract features for graph1
    graph1 = graph1.to(device)
    if not hasattr(graph1, 'batch') or graph1.batch is None:
        graph1.batch = torch.zeros(graph1.num_nodes, dtype=torch.long, device=device)
    setattr(graph1, 'num_graphs', 1) 
    
    node_features1 = model.graph_encoder(graph1) # Shape: [num_nodes_1, hidden_dim]
    
    # 2. Extract features for graph2 if provided and combine node embeddings (acting as node feature noise)
    if graph2 is not None:
        graph2 = graph2.to(device)
        if not hasattr(graph2, 'batch') or graph2.batch is None:
            graph2.batch = torch.zeros(graph2.num_nodes, dtype=torch.long, device=device)
        setattr(graph2, 'num_graphs', 1) 

        node_features2 = model.graph_encoder(graph2) # Shape: [num_nodes_2, hidden_dim]

        if node_features1.shape == node_features2.shape:
            mean_node_features = (node_features1 + node_features2) / 2.0
        else:
            max_nodes = max(node_features1.size(0), node_features2.size(0))
            hidden_dim = node_features1.size(1)
            
            padded_1 = torch.zeros((max_nodes, hidden_dim), device=device, dtype=node_features1.dtype)
            padded_1[:node_features1.size(0)] = node_features1
            
            padded_2 = torch.zeros((max_nodes, hidden_dim), device=device, dtype=node_features2.dtype)
            padded_2[:node_features2.size(0)] = node_features2
            
            counts = torch.zeros((max_nodes, 1), device=device, dtype=node_features1.dtype)
            counts[:node_features1.size(0)] += 1.0
            counts[:node_features2.size(0)] += 1.0
            
            mean_node_features = (padded_1 + padded_2) / counts
    else:
        mean_node_features = node_features1
    
    # 3. Shape features for cross-attention format: [Batch, Seq_Len, Hidden_Dim]
    encoder_hidden_states = mean_node_features.unsqueeze(0) # [1, num_nodes, hidden_dim]
    encoder_attention_mask = torch.ones(
        (1, mean_node_features.size(0)), 
        dtype=torch.long, 
        device=device
    ) # [1, num_nodes]

    # 4. Define starting token for generation
    bos_id = tokenizer.token_to_id("[BOS]")
    input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    # 5. Execute Auto-Regressive Generation
    generated_ids = model.decoder.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        eos_token_id=tokenizer.token_to_id("[EOS]"),
        num_beams=1,             # Greedy search
        do_sample=False,         # Deterministic decoding
        
        # Cross-attention context tensors passed here
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask
    )

    # 6. Decode output token IDs to clean string
    token_ids = generated_ids[0].cpu().tolist()
    output_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return output_text

def load_checkpoint(filepath, model, optimizer, scaler=None):
    if not os.path.exists(filepath):
        for alt_path in [os.path.join("models", filepath), os.path.join("../models", filepath)]:
            if os.path.exists(alt_path):
                filepath = alt_path
                break
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    
    # Load weights and states
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if scaler is not None and checkpoint['scaler_state_dict'] is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    loss = checkpoint['loss']
    
    return model, optimizer, scaler, epoch, step, loss

def convert_to_pyg_data(graph_dict):
    # Ensure matrices are explicitly converted to torch Tensors
    x = torch.tensor(graph_dict["x"], dtype=torch.float)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float)
    
    # Instantiate the PyG Data container
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <graph1_path> [graph2_path]")
        sys.exit(1)

    graph_path_1 = sys.argv[1]
    graph_path_2 = sys.argv[2] if len(sys.argv) > 2 else None

    tokenizer = train_and_get_tokenizer()
    graph_sample_1 = load_graph_to_features(graph_path_1)
    graph_sample_2 = load_graph_to_features(graph_path_2) if graph_path_2 else None

    node_in_dim = len(graph_sample_1["x"][0])
    edge_in_dim = len(graph_sample_1["edge_attr"][0])

    model = GraphToTextConditionalGeneration(
        node_in_dim, 
        edge_in_dim, 
        10000, 
        tokenizer.token_to_id("[BOS]"), 
        tokenizer.token_to_id("[EOS]"), 
        tokenizer.token_to_id("[PAD]")
    )
    checkpoint = load_checkpoint("final_model_80m.pt", model, None, None)
    model = checkpoint[0]

    g1_data = convert_to_pyg_data(graph_sample_1)
    g2_data = convert_to_pyg_data(graph_sample_2) if graph_sample_2 else None

    generated_string = generate_text_from_graph(
        model=model, 
        tokenizer=tokenizer, 
        graph1=g1_data,
        graph2=g2_data, 
        max_new_tokens=2048,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Generated Output: {generated_string}")
    with open("out.fzn", "w") as f:
        f.write(generated_string)
