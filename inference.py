import torch
from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from torch_geometric.data import Data

@torch.no_grad()
def generate_text_from_graph(model, tokenizer, graph, max_new_tokens=50, device="cuda"):
    """
    Given a single PyG graph, encodes it and generates text output.
    """
    model.eval()
    model.to(device)
    
    # 1. Prepare graph batching for a single graph
    # PyG models expect data batch assignments; simulate a batch size of 1
    graph = graph.to(device)
    if not hasattr(graph, 'batch') or graph.batch is None:
        graph.batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=device)
    setattr(graph, 'num_graphs', 1) 
    
    # 2. Extract graph features via the encoder
    node_features = model.graph_encoder(graph) # Shape: [num_nodes, hidden_dim]
    
    # 3. Shape features for cross-attention format: [Batch, Seq_Len, Hidden_Dim]
    encoder_hidden_states = node_features.unsqueeze(0) # [1, num_nodes, hidden_dim]
    encoder_attention_mask = torch.ones(
        (1, node_features.size(0)), 
        dtype=torch.long, 
        device=device
    ) # [1, num_nodes]

    # 4. Define starting token for generation (BOS or EOS depending on your tokenizer setup)
    # If your tokenizer doesn't have a bos_token, fallback to eos_token
    bos_id = tokenizer.token_to_id("[BOS]")
    input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    # 5. Execute Auto-Regressive Generation
    # We pass the cross-attention tensors into keyword arguments
    generated_ids = model.decoder.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.token_to_id("[PAD]"),
        eos_token_id=tokenizer.token_to_id("[EOS]"),
        num_beams=1,             # Greedy search; change to >1 for beam search
        do_sample=False,         # Deterministic decoding
        
        # Cross-attention context tensors passed here
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask
    )

    # 6. Decode output token IDs to clean string
    # Convert the PyTorch tensor row into a standard Python list of integers
    token_ids = generated_ids[0].cpu().tolist()

    # Decode using your raw tokenizer object
    output_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return output_text

def load_checkpoint(filepath, model, optimizer, scaler=None):
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    
    # Load weights and states
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
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

    # Assuming your model, tokenizer, and a test graph are initialized:
    # graph_sample = your_dataset[0]["graph"]
    import sys
    graph_path = sys.argv[1]
    tokenizer = train_and_get_tokenizer()
    graph_sample = load_graph_to_features(graph_path)

    node_in_dim = len(graph_sample["x"][0])
    edge_in_dim = len(graph_sample["edge_attr"][0])

    model = GraphToTextConditionalGeneration(node_in_dim, 
                                             edge_in_dim, 
                                             10000, 
                                             tokenizer.token_to_id("[BOS]"), 
                                             tokenizer.token_to_id("[EOS]"), 
                                             tokenizer.token_to_id("[PAD]")
                                             )
    checkpoint = load_checkpoint("checkpoint_9.pt", model, None, None)
    model = checkpoint[0]
    generated_string = generate_text_from_graph(
        model=model, 
        tokenizer=tokenizer, 
        graph=convert_to_pyg_data(graph_sample), 
        max_new_tokens=64,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Generated Output: {generated_string}")
    with open("out.fzn", "w") as f:
        f.write(generated_string)
