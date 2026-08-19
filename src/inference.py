import os
import argparse
import subprocess
import torch
from torch_geometric.data import Data
from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from helpers import format_flatzinc_text, parse_fzn_properties

# Default to latest multi-domain model and tokenizer (matching web_app.py)
DEFAULT_MODEL_PATH = "models/model_tsp_knapsack_graph"
DEFAULT_TOKENIZER_PATH = "models/tokenizer_tsp_knapsack_graph"

@torch.no_grad()
def generate_text_from_graph(
    model, 
    tokenizer, 
    graph, 
    noise_std=0.0, 
    max_new_tokens=2048, 
    max_retries=5,
    temperature=0.7,
    top_p=0.9,
    device="cuda", 
    format_output=True,
    validate_syntax=True
):
    """
    Encodes graph with graph encoder and generates FlatZinc instance code using the same
    iterative retry, sampling, and syntax validation pipeline as web_app.py.
    """
    model.eval()
    model.to(device)
    
    # 1. Prepare graph batching and extract node embeddings
    g = graph.to(device)
    if not hasattr(g, 'batch') or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
    setattr(g, 'num_graphs', 1) 
    
    node_embeddings = model.graph_encoder(g) # Shape: [num_nodes, hidden_dim]
    
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    
    pid = os.getpid()
    os.makedirs(".cache", exist_ok=True)
    
    best_formatted = None
    best_raw = None

    for attempt in range(max_retries):
        # 2. Add Gaussian noise perturbation if requested
        if noise_std > 0.0:
            noise = torch.randn_like(node_embeddings) * noise_std
            noisy_embeddings = node_embeddings + noise
        else:
            noisy_embeddings = node_embeddings

        # 3. Shape features for cross-attention
        encoder_hidden_states = noisy_embeddings.unsqueeze(0) # [1, num_nodes, hidden_dim]
        encoder_attention_mask = torch.ones(
            (1, noisy_embeddings.size(0)), 
            dtype=torch.long, 
            device=device
        ) # [1, num_nodes]

        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

        # 4. Sampling strategy matching web_app (deterministic on attempt 0 unless noise/temp given, sampled on retries)
        do_sample = attempt > 0
        kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "num_beams": 1,
            "do_sample": do_sample,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "use_cache": True
        }
        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        generated_ids = model.decoder.generate(**kwargs)
        token_ids = generated_ids[0].cpu().tolist()
        final_raw_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        formatted_final = format_flatzinc_text(final_raw_text)
        best_formatted = formatted_final
        best_raw = final_raw_text

        # 5. Check if properties are parseable
        nc_gen, nv_gen = parse_fzn_properties(formatted_final)
        if nc_gen is None or nv_gen is None:
            continue

        if not validate_syntax:
            return formatted_final if format_output else final_raw_text

        # 6. Syntax validation with MiniZinc solver
        temp_eval_path = f".cache/cli_eval_{pid}_{attempt}.fzn"
        with open(temp_eval_path, "w") as f:
            f.write(formatted_final)

        eval_solver = "gecode" if ("gecode_" in formatted_final or "circuit" in formatted_final) else "cp-sat"
        sol_proc = subprocess.run(
            ["minizinc", temp_eval_path, "--solver", eval_solver, "-t", "5000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if os.path.exists(temp_eval_path):
            os.remove(temp_eval_path)

        sol_out = sol_proc.stdout.decode()
        sol_err = sol_proc.stderr.decode()
        combined_output = (sol_err + "\n" + sol_out).lower()

        has_syntax_error = (
            sol_proc.returncode != 0 and "=====unsatisfiable=====" not in combined_output and "----------" not in combined_output
        ) or any(
            err_kw in combined_output 
            for err_kw in ["syntax error", "type error", "undefined identifier", "unexpected", "error:"]
        )
        
        if not has_syntax_error:
            return formatted_final if format_output else final_raw_text

    return best_formatted if (best_formatted and format_output) else best_raw

def load_checkpoint(filepath, model, optimizer=None, scaler=None):
    """
    Loads a model checkpoint with fallback search in standard directory locations.
    """
    if not os.path.exists(filepath):
        for alt_path in [os.path.join("models", filepath), os.path.join("../models", filepath)]:
            if os.path.exists(alt_path):
                filepath = alt_path
                break
    checkpoint = torch.load(filepath, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    
    # Load weights and states
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        loss = checkpoint.get('loss', 0.0)
    else:
        model.load_state_dict(checkpoint)
        epoch = 0
        step = 0
        loss = 0.0

    if optimizer is not None and isinstance(checkpoint, dict) and checkpoint.get('optimizer_state_dict') is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    if scaler is not None and isinstance(checkpoint, dict) and checkpoint.get('scaler_state_dict') is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
    return model, optimizer, scaler, epoch, step, loss

def convert_to_pyg_data(graph_dict):
    """Converts a raw graph feature dictionary into a PyTorch Geometric Data instance."""
    x = torch.tensor(graph_dict["x"], dtype=torch.float)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def main():
    parser = argparse.ArgumentParser(description="Graph Node Embedding Inference")
    parser.add_argument("graph", type=str, help="Path to .graph file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help=f"Path to model checkpoint (default: {DEFAULT_MODEL_PATH})")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER_PATH, help=f"Path to tokenizer (default: {DEFAULT_TOKENIZER_PATH})")
    parser.add_argument("--noise", "--noise_std", dest="noise_std", type=float, default=0.0, help="Standard deviation of Gaussian noise added to node embeddings (default: 0.0)")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum new generation tokens (default: 2048)")
    parser.add_argument("--max_retries", type=int, default=5, help="Maximum generation retries for valid syntax (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature on retries (default: 0.7)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling on retries (default: 0.9)")
    parser.add_argument("--output", "-o", type=str, default="out.fzn", help="Output file path for generated FlatZinc (default: out.fzn)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Computation device (cuda or cpu)")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of CPU threads for PyTorch operations (default: 8)")
    args = parser.parse_args()

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)

    print(f"Loading Tokenizer from '{args.tokenizer}'...")
    tok_path = args.tokenizer
    if not os.path.exists(tok_path):
        for alt in [os.path.join("models", tok_path), os.path.join("../models", tok_path)]:
            if os.path.exists(alt):
                tok_path = alt
                break
    tokenizer = train_and_get_tokenizer(tokenizer_path=tok_path)

    print(f"Loading Graph from '{args.graph}'...")
    graph_sample = load_graph_to_features(args.graph)
    node_in_dim = len(graph_sample["x"][0])
    edge_in_dim = len(graph_sample["edge_attr"][0])

    print(f"Initializing Model architecture (node_in_dim={node_in_dim}, edge_in_dim={edge_in_dim})...")
    model = GraphToTextConditionalGeneration(
        node_in_dim, 
        edge_in_dim, 
        10000, 
        tokenizer.token_to_id("[BOS]"), 
        tokenizer.token_to_id("[EOS]"), 
        tokenizer.token_to_id("[PAD]")
    ).to(device)

    print(f"Loading Model Checkpoint from '{args.model}'...")
    model, _, _, epoch, step, loss = load_checkpoint(args.model, model, None, None)
    model.eval()

    g_data = convert_to_pyg_data(graph_sample)

    print(f"Executing Generation (max_new_tokens={args.max_tokens}, noise_std={args.noise_std}, max_retries={args.max_retries}, device={device})...")
    generated_string = generate_text_from_graph(
        model=model, 
        tokenizer=tokenizer, 
        graph=g_data,
        noise_std=args.noise_std,
        max_new_tokens=args.max_tokens,
        max_retries=args.max_retries,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        format_output=True,
        validate_syntax=True
    )

    print(f"\n--- Generated FlatZinc Output ---\n{generated_string}")
    with open(args.output, "w") as f:
        f.write(generated_string)
    print(f"\nGenerated FlatZinc saved to '{args.output}'.")

if __name__ == '__main__':
    main()
