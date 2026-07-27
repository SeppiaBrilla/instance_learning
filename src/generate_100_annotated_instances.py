import os
import re
import sys
import torch
import random
import subprocess
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure minizinc binary is in PATH
os.environ['PATH'] = '/work/minizinc_bundle/bin:' + os.environ.get('PATH', '')

from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from inference import load_checkpoint, convert_to_pyg_data

NUM_VERTICES_LIST = [10, 12, 14, 16, 18, 20]
EDGE_PROB_LIST = [0.2, 0.4, 0.6, 0.8]
NUM_COLORS_LIST = [3, 5, 7, 9, 11, 13, 15]

def normalize_text(text):
    return " ".join(text.split())

def format_flatzinc_text(text):
    lines = []
    statements = text.split(";")
    for stmt in statements:
        s = stmt.strip()
        if s:
            lines.append(s + ";")
    return "\n".join(lines) + "\n"

def parse_fzn_properties(fzn_text):
    nc_match = re.search(r"var\s+1\s*\.\.\s*(\d+)\s*:", fzn_text)
    nc_gen = int(nc_match.group(1)) if nc_match else None
    
    nv_match = re.search(r"array\s*\[\s*1\s*\.\.\s*(\d+)\s*\]\s*of\s*var\s*int\s*:\s*color", fzn_text)
    if not nv_match:
        all_indices = [int(m) for m in re.findall(r"X_INTRODUCED_(\d+)_", fzn_text)]
        nv_gen = max(all_indices) + 1 if all_indices else None
    else:
        nv_gen = int(nv_match.group(1))
        
    return nc_gen, nv_gen

def generate_coloring_instance(num_vertices, edge_probability, num_colors, filename="instance.dzn"):
    import networkx as nx
    graph = nx.erdos_renyi_graph(n=num_vertices, p=edge_probability)
    edges = [(u + 1, v + 1) for u, v in graph.edges()]
    num_edges = len(edges)
    if num_edges > 0:
        edge_strings = [f" {u}, {v} " for u, v in edges]
        minizinc_edges = "| " + " | ".join(edge_strings) + " |"
    else:
        minizinc_edges = ""
    with open(filename, "w") as f:
        f.write(f"% Generated Graph Coloring Instance\n")
        f.write(f"nc = {num_colors};\n")
        f.write(f"nv = {num_vertices};\n")
        f.write(f"ne = {num_edges};\n\n")
        f.write(f"edges = [{minizinc_edges}];\n")

def generate_source_graph_pair(pair_idx, out_dir="tmp_source_instances"):
    rng = random.Random(pair_idx + 123456)
    
    # Graph A
    nv_a, p_a, nc_a = rng.choice(NUM_VERTICES_LIST), rng.choice(EDGE_PROB_LIST), rng.choice(NUM_COLORS_LIST)
    dzn_a = os.path.join(out_dir, f"pair_{pair_idx}_a.dzn")
    fzn_a = os.path.join(".cache", f"pair_{pair_idx}_a.fzn")
    graph_a = os.path.join(out_dir, f"pair_{pair_idx}_a.graph")

    generate_coloring_instance(nv_a, p_a, nc_a, filename=dzn_a)
    cp_res_a = subprocess.run(f"minizinc model.mzn {dzn_a} --solver cp-sat", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
    sat_a = "=====UNSATISFIABLE=====" not in cp_res_a

    subprocess.run(f"minizinc model.mzn {dzn_a} --solver gecode -c --no-output-ozn --fzn {fzn_a}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(f"python /work/flatzinc_parser/flatzinc_parser.py {fzn_a} {graph_a}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(fzn_a): os.remove(fzn_a)
    if os.path.exists(dzn_a): os.remove(dzn_a)

    # Graph B
    nv_b, p_b, nc_b = rng.choice(NUM_VERTICES_LIST), rng.choice(EDGE_PROB_LIST), rng.choice(NUM_COLORS_LIST)
    dzn_b = os.path.join(out_dir, f"pair_{pair_idx}_b.dzn")
    fzn_b = os.path.join(".cache", f"pair_{pair_idx}_b.fzn")
    graph_b = os.path.join(out_dir, f"pair_{pair_idx}_b.graph")

    generate_coloring_instance(nv_b, p_b, nc_b, filename=dzn_b)
    cp_res_b = subprocess.run(f"minizinc model.mzn {dzn_b} --solver cp-sat", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
    sat_b = "=====UNSATISFIABLE=====" not in cp_res_b

    subprocess.run(f"minizinc model.mzn {dzn_b} --solver gecode -c --no-output-ozn --fzn {fzn_b}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(f"python /work/flatzinc_parser/flatzinc_parser.py {fzn_b} {graph_b}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if os.path.exists(fzn_b): os.remove(fzn_b)
    if os.path.exists(dzn_b): os.remove(dzn_b)

    meta_a = {"nc": nc_a, "nv": nv_a, "sat": sat_a}
    meta_b = {"nc": nc_b, "nv": nv_b, "sat": sat_b}

    return graph_a, graph_b, meta_a, meta_b

@torch.no_grad()
def batch_generate_from_graph_pairs(model, tokenizer, graph_pairs, max_new_tokens=2048, device="cuda"):
    """
    Encodes graph pairs and combines their node embeddings. Because node ordering across different graphs
    is arbitrary and unaligned, combining node embeddings acts as random node feature noise perturbation
    on sequence nodes rather than aligned latent space interpolation.
    """
    B = len(graph_pairs)
    mean_embeddings_list = []
    num_nodes_list = []

    for gA, gB in graph_pairs:
        gA = gA.to(device)
        gB = gB.to(device)
        if not hasattr(gA, 'batch') or gA.batch is None:
            gA.batch = torch.zeros(gA.num_nodes, dtype=torch.long, device=device)
        if not hasattr(gB, 'batch') or gB.batch is None:
            gB.batch = torch.zeros(gB.num_nodes, dtype=torch.long, device=device)
        setattr(gA, 'num_graphs', 1)
        setattr(gB, 'num_graphs', 1)

        embA = model.graph_encoder(gA)
        embB = model.graph_encoder(gB)

        N_A, D = embA.shape
        N_B = embB.shape[0]
        max_N = max(N_A, N_B)

        padA = torch.zeros((max_N, D), device=device)
        padB = torch.zeros((max_N, D), device=device)
        padA[:N_A] = embA
        padB[:N_B] = embB

        counts = torch.zeros((max_N, 1), device=device)
        counts[:N_A] += 1
        counts[:N_B] += 1
        counts = torch.clamp(counts, min=1)

        mean_emb = (padA + padB) / counts
        mean_embeddings_list.append(mean_emb)
        num_nodes_list.append(max_N)

    max_nodes_batch = max(num_nodes_list)

    encoder_hidden_states = torch.zeros((B, max_nodes_batch, D), device=device)
    encoder_attention_mask = torch.zeros((B, max_nodes_batch), dtype=torch.long, device=device)

    for i in range(B):
        n_nodes = num_nodes_list[i]
        encoder_hidden_states[i, :n_nodes] = mean_embeddings_list[i]
        encoder_attention_mask[i, :n_nodes] = 1

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

    generated_ids = model.decoder.generate(
        input_ids=input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        num_beams=1,
        do_sample=False
    )

    results = []
    for i in range(B):
        ids = generated_ids[i].cpu().tolist()
        text = tokenizer.decode(ids, skip_special_tokens=True)
        results.append(text)

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading training dataset instances for uniqueness lookup...")
    existing_instances = set()
    flat_dir = "data/flat" if os.path.exists("data/flat") else "flat"
    for file in tqdm(os.listdir(flat_dir)):
        if file.endswith(".fzn"):
            with open(os.path.join(flat_dir, file), "r") as f:
                existing_instances.add(normalize_text(f.read()))

    out_dir = "data/annotated_100_instances" if os.path.exists("data/annotated_100_instances") else "annotated_100_instances"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(".cache", exist_ok=True)
    os.makedirs("tmp_source_instances", exist_ok=True)

    tokenizer = train_and_get_tokenizer()
    sample_path = "data/graphs/instance_100.graph" if os.path.exists("data/graphs/instance_100.graph") else "graphs/instance_100.graph"
    sample_g = load_graph_to_features(sample_path)
    node_in_dim = len(sample_g["x"][0])
    edge_in_dim = len(sample_g["edge_attr"][0])

    model = GraphToTextConditionalGeneration(
        node_in_dim, edge_in_dim, 10000,
        tokenizer.token_to_id("[BOS]"),
        tokenizer.token_to_id("[EOS]"),
        tokenizer.token_to_id("[PAD]")
    ).to(device)
    model_path = "models/final_model_80m.pt" if os.path.exists("models/final_model_80m.pt") else "final_model_80m.pt"
    model, _, _, _, _, _ = load_checkpoint(model_path, model, None, None)
    model.eval()

    generated_set = set()
    records = []
    target_count = 100
    generated_count = 0
    batch_size = 4
    pair_counter = 0

    pbar = tqdm(total=target_count, desc="Generating Annotated Mean Instances")

    while generated_count < target_count:
        needed = (target_count - generated_count) * 2
        chunk_size = max(20, needed)

        pairs_info = []
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(generate_source_graph_pair, pair_counter + idx) for idx in range(chunk_size)]
            pair_counter += chunk_size
            for f in as_completed(futures):
                g_a, g_b, meta_a, meta_b = f.result()
                if os.path.exists(g_a) and os.path.exists(g_b):
                    pairs_info.append((g_a, g_b, meta_a, meta_b))

        for i in range(0, len(pairs_info), batch_size):
            if generated_count >= target_count:
                break

            batch_items = pairs_info[i:i+batch_size]
            graph_pairs_pyg = []
            valid_items = []

            for g_a, g_b, meta_a, meta_b in batch_items:
                try:
                    gA_dict = load_graph_to_features(g_a)
                    gB_dict = load_graph_to_features(g_b)
                    gA_pyg = convert_to_pyg_data(gA_dict)
                    gB_pyg = convert_to_pyg_data(gB_dict)
                    graph_pairs_pyg.append((gA_pyg, gB_pyg))
                    valid_items.append((g_a, g_b, meta_a, meta_b))
                except Exception:
                    continue

            if not graph_pairs_pyg:
                continue

            decoded_texts = batch_generate_from_graph_pairs(
                model=model,
                tokenizer=tokenizer,
                graph_pairs=graph_pairs_pyg,
                max_new_tokens=2048,
                device=device
            )

            for idx, text in enumerate(decoded_texts):
                if generated_count >= target_count:
                    break

                raw_gen_text = text.strip()
                norm_gen = normalize_text(raw_gen_text)

                if norm_gen in existing_instances or norm_gen in generated_set:
                    continue

                g_a, g_b, meta_a, meta_b = valid_items[idx]
                nc_gen, nv_gen = parse_fzn_properties(raw_gen_text)

                # Format FlatZinc with linebreaks
                formatted_body = format_flatzinc_text(raw_gen_text)

                # Solve generated instance with MiniZinc CP-SAT solver
                temp_gen_path = f".cache/eval_{generated_count}.fzn"
                with open(temp_gen_path, "w") as f:
                    f.write(formatted_body)

                sol_res = subprocess.run(
                    [f"minizinc {temp_gen_path} --solver cp-sat -t 5000"],
                    shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ).stdout.decode()
                sat_gen = "=====UNSATISFIABLE=====" not in sol_res

                if os.path.exists(temp_gen_path):
                    os.remove(temp_gen_path)

                header_comments = (
                    f"% ========================================================\n"
                    f"% DUAL-GRAPH GENERATION METADATA\n"
                    f"% Source Graph A: nc = {meta_a['nc']}, nv = {meta_a['nv']}, sat = {str(meta_a['sat']).lower()}\n"
                    f"% Source Graph B: nc = {meta_b['nc']}, nv = {meta_b['nv']}, sat = {str(meta_b['sat']).lower()}\n"
                    f"% Source Averages: nc_mean = {(meta_a['nc'] + meta_b['nc'])/2:.1f}, nv_mean = {(meta_a['nv'] + meta_b['nv'])/2:.1f}\n"
                    f"% Generated Instance: nc = {nc_gen}, nv = {nv_gen}, sat = {str(sat_gen).lower()}\n"
                    f"% ========================================================\n\n"
                )

                full_annotated_text = header_comments + formatted_body
                out_filename = os.path.join(out_dir, f"instance_{generated_count}.fzn")
                with open(out_filename, "w") as f:
                    f.write(full_annotated_text)

                generated_set.add(norm_gen)

                rec = {
                    "instance_id": generated_count,
                    "nc_a": meta_a["nc"],
                    "nv_a": meta_a["nv"],
                    "sat_a": meta_a["sat"],
                    "nc_b": meta_b["nc"],
                    "nv_b": meta_b["nv"],
                    "sat_b": meta_b["sat"],
                    "nc_mean": (meta_a["nc"] + meta_b["nc"]) / 2.0,
                    "nv_mean": (meta_a["nv"] + meta_b["nv"]) / 2.0,
                    "nc_max": max(meta_a["nc"], meta_b["nc"]),
                    "nv_max": max(meta_a["nv"], meta_b["nv"]),
                    "nc_gen": nc_gen,
                    "nv_gen": nv_gen,
                    "sat_gen": sat_gen
                }
                records.append(rec)

                generated_count += 1
                pbar.update(1)

            for g_a, g_b, _, _ in valid_items:
                if os.path.exists(g_a): os.remove(g_a)
                if os.path.exists(g_b): os.remove(g_b)

    pbar.close()

    df = pd.DataFrame(records)
    csv_out = "data/outputs/generation_pattern_analysis.csv" if os.path.exists("data/outputs") else "generation_pattern_analysis.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nGeneration pattern data saved to '{csv_out}'.")

    # PATTERN ANALYSIS & STATISTICAL METRICS
    print("\n=======================================================")
    print("        GENERATION PATTERN ANALYSIS RESULTS           ")
    print("=======================================================")

    print("\n--- 1. NUMBER OF COLORS (nc) ---")
    corr_nc_mean = df["nc_gen"].corr(df["nc_mean"])
    corr_nc_max = df["nc_gen"].corr(df["nc_max"])
    print(f"Pearson Correlation (nc_gen vs nc_mean): {corr_nc_mean:.4f}")
    print(f"Pearson Correlation (nc_gen vs nc_max):  {corr_nc_max:.4f}")
    df["nc_diff_mean"] = df["nc_gen"] - df["nc_mean"]
    df["nc_diff_max"] = df["nc_gen"] - df["nc_max"]
    print(f"Mean Difference (nc_gen - nc_mean): {df['nc_diff_mean'].mean():.2f} (std: {df['nc_diff_mean'].std():.2f})")
    print(f"Mean Difference (nc_gen - nc_max):  {df['nc_diff_max'].mean():.2f} (std: {df['nc_diff_max'].std():.2f})")

    print("\n--- 2. NUMBER OF VERTICES (nv) ---")
    corr_nv_mean = df["nv_gen"].corr(df["nv_mean"])
    corr_nv_max = df["nv_gen"].corr(df["nv_max"])
    print(f"Pearson Correlation (nv_gen vs nv_mean): {corr_nv_mean:.4f}")
    print(f"Pearson Correlation (nv_gen vs nv_max):  {corr_nv_max:.4f}")
    df["nv_diff_mean"] = df["nv_gen"] - df["nv_mean"]
    df["nv_diff_max"] = df["nv_gen"] - df["nv_max"]
    print(f"Mean Difference (nv_gen - nv_mean): {df['nv_diff_mean'].mean():.2f} (std: {df['nv_diff_mean'].std():.2f})")
    print(f"Mean Difference (nv_gen - nv_max):  {df['nv_diff_max'].mean():.2f} (std: {df['nv_diff_max'].std():.2f})")

    print("\n--- 3. SATISFIABILITY TRANSITIONS ---")
    df["sat_source_type"] = "MIXED"
    df.loc[(df["sat_a"] == True) & (df["sat_b"] == True), "sat_source_type"] = "BOTH_SAT"
    df.loc[(df["sat_a"] == False) & (df["sat_b"] == False), "sat_source_type"] = "BOTH_UNSAT"

    for st in ["BOTH_SAT", "BOTH_UNSAT", "MIXED"]:
        sub = df[df["sat_source_type"] == st]
        if len(sub) > 0:
            sat_rate = sub["sat_gen"].mean() * 100
            print(f"Source Type: {st:10s} (N={len(sub):2d}) -> Generated SAT Rate: {sat_rate:.1f}%")

if __name__ == "__main__":
    main()
