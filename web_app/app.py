import os
import re
import sys
import time
import random
import subprocess
import numpy as np
import pandas as pd
import torch

# Ensure repository root and src/ are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

# Ensure minizinc binary is in PATH
os.environ['PATH'] = '/work/minizinc_bundle/bin:' + os.environ.get('PATH', '')

from flask import Flask, render_template, request, jsonify
from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from inference import load_checkpoint, convert_to_pyg_data
from helpers import (
    load_training_graph_metadata,
    normalize_text,
    format_flatzinc_text,
    parse_fzn_properties,
    check_model_duplication,
    generate_and_save_lda_csv
)

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

ALL_NODE_TYPES = [
    'mult_node', 'literal_node', 'inequality_node', 'lin_sum_node',
    'par_node', 'var_node', 'equality_node', 'leq_node', 'geq_node',
    'circuit_node', 'int_element_node'
]

# Global state and domain configurations
DOMAINS = {
    "coloring": {
        "name": "Graph Coloring",
        "model_path": "models/model_tsp_knapsack_graph",
        "tokenizer_path": "models/tokenizer_tsp_knapsack_graph",
        "graph_dir": "data/graph_coloring_graphs" if os.path.exists("data/graph_coloring_graphs") else "data/graphs",
        "flat_dir": "data/graph_coloring_flat" if os.path.exists("data/graph_coloring_flat") else "data/flat",
        "lda_csv": "data/outputs/lda_2d_sat_coordinates.csv",
        "node_types": ALL_NODE_TYPES,
        "node_in_dim": 14,
        "nv_label": "Vertices (NV)",
        "nc_label": "Colors (NC)",
        "default_solver": "cp-sat"
    },
    "knapsack": {
        "name": "Knapsack",
        "model_path": "models/model_tsp_knapsack_graph",
        "tokenizer_path": "models/tokenizer_tsp_knapsack_graph",
        "graph_dir": "data/knapsack_graphs",
        "flat_dir": "data/knapsack_flat",
        "lda_csv": "data/outputs/lda_2d_knapsack_coordinates.csv",
        "node_types": ALL_NODE_TYPES,
        "node_in_dim": 14,
        "nv_label": "Items (NV)",
        "nc_label": "Capacity (Cap)",
        "default_solver": "cp-sat"
    },
    "tsp": {
        "name": "TSP (Traveling Salesperson)",
        "model_path": "models/model_tsp_knapsack_graph",
        "tokenizer_path": "models/tokenizer_tsp_knapsack_graph",
        "graph_dir": "data/tsp_graphs",
        "flat_dir": "data/tsp_flat",
        "lda_csv": "data/outputs/lda_2d_tsp_coordinates.csv",
        "node_types": ALL_NODE_TYPES,
        "node_in_dim": 14,
        "nv_label": "Cities (N)",
        "nc_label": "Max Distance (Cost)",
        "default_solver": "gecode"
    }
}

CURRENT_DOMAIN = "coloring"
MODEL = None
TOKENIZER = None
DEVICE = None
TRAINING_INSTANCES = []
INSTANCE_TREE_DATA = []
TRAINING_NORMALIZED_MAP = {}

def init_app(domain_name="coloring"):
    global MODEL, TOKENIZER, DEVICE, TRAINING_INSTANCES, INSTANCE_TREE_DATA, TRAINING_NORMALIZED_MAP, CURRENT_DOMAIN

    if domain_name not in DOMAINS:
        domain_name = "coloring"

    CURRENT_DOMAIN = domain_name
    config = DOMAINS[domain_name]

    torch.set_num_threads(8)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WebApp] Initializing domain '{domain_name}' on device: {DEVICE} with {torch.get_num_threads()} CPU threads")

    import load_graph
    load_graph.node_types = config["node_types"]

    flat_dir = config["flat_dir"]
    TRAINING_NORMALIZED_MAP = {}
    if os.path.exists(flat_dir):
        t0_flat = time.time()
        for file in os.listdir(flat_dir):
            if file.endswith(".fzn"):
                with open(os.path.join(flat_dir, file), "r") as f:
                    TRAINING_NORMALIZED_MAP[file] = normalize_text(f.read())
        print(f"[WebApp] Indexed {len(TRAINING_NORMALIZED_MAP)} training FlatZinc files from {flat_dir} in {time.time()-t0_flat:.2f}s.")

    TOKENIZER = train_and_get_tokenizer(tokenizer_path=config["tokenizer_path"])

    MODEL = GraphToTextConditionalGeneration(
        config["node_in_dim"], 1, 10000,
        TOKENIZER.token_to_id("[BOS]"),
        TOKENIZER.token_to_id("[EOS]"),
        TOKENIZER.token_to_id("[PAD]")
    ).to(DEVICE)
    MODEL, _, _, _, _, _ = load_checkpoint(config["model_path"], MODEL, None, None)
    MODEL.eval()
    print(f"[WebApp] Domain '{domain_name}' Model & Tokenizer successfully loaded into memory.")

    lda_csv_path = config["lda_csv"]
    INSTANCE_TREE_DATA = []
    if os.path.exists(lda_csv_path):
        df_lda = pd.read_csv(lda_csv_path)
        print(f"[WebApp] Loaded {len(df_lda)} instances from LDA CSV: {lda_csv_path}")
    else:
        print(f"[WebApp] LDA CSV '{lda_csv_path}' missing. Generating GNN embeddings and 2D LDA/PCA projection...")
        try:
            df_lda = generate_and_save_lda_csv(
                graph_dir=config["graph_dir"],
                model=MODEL,
                device=DEVICE,
                output_csv_path=lda_csv_path,
                max_samples=None
            )
        except Exception as e:
            print(f"[WebApp] Error during LDA generation ({e}), falling back to filesystem scan.")
            df_lda = pd.DataFrame()

    if not df_lda.empty:
        for idx, row in df_lda.iterrows():
            INSTANCE_TREE_DATA.append({
                "id": idx,
                "filename": str(row["filename"]),
                "path": str(row["path"]) if "path" in row else os.path.join(config["graph_dir"], str(row["filename"])),
                "nv": int(row["nv"]) if "nv" in row else 0,
                "nc": int(row["nc"]) if "nc" in row else 0,
                "sat": bool(row["sat"]) if "sat" in row else True,
                "x": float(row["lda_x"]) if "lda_x" in row else 0.0,
                "y": float(row["pca_y"]) if "pca_y" in row else 0.0
            })
    else:
        g_dir = config["graph_dir"]
        if os.path.exists(g_dir):
            meta_list = load_training_graph_metadata(g_dir) if os.path.exists(g_dir) else []
            meta_map = {item["filename"]: item for item in meta_list if isinstance(item, dict) and "filename" in item}
            files = sorted([f for f in os.listdir(g_dir) if f.endswith(".graph")])
            for idx, fname in enumerate(files):
                fpath = os.path.join(g_dir, fname)
                hdr = meta_map.get(fname, {"nc": 0, "nv": 0, "sat": True})
                INSTANCE_TREE_DATA.append({
                    "id": idx,
                    "filename": fname,
                    "path": fpath,
                    "nv": hdr.get("nv", 0),
                    "nc": hdr.get("nc", 0),
                    "sat": hdr.get("sat", True),
                    "x": float(np.random.randn()),
                    "y": float(np.random.randn())
                })
        print(f"[WebApp] Loaded {len(INSTANCE_TREE_DATA)} graph instances from filesystem fallback.")

    TRAINING_INSTANCES = INSTANCE_TREE_DATA

def find_nearest_instance(x_click, y_click):
    best_item = None
    best_dist = float('inf')
    for item in INSTANCE_TREE_DATA:
        # Distance in LDA 2D projection space (lda_x, pca_y)
        dx = item["x"] - x_click
        dy = item["y"] - y_click
        dist = dx*dx + dy*dy
        if dist < best_dist:
            best_dist = dist
            best_item = item
    return best_item

@torch.no_grad()
def generate_instance_from_noisy_graph(graph_pyg, noise_std=0.1, max_new_tokens=1000, max_retries=5):
    bos_id = TOKENIZER.token_to_id("[BOS]")
    eos_id = TOKENIZER.token_to_id("[EOS]")
    pad_id = TOKENIZER.token_to_id("[PAD]")

    g = graph_pyg.to(DEVICE)
    if not hasattr(g, 'batch') or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=DEVICE)
    setattr(g, 'num_graphs', 1)

    node_embeddings = MODEL.graph_encoder(g) # [N, D]

    pid = os.getpid()

    for attempt in range(max_retries):
        if noise_std > 0:
            noise = torch.randn_like(node_embeddings) * noise_std
            noisy_embeddings = node_embeddings + noise
        else:
            noisy_embeddings = node_embeddings

        encoder_hidden_states = noisy_embeddings.unsqueeze(0) # [1, N, D]
        encoder_attention_mask = torch.ones((1, noisy_embeddings.size(0)), dtype=torch.long, device=DEVICE)

        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

        do_sample = attempt > 0
        kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
            "num_beams": 1,
            "do_sample": do_sample,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask
        }
        if do_sample:
            kwargs["temperature"] = 0.7
            kwargs["top_p"] = 0.9

        generated_ids = MODEL.decoder.generate(**kwargs)
        token_ids = generated_ids[0].cpu().tolist()
        final_raw_text = TOKENIZER.decode(token_ids, skip_special_tokens=True).strip()

        formatted_final = format_flatzinc_text(final_raw_text)
        nc_gen, nv_gen = parse_fzn_properties(formatted_final)
        if nc_gen is None or nv_gen is None:
            continue

        temp_eval_path = f".cache/ui_final_eval_{pid}_{attempt}.fzn"
        with open(temp_eval_path, "w") as f:
            f.write(formatted_final)

        eval_solver = "gecode" if (CURRENT_DOMAIN == "tsp" or "gecode_" in formatted_final or "circuit" in formatted_final) else "cp-sat"
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
            sat_gen = "=====unsatisfiable=====" not in combined_output
            return final_raw_text, formatted_final, nc_gen, nv_gen, sat_gen

    return None, None, None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/instances', methods=['GET'])
def get_instances():
    config = DOMAINS[CURRENT_DOMAIN]
    return jsonify({
        "success": True,
        "count": len(INSTANCE_TREE_DATA),
        "instances": INSTANCE_TREE_DATA,
        "current_domain": CURRENT_DOMAIN,
        "domain_name": config["name"],
        "nv_label": config["nv_label"],
        "nc_label": config["nc_label"],
        "default_solver": config.get("default_solver", "cp-sat")
    })

@app.route('/api/switch_domain', methods=['POST'])
def switch_domain():
    data = request.json or {}
    domain_name = data.get('domain', 'coloring')
    if domain_name not in DOMAINS:
        return jsonify({"success": False, "error": f"Invalid domain '{domain_name}'"}), 400

    try:
        init_app(domain_name)
        config = DOMAINS[CURRENT_DOMAIN]
        return jsonify({
            "success": True,
            "current_domain": CURRENT_DOMAIN,
            "domain_name": config["name"],
            "nv_label": config["nv_label"],
            "nc_label": config["nc_label"],
            "default_solver": config.get("default_solver", "cp-sat"),
            "count": len(INSTANCE_TREE_DATA),
            "instances": INSTANCE_TREE_DATA
        })
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed to switch domain: {str(e)}"}), 500

@app.route('/api/generate', methods=['POST'])
def generate_instance():
    data = request.json or {}
    x_val = float(data.get('x', 0.0))
    y_val = float(data.get('y', 0.0))
    noise_std = float(data.get('noise_std', 0.15))

    nearest = find_nearest_instance(x_val, y_val)
    if not nearest:
        return jsonify({"success": False, "error": "No dataset instances found."}), 404

    t0 = time.time()
    try:
        g_dict = load_graph_to_features(nearest["path"])
        g_pyg = convert_to_pyg_data(g_dict)
    except Exception as e:
        return jsonify({"success": False, "error": f"Failed loading graph features: {str(e)}"}), 500

    raw_text, formatted_body, nc_gen, nv_gen, sat_gen = generate_instance_from_noisy_graph(
        graph_pyg=g_pyg,
        noise_std=noise_std,
        max_new_tokens=1000,
        max_retries=3
    )

    gen_time = round(time.time() - t0, 2)

    if raw_text is None:
        return jsonify({"success": False, "error": "Generation failed to produce a valid model within retry limit."}), 500

    norm_gen = normalize_text(formatted_body)
    is_dup, duplicate_file = check_model_duplication(norm_gen, TRAINING_NORMALIZED_MAP)
    is_unique = not is_dup
    uniqueness_str = "UNIQUE (Novel Instance)" if is_unique else f"DUPLICATE (Identical to training instance: {duplicate_file})"

    header_comments = (
        f"% ========================================================\n"
        f"% NOISY EMBEDDING GENERATION METADATA\n"
        f"% Problem Domain: {DOMAINS[CURRENT_DOMAIN]['name']}\n"
        f"% Nearest Source Instance: {nearest['filename']}\n"
        f"% Source Properties: {DOMAINS[CURRENT_DOMAIN]['nc_label']} = {nearest['nc']}, {DOMAINS[CURRENT_DOMAIN]['nv_label']} = {nearest['nv']}, sat = {str(nearest['sat']).lower()}\n"
        f"% Noise Standard Deviation: σ = {noise_std}\n"
        f"% Clicked Coordinates: lda_x = {x_val:.2f}, pca_y = {y_val:.2f}\n"
        f"% Generated Properties: nc = {nc_gen}, nv = {nv_gen}, sat = {str(sat_gen).lower()}\n"
        f"% Training Set Uniqueness: {uniqueness_str}\n"
        f"% ========================================================\n\n"
    )

    full_flatzinc = header_comments + formatted_body

    return jsonify({
        "success": True,
        "closest_instance": nearest,
        "clicked": {"x": x_val, "y": y_val},
        "noise_std": noise_std,
        "nc_gen": nc_gen,
        "nv_gen": nv_gen,
        "sat_gen": sat_gen,
        "is_unique": is_unique,
        "duplicate_file": duplicate_file,
        "flatzinc_code": full_flatzinc,
        "body_code": formatted_body,
        "generation_time_sec": gen_time
    })

@app.route('/api/solve', methods=['POST'])
def solve_instance():
    data = request.json or {}
    code = data.get('flatzinc_code', '')
    default_solver = DOMAINS.get(CURRENT_DOMAIN, {}).get("default_solver", "cp-sat")
    if "gecode_" in code or CURRENT_DOMAIN == "tsp":
        default_solver = "gecode"
    solver = data.get('solver', default_solver)

    if not code:
        return jsonify({"success": False, "error": "No FlatZinc code provided."}), 400

    os.makedirs(".cache", exist_ok=True)
    temp_path = f".cache/web_solve_{os.getpid()}_{int(time.time()*1000)}.fzn"

    with open(temp_path, "w") as f:
        f.write(code)

    t0 = time.time()
    proc = subprocess.run(
        ["minizinc", temp_path, "--solver", solver, "-t", "10000"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    solve_time = round(time.time() - t0, 3)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    stdout = proc.stdout.decode()
    stderr = proc.stderr.decode()

    sat_status = "UNKNOWN"
    if "=====UNSATISFIABLE=====" in stdout:
        sat_status = "UNSATISFIABLE"
    elif "==========" in stdout:
        sat_status = "OPTIMAL / SATISFIED"
    elif "----------" in stdout:
        sat_status = "SATISFIED"
    elif proc.returncode != 0:
        sat_status = "SOLVER ERROR"

    return jsonify({
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "solver_used": solver,
        "solve_time_sec": solve_time,
        "status": sat_status
    })

if __name__ == '__main__':
    os.makedirs(".cache", exist_ok=True)
    init_app()
    app.run(host='0.0.0.0', port=5000, debug=False)
