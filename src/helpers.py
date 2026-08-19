"""
Helpers and Utility Functions for Graph-to-Text Constraint Generation,
FlatZinc AST Parsing, Feature Extraction, Embedding Probing, and Evaluation.
"""

import os
import re
import sys
import time
import random
import subprocess
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Scikit-learn & SciPy for probing and metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from scipy.stats import spearmanr

from torch_geometric.data import Data

def convert_to_pyg_data(graph_dict):
    """Converts a raw graph feature dictionary into a PyTorch Geometric Data instance."""
    x = torch.tensor(graph_dict["x"], dtype=torch.float)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(graph_dict["edge_attr"], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

try:
    from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
    from load_graph import load_graph_to_features
    from inference import load_checkpoint
except ImportError:
    pass


# ==============================================================================
# 1. CONSTANTS & CONFIGURATIONS
# ==============================================================================

ALL_NODE_TYPES = [
    'mult_node', 'literal_node', 'inequality_node', 'lin_sum_node',
    'par_node', 'var_node', 'equality_node', 'leq_node', 'geq_node',
    'circuit_node', 'int_element_node'
]

# In-Distribution parameter ranges
NUM_VERTICES_LIST = [10, 12, 14, 16, 18, 20]
EDGE_PROB_LIST = [0.2, 0.4, 0.6, 0.8]
NUM_COLORS_LIST = [3, 5, 7, 9, 11, 13, 15]

# Out-Of-Distribution (OOD) parameter ranges
OOD_NUM_VERTICES = [22, 24, 26, 28, 30]
OOD_EDGE_PROB = [0.85, 0.90, 0.95]
OOD_NUM_COLORS = [17, 19, 21, 25]


# ==============================================================================
# 2. FLATZINC TEXT, AST & STRUCTURAL NORMALIZATION
# ==============================================================================

def extract_model_features(text):
    """
    Extracts structural feature representations from FlatZinc / MiniZinc model code
    (variable counts, sorted variable domains, and unique stripped constraint lines).
    """
    text = re.sub(r"%.*?$", "", text, flags=re.MULTILINE)
    var_domains = []
    num_vars = 0
    constraint_lines = set()

    statements = text.split(";")
    for stmt in statements:
        s = stmt.strip()
        if not s:
            continue
        cleaned = re.sub(r"\s+", "", s)

        if s.startswith("var ") or " of var " in s or (s.startswith("array") and "var" in s):
            num_vars += 1
            dom_match = re.search(r"var\s+([^:]+):", s)
            if dom_match:
                dom = re.sub(r"\s+", "", dom_match.group(1))
                var_domains.append(dom)
            elif "of var int" in s:
                var_domains.append("int")
            else:
                var_domains.append("unknown")
        elif s.startswith("constraint "):
            constraint_lines.add(cleaned + ";")

    return {
        "num_vars": num_vars,
        "var_domains": tuple(sorted(var_domains)),
        "constraint_lines": frozenset(constraint_lines)
    }


def normalize_text(text):
    """
    Canonical normalized feature representation wrapper around extract_model_features.
    """
    return extract_model_features(text)


def check_model_duplication(model_feat, training_feat_map):
    """
    Compares model features against a dictionary of indexed models to detect exact structural duplicates.
    Returns (is_duplicate: bool, duplicate_filename: str or None).
    """
    if not model_feat or not model_feat["num_vars"]:
        return False, None
    for file, train_feat in training_feat_map.items():
        if (model_feat["num_vars"] == train_feat["num_vars"] and
            model_feat["var_domains"] == train_feat["var_domains"] and
            model_feat["constraint_lines"] == train_feat["constraint_lines"]):
            return True, file
    return False, None


def format_flatzinc_text(text):
    """
    Cleans, standardizes, and formats generated FlatZinc code strings.
    Ensures brackets, colons, and solve statements adhere to MiniZinc FlatZinc grammar.
    """
    if not text:
        return "solve satisfy;\n"

    # 1. Strip comments (% until next FlatZinc statement keyword or end of string)
    text = re.sub(
        r"%.*?(?=\b(predicate|var|array|constraint|solve)\b|$)", 
        "", 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    ).strip()

    # 2. Syntax and punctuation normalizations
    text = re.sub(r":\s*:", r"::", text)
    text = re.sub(r"(\d+)\s*\.\.\s*(\d+)", r"\1..\2", text)
    text = re.sub(r"\s*,\s*", r",", text)
    text = re.sub(r"\[\s*", r"[", text)
    text = re.sub(r"\s*\]", r"]", text)
    text = re.sub(r"\(\s*", r"(", text)
    text = re.sub(r"\s*\)", r")", text)
    text = re.sub(r"-\s*(\d+)", r"-\1", text)
    text = text.replace(": :", "::")

    # 3. Split by statement semicolon and filter valid statements
    lines = []
    statements = text.split(";")
    for stmt in statements:
        s = stmt.strip()
        if "%" in s:
            s = re.sub(
                r"%.*?(?=\b(predicate|var|array|constraint|solve)\b|$)", 
                "", 
                s, 
                flags=re.IGNORECASE | re.DOTALL
            ).strip()
        if s:
            if s.count("[") != s.count("]") or s.count("(") != s.count(")"):
                continue
            lines.append(s + ";")
    out = "\n".join(lines) + "\n"
    if "solve satisfy;" not in out and "solve maximize" not in out and "solve minimize" not in out:
        out += "solve satisfy;\n"
    return out


def clean_for_parser(fzn_text):
    """
    Normalizes whitespace and syntax for FlatZinc parser (flatzinc_parser.py) compatibility.
    """
    lines = []
    for line in fzn_text.splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        line = re.sub(r':\s*:', '::', line)
        line = re.sub(r'\s+:', ':', line)
        line = re.sub(r':\s+', ': ', line)
        line = re.sub(r'::\s*', '::', line)
        line = re.sub(r'array\s*\[\s*', 'array [', line)
        line = re.sub(r'\s*\]\s*of\s*', '] of ', line)
        line = re.sub(r'(\d+)\s*\.\.\s*(\d+)', r'\1..\2', line)
        line = re.sub(r'\s*,\s*', ',', line)
        line = re.sub(r'\[\s*', '[', line)
        line = re.sub(r'\s*\]', ']', line)
        line = re.sub(r'\s*\(\s*', '(', line)
        line = re.sub(r'\s*\)\s*', ')', line)
        lines.append(line)
    return '\n'.join(lines) + '\n'


def parse_fzn_properties(fzn_text):
    """
    Parses key domain parameters (nc / capacity / cost and nv / vertices / items / cities)
    from FlatZinc text across Graph Coloring, Knapsack, and TSP domains.
    """
    nc_gen = None
    nv_gen = None

    # 1. Look for TSP cost upper bound, Coloring NC, or Knapsack capacity
    cost_match = re.search(r"var\s+(-?\d+)\s*\.\.\s*(\d+)\s*:\s*cost", fzn_text)
    if cost_match:
        nc_gen = int(cost_match.group(2))
    else:
        nc_match = re.search(r"var\s+1\s*\.\.\s*(\d+)\s*:", fzn_text)
        if nc_match:
            nc_gen = int(nc_match.group(1))
        else:
            cap_match = re.search(r"int_lin_le\s*\([^;]*,\s*(\d+)\s*\)\s*;", fzn_text)
            if cap_match:
                nc_gen = int(cap_match.group(1))

    # 2. Look for variable count / array length (cities N, vertices NV, items NV)
    arr_match = re.search(r"array\s*\[\s*1\s*\.\.\s*(\d+)\s*\]\s*of\s*var\s*int\s*:\s*(color|x|d)", fzn_text)
    if arr_match:
        nv_gen = int(arr_match.group(1))
    else:
        all_indices = [int(m) for m in re.findall(r"X_INTRODUCED_(\d+)_", fzn_text)]
        nv_gen = (max(all_indices) + 1) if all_indices else None

    return nc_gen, nv_gen


# ==============================================================================
# 3. GRAPH PARSING & DATASET METADATA LOADERS
# ==============================================================================

def parse_header(file_path):
    """
    Parses graph metadata header across domains:
    - Coloring: %nc: 15, nv: 22, sat: true
    - Knapsack: %capacity: 60, profit: [...], desired_profit: 343, sat: true
    - TSP: %n: 6, min_dist: 4, max_dist: 15, sat: false
    - FlatZinc fallback: parses matching .fzn in companion _flat directory if header is omitted
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            first_line = f.readline().strip()
    except Exception:
        return None

    if first_line.startswith("%"):
        # 1. Coloring: %nc: 15, nv: 22, sat: true
        m = re.search(r"nc:\s*(\d+),\s*nv:\s*(\d+),\s*sat:\s*(true|false)", first_line, re.IGNORECASE)
        if m:
            return {
                "nc": int(m.group(1)),
                "nv": int(m.group(2)),
                "sat": m.group(3).lower() == "true"
            }

        # 2. Knapsack: %capacity: 60, profit: [...], size: [...], desired_profit: 343, sat: true
        m_cap = re.search(r"capacity:\s*(\d+)", first_line, re.IGNORECASE)
        m_sat = re.search(r"sat:\s*(true|false)", first_line, re.IGNORECASE)
        m_prof = re.search(r"profit:\s*\[([^\]]+)\]", first_line, re.IGNORECASE)
        if m_cap:
            cap = int(m_cap.group(1))
            nv = len(m_prof.group(1).split(",")) if m_prof else 0
            sat = (m_sat.group(1).lower() == "true") if m_sat else True
            return {"nc": cap, "nv": nv, "sat": sat}

        # 3. TSP: %n: 6, min_dist: 4, max_dist: 15, sat: false
        m_n = re.search(r"n:\s*(\d+)", first_line, re.IGNORECASE)
        m_cost = re.search(r"max_dist:\s*(\d+)", first_line, re.IGNORECASE)
        if m_n:
            n = int(m_n.group(1))
            cost = int(m_cost.group(1)) if m_cost else 0
            sat = (m_sat.group(1).lower() == "true") if m_sat else True
            return {"nc": cost, "nv": n, "sat": sat}

    # 4. Fallback: check matching .fzn in companion _flat directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    parent_dir = os.path.dirname(file_path)
    flat_dir = parent_dir.replace("_graphs", "_flat").replace("graphs", "flat")
    fzn_path = os.path.join(flat_dir, f"{base_name}.fzn")
    if os.path.exists(fzn_path):
        try:
            with open(fzn_path, "r") as f:
                content = f.read()
                m_nc = len(re.findall(r"\bconstraint\s+", content))
                m_nv = len(re.findall(r"\bvar\s+", content))
                return {"nc": m_nc, "nv": m_nv, "sat": True}
        except Exception:
            pass

    return {"nc": 0, "nv": 0, "sat": True}


def parse_graph_header(file_path):
    """Alias for parse_header."""
    return parse_header(file_path)


def parse_graph_tabular_features(file_path):
    """
    Parses metadata header (%nc: 15, nv: 22, sat: true)
    and counts total nodes and edges in the .graph file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    if not lines:
        return None

    first_line = lines[0].strip()
    if not first_line.startswith("%nc:"):
        return None

    match = re.search(r"nc:\s*(\d+),\s*nv:\s*(\d+),\s*sat:\s*(true|false)", first_line, re.IGNORECASE)
    if not match:
        return None

    nc = int(match.group(1))
    nv = int(match.group(2))
    sat = 1 if match.group(3).lower() == "true" else 0

    num_nodes = 0
    num_edges = 0
    current_section = None

    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if line == "nodes:":
            current_section = "nodes"
            continue
        elif line == "edges:":
            current_section = "edges"
            continue

        if current_section == "nodes":
            num_nodes += 1
        elif current_section == "edges":
            num_edges += 1

    return {
        "file": os.path.basename(file_path),
        "nc": nc,
        "nv": nv,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "sat": sat,
    }


def load_training_graph_metadata(graph_dir=None, csv_path=None):
    """
    Loads training graph metadata (nc, nv, sat, path) from CSV or by scanning .graph files.
    """
    if graph_dir is None:
        graph_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    if csv_path is None:
        csv_path = "data/outputs/baseline_10k_distribution.csv" if os.path.exists("data/outputs/baseline_10k_distribution.csv") else "baseline_10k_distribution.csv"

    items = []
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            g_path = os.path.join(graph_dir, row["filename"])
            if os.path.exists(g_path):
                items.append({
                    "path": g_path,
                    "nc": int(row["nc"]),
                    "nv": int(row["nv"]),
                    "sat": bool(row["sat"])
                })

    if not items and os.path.exists(graph_dir):
        for f in os.listdir(graph_dir):
            if f.endswith(".graph"):
                g_path = os.path.join(graph_dir, f)
                meta = parse_header(g_path)
                if meta:
                    items.append({
                        "path": g_path,
                        "nc": meta["nc"],
                        "nv": meta["nv"],
                        "sat": meta["sat"]
                    })
    return items


def load_dataset(graph_dir, max_samples=None):
    """
    Loads tabular structural features for all .graph files in a directory into a DataFrame.
    """
    if not os.path.exists(graph_dir):
        raise FileNotFoundError(f"Directory {graph_dir} not found.")

    all_files = [os.path.join(graph_dir, f) for f in sorted(os.listdir(graph_dir)) if f.endswith(".graph")]
    if max_samples:
        all_files = all_files[:max_samples]

    records = []
    for path in tqdm(all_files, desc=f"Loading {os.path.basename(graph_dir)}"):
        feat = parse_graph_tabular_features(path)
        if feat is not None:
            records.append(feat)

    df = pd.DataFrame(records)
    return df


def load_tabular_dataset(graph_dir, max_samples=None):
    """Alias for load_dataset."""
    return load_dataset(graph_dir, max_samples)


# ==============================================================================
# 4. EMBEDDING EXTRACTION, LATENT PROBING & EVALUATION
# ==============================================================================

@torch.no_grad()
def extract_dataset_embeddings(graph_files, model, device):
    """
    Extracts mean-pooled GNN encoder embeddings and labels (sat, nv, nc) for a list of graph filepaths.
    """
    embeddings = []
    metadata_list = []

    for path in tqdm(graph_files, desc="Extracting embeddings"):
        meta = parse_header(path)
        if meta is None:
            continue
        try:
            graph_dict = load_graph_to_features(path)
            pyg_data = convert_to_pyg_data(graph_dict).to(device)
            if not hasattr(pyg_data, 'batch') or pyg_data.batch is None:
                pyg_data.batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)
            setattr(pyg_data, 'num_graphs', 1)

            node_embeddings = model.graph_encoder(pyg_data)
            graph_embedding = node_embeddings.mean(dim=0).cpu().numpy()

            embeddings.append(graph_embedding)
            metadata_list.append(meta)
        except Exception:
            continue

    X = np.array(embeddings)
    y_sat = np.array([m["sat"] for m in metadata_list], dtype=int)
    y_nv = np.array([m["nv"] for m in metadata_list], dtype=int)
    y_nc = np.array([m["nc"] for m in metadata_list], dtype=int)

    return X, y_sat, y_nv, y_nc


@torch.no_grad()
def extract_embeddings_and_labels(graph_dir=None, max_samples=3000, model=None, tokenizer=None, device=None, model_path="models/model_mixed.pt"):
    """
    Convenience wrapper to load model checkpoint and extract encoder embeddings from a directory of .graph files.
    """
    if graph_dir is None:
        graph_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = train_and_get_tokenizer()

    all_files = [f for f in sorted(os.listdir(graph_dir)) if f.endswith(".graph")]
    valid_files = []
    metadata_list = []

    for file in all_files:
        path = os.path.join(graph_dir, file)
        meta = parse_header(path)
        if meta is not None:
            valid_files.append(path)
            metadata_list.append(meta)
        if len(valid_files) >= max_samples:
            break

    if not valid_files:
        raise ValueError(f"No valid graph files found in {graph_dir}")

    if model is None:
        sample_g = load_graph_to_features(valid_files[0])
        node_in_dim = len(sample_g["x"][0])
        edge_in_dim = len(sample_g["edge_attr"][0])

        model = GraphToTextConditionalGeneration(
            node_in_dim, edge_in_dim, 10000,
            tokenizer.token_to_id("[BOS]"),
            tokenizer.token_to_id("[EOS]"),
            tokenizer.token_to_id("[PAD]")
        ).to(device)
        model, _, _, _, _, _ = load_checkpoint(model_path, model, None, None)
        model.eval()

    return extract_dataset_embeddings(valid_files, model, device)


def compute_metrics(y_true, y_pred, dataset_name="Dataset"):
    """
    Computes accuracy, balanced accuracy, macro-F1, class-specific recall and precision, and confusion matrix.
    """
    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    unsat_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    sat_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    unsat_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0) * 100
    sat_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0) * 100
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n--- {dataset_name} ---")
    print(f"Overall Accuracy:  {acc:.2f}%")
    print(f"Balanced Accuracy: {bal_acc:.2f}%")
    print(f"Macro F1-Score:    {macro_f1:.4f}")
    print(f"SAT Recall (1):    {sat_rec:.2f}%  | SAT Precision:   {sat_prec:.2f}%")
    print(f"UNSAT Recall (0):  {unsat_rec:.2f}%  | UNSAT Precision: {unsat_prec:.2f}%")
    print("Confusion Matrix (Rows=True, Cols=Pred [UNSAT=0, SAT=1]):")
    print(cm)

    return {
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
        "sat_rec": sat_rec,
        "unsat_rec": unsat_rec,
        "sat_prec": sat_prec,
        "unsat_prec": unsat_prec,
        "cm": cm
    }


def evaluate_lda(X_train, y_train, X_test, y_test, target_name="Target"):
    """
    Fits Linear Discriminant Analysis on X_train/y_train and evaluates on X_test/y_test.
    """
    print(f"\n==========================================")
    print(f"  LDA Classification Evaluation: {target_name}")
    print(f"==========================================")

    unique_classes, counts = np.unique(y_test, return_counts=True)
    majority_class_acc = np.max(counts) / len(y_test)
    print(f"Test set classes: {unique_classes}")
    print(f"Test set class distribution: {dict(zip(unique_classes, counts))}")
    print(f"Baseline (Majority Class Accuracy on Test Set): {majority_class_acc * 100:.2f}%")

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_pred = lda.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Held-out Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return lda, acc, y_pred


def evaluate_classifier_performance(model, model_name, X_train, y_train, X_id, y_id, X_ood, y_ood, feature_names):
    """
    Evaluates a scikit-learn classifier pipeline with 5-fold CV on train set, and tests on ID and OOD test sets.
    """
    print(f"\n========================================================================")
    print(f"  MODEL: {model_name}")
    print(f"  Features used: {feature_names}")
    print(f"========================================================================")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X_train, y_train, cv=skf, scoring=['accuracy', 'balanced_accuracy', 'f1_macro']
    )
    cv_acc = np.mean(cv_results['test_accuracy']) * 100
    cv_bal_acc = np.mean(cv_results['test_balanced_accuracy']) * 100
    cv_f1 = np.mean(cv_results['test_f1_macro'])

    print(f"5-Fold CV Accuracy:          {cv_acc:.2f}%")
    print(f"5-Fold CV Balanced Accuracy: {cv_bal_acc:.2f}%")
    print(f"5-Fold CV Macro F1:          {cv_f1:.4f}")

    model.fit(X_train, y_train)

    if hasattr(model, 'coef_'):
        coefs = model.coef_[0]
        intercept = model.intercept_[0] if hasattr(model, 'intercept_') else 0.0
        coef_str = " + ".join([f"({c:+.4f} * {name})" for c, name in zip(coefs, feature_names)])
        print(f"Decision Boundary Equation: Logit = {intercept:+.4f} + {coef_str}")
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'coef_'):
        clf = model.named_steps['classifier']
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]
        coef_str = " + ".join([f"({c:+.4f} * std_{name})" for c, name in zip(coefs, feature_names)])
        print(f"Scaled Linear Equation: Logit = {intercept:+.4f} + {coef_str}")

    y_pred_id = model.predict(X_id)
    id_metrics = compute_metrics(y_id, y_pred_id, "In-Distribution (ID) Test Set")

    y_pred_ood = model.predict(X_ood)
    ood_metrics = compute_metrics(y_ood, y_pred_ood, "Out-of-Distribution (OOD) Test Set")

    return {
        "model_name": model_name,
        "features": feature_names,
        "cv_acc": cv_acc,
        "cv_bal_acc": cv_bal_acc,
        "cv_f1": cv_f1,
        "id_metrics": id_metrics,
        "ood_metrics": ood_metrics
    }


def generate_lda_plots(X_train, y_train_dict, X_test, y_test_dict, output_dir="data/outputs"):
    """
    Fits LDA/PCA on training embeddings and creates 2D scatter plots for test embeddings.
    `y_train_dict` and `y_test_dict` should contain 'nc', 'nv', and 'sat' label arrays.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_theme(style="darkgrid", palette="deep")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Colors (nc)
    lda_nc = LinearDiscriminantAnalysis(n_components=2)
    lda_nc.fit(X_train, y_train_dict["nc"])
    X_nc_2d = lda_nc.transform(X_test)

    # 2. Vertices (nv)
    lda_nv = LinearDiscriminantAnalysis(n_components=2)
    lda_nv.fit(X_train, y_train_dict["nv"])
    X_nv_2d = lda_nv.transform(X_test)

    # 3. Satisfiability (sat)
    lda_sat = LinearDiscriminantAnalysis(n_components=1)
    lda_sat.fit(X_train, y_train_dict["sat"])
    ld1_sat = lda_sat.transform(X_test).flatten()
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_pca = pca.transform(X_test)
    X_sat_2d = np.column_stack((ld1_sat, X_pca[:, 1]))

    # Plot nc
    plt.figure(figsize=(9, 7), dpi=300)
    palette_nc = sns.color_palette("plasma", n_colors=len(np.unique(y_test_dict["nc"])))
    sns.scatterplot(
        x=X_nc_2d[:, 0], y=X_nc_2d[:, 1], hue=y_test_dict["nc"], palette=palette_nc,
        s=45, alpha=0.85, edgecolor='w', linewidth=0.3
    )
    plt.title("2D LDA Projection: Number of Colors (nc)", fontsize=14, fontweight='bold')
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
    plt.legend(title="Colors (nc)", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lda_2d_scatter_nc.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot nv
    plt.figure(figsize=(9, 7), dpi=300)
    palette_nv = sns.color_palette("viridis", n_colors=len(np.unique(y_test_dict["nv"])))
    sns.scatterplot(
        x=X_nv_2d[:, 0], y=X_nv_2d[:, 1], hue=y_test_dict["nv"], palette=palette_nv,
        s=45, alpha=0.85, edgecolor='w', linewidth=0.3
    )
    plt.title("2D LDA Projection: Number of Vertices (nv)", fontsize=14, fontweight='bold')
    plt.xlabel("LDA Component 1")
    plt.ylabel("LDA Component 2")
    plt.legend(title="Vertices (nv)", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lda_2d_scatter_nv.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot sat
    plt.figure(figsize=(9, 7), dpi=300)
    sat_labels = np.array(["UNSAT" if s == 0 else "SAT" for s in y_test_dict["sat"]])
    palette_sat = {"UNSAT": "#E63946", "SAT": "#1D3557"}
    sns.scatterplot(
        x=X_sat_2d[:, 0], y=X_sat_2d[:, 1], hue=sat_labels, palette=palette_sat,
        s=45, alpha=0.85, edgecolor='w', linewidth=0.3
    )
    plt.title("2D LDA/PCA Projection: Satisfiability (sat)", fontsize=14, fontweight='bold')
    plt.xlabel("LDA Component 1 (Separating Axis)")
    plt.ylabel("Orthogonal PCA Component")
    plt.legend(title="Satisfiability", bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lda_2d_scatter_sat.png"), dpi=300, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def generate_and_save_lda_csv(graph_dir, model, device, output_csv_path, max_samples=1000):
    """
    Extracts graph node embeddings for graph files in graph_dir using model.graph_encoder,
    computes 2D LDA/PCA coordinates (lda_x, pca_y), and saves them to output_csv_path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
    if not os.path.exists(graph_dir):
        raise FileNotFoundError(f"Graph directory '{graph_dir}' not found.")

    files = sorted([f for f in os.listdir(graph_dir) if f.endswith(".graph")])
    if max_samples and max_samples > 0:
        files = files[:max_samples]

    embeddings = []
    metadata = []
    valid_paths = []
    valid_fnames = []

    model.eval()
    for fname in tqdm(files, desc=f"Extracting embeddings from {os.path.basename(graph_dir)}"):
        fpath = os.path.join(graph_dir, fname)
        hdr = parse_header(fpath)
        if hdr is None:
            continue
        try:
            g_dict = load_graph_to_features(fpath)
            pyg = convert_to_pyg_data(g_dict).to(device)
            if not hasattr(pyg, 'batch') or pyg.batch is None:
                pyg.batch = torch.zeros(pyg.num_nodes, dtype=torch.long, device=device)
            setattr(pyg, 'num_graphs', 1)

            node_embs = model.graph_encoder(pyg)
            graph_emb = node_embs.mean(dim=0).cpu().numpy()

            embeddings.append(graph_emb)
            metadata.append(hdr)
            valid_paths.append(fpath)
            valid_fnames.append(fname)
        except Exception:
            continue

    if not embeddings:
        raise ValueError(f"No valid embeddings could be extracted from {graph_dir}")

    X = np.array(embeddings)
    y_sat = np.array([m["sat"] for m in metadata], dtype=int)
    y_nv = np.array([m["nv"] for m in metadata], dtype=int)
    y_nc = np.array([m["nc"] for m in metadata], dtype=int)

    pca = PCA(n_components=min(2, X.shape[1], X.shape[0]))
    X_pca = pca.fit_transform(X)

    lda_x = X_pca[:, 0]
    pca_y = X_pca[:, 1] if X_pca.shape[1] > 1 else np.zeros_like(lda_x)

    if len(np.unique(y_sat)) > 1:
        try:
            lda_sat = LinearDiscriminantAnalysis(n_components=1)
            lda_sat.fit(X, y_sat)
            lda_x = lda_sat.transform(X).flatten()
            if X_pca.shape[1] > 1:
                pca_y = X_pca[:, 1]
        except Exception:
            pass
    elif len(np.unique(y_nc)) > 1:
        try:
            n_comps = min(2, len(np.unique(y_nc)) - 1)
            lda_nc = LinearDiscriminantAnalysis(n_components=n_comps)
            X_lda = lda_nc.fit_transform(X, y_nc)
            lda_x = X_lda[:, 0]
            if X_lda.shape[1] > 1:
                pca_y = X_lda[:, 1]
            elif X_pca.shape[1] > 1:
                pca_y = X_pca[:, 1]
        except Exception:
            pass

    df = pd.DataFrame({
        "filename": valid_fnames,
        "path": valid_paths,
        "nv": y_nv,
        "nc": y_nc,
        "sat": [bool(s) for s in y_sat],
        "lda_x": lda_x,
        "pca_y": pca_y
    })

    df.to_csv(output_csv_path, index=False)
    print(f"[Helpers] Saved {len(df)} LDA/PCA coordinates to '{output_csv_path}'")
    return df


# ==============================================================================
# 5. INSTANCE GENERATION & COMPILATION PIPELINES
# ==============================================================================

def generate_coloring_instance(num_vertices, edge_probability, num_colors, filename="instance.dzn"):
    """
    Generates a random Erdős-Rényi graph coloring instance and writes it as a MiniZinc .dzn data file.
    """
    import networkx as nx
    graph = nx.erdos_renyi_graph(n=num_vertices, p=edge_probability)
    edges = [(u + 1, v + 1) for u, v in graph.edges()]
    num_edges = len(edges)

    if num_edges > 0:
        edge_strings = [f" {u}, {v} " for u, v in edges]
        minizinc_edges = "| " + " | ".join(edge_strings) + " |"
    else:
        minizinc_edges = ""

    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filename, "w") as f:
        f.write(f"% Generated Graph Coloring Instance\n")
        f.write(f"nc = {num_colors};\n")
        f.write(f"nv = {num_vertices};\n")
        f.write(f"ne = {num_edges};\n\n")
        f.write(f"edges = [{minizinc_edges}];\n")


def generate_graph(i, num_vertices, edge_probability, num_colors, mzn_model="model.mzn", instances_dir="instances", graphs_dir="graphs", parser_path="../flatzinc_parser/flatzinc_parser.py"):
    """
    Solves a generated instance, compiles it to FlatZinc, parses to .graph, and saves with metadata header.
    """
    dzn_path = os.path.join(instances_dir, f"instance_{i}.dzn")
    graph_path = os.path.join(graphs_dir, f"instance_{i}.graph")
    fzn_path = f".cache/instance_{i}.fzn"
    os.makedirs(".cache", exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)

    stdout = subprocess.run([f'minizinc {mzn_model} {dzn_path} --solver cp-sat'], shell=True, stdout=subprocess.PIPE).stdout.decode()
    sat = "=====UNSATISFIABLE=====" not in stdout

    subprocess.run([f'minizinc {mzn_model} {dzn_path} --solver gecode -c --no-output-ozn --fzn {fzn_path}'], shell=True)
    subprocess.run([f'python {parser_path} {fzn_path} {graph_path}'], shell=True)
    if os.path.exists(fzn_path):
        os.remove(fzn_path)

    if os.path.exists(graph_path):
        with open(graph_path, 'r+') as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"%nc: {num_colors}, nv: {num_vertices}, sat: {str(sat).lower()}\n" + content)


def process_instance(i, num_vertices_list=NUM_VERTICES_LIST, edge_probability_list=EDGE_PROB_LIST, num_colors_list=NUM_COLORS_LIST):
    """
    Task worker for generating and compiling a single graph coloring dataset instance with an isolated RNG.
    """
    rng = random.Random(i)
    n_vertices = rng.choice(num_vertices_list)
    e_prob = rng.choice(edge_probability_list)
    n_cols = rng.choice(num_colors_list)

    generate_coloring_instance(num_vertices=n_vertices, edge_probability=e_prob, num_colors=n_cols, filename=f"instances/instance_{i}.dzn")
    generate_graph(i, n_vertices, e_prob, n_cols)
    return i


def process_flattening_instance(instance, instance_dir="instances/", flat_dir="flat/", model_path="model.mzn"):
    """
    Compiles a .dzn file to .fzn using MiniZinc gecode solver and strips comment lines.
    """
    os.makedirs(flat_dir, exist_ok=True)
    fzn_filename = instance.replace(".dzn", ".fzn") if instance.endswith(".dzn") else instance + ".fzn"
    fzn_path = os.path.join(flat_dir, fzn_filename)
    dzn_path = os.path.join(instance_dir, instance)

    if os.path.exists(fzn_path):
        return

    cmd = f"minizinc -c --no-output-ozn --solver gecode {model_path} {dzn_path} --fzn {fzn_path}"
    subprocess.run([cmd], shell=True)

    if os.path.exists(fzn_path):
        lines = []
        with open(fzn_path, "r") as f:
            for line in f:
                if line[0] != "%":
                    if "%" in line:
                        lines.append(line[: line.index("%")])
                    else:
                        lines.append(line)
        with open(fzn_path, "w") as f:
            f.write("\n".join(lines))


def generate_single_ood_instance(idx, out_dzn_dir="data/ood_test_instances", out_graph_dir="data/ood_test_graphs", model_path="model.mzn", parser_path="/work/flatzinc_parser/flatzinc_parser.py"):
    """
    Generates a single out-of-distribution (OOD) test graph instance with larger parameters.
    """
    os.makedirs(out_dzn_dir, exist_ok=True)
    os.makedirs(out_graph_dir, exist_ok=True)
    os.makedirs(".cache", exist_ok=True)

    rng = random.Random(idx + 90000)
    nv = rng.choice(OOD_NUM_VERTICES)
    p = rng.choice(OOD_EDGE_PROB)
    nc = rng.choice(OOD_NUM_COLORS)

    dzn_file = os.path.join(out_dzn_dir, f"ood_instance_{idx}.dzn")
    graph_file = os.path.join(out_graph_dir, f"ood_instance_{idx}.graph")
    fzn_file = os.path.join(".cache", f"ood_{idx}.fzn")

    generate_coloring_instance(num_vertices=nv, edge_probability=p, num_colors=nc, filename=dzn_file)

    cp_res = subprocess.run(
        [f"minizinc {model_path} {dzn_file} --solver cp-sat -t 10000"],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.decode()
    sat = "=====UNSATISFIABLE=====" not in cp_res

    subprocess.run(
        [f"minizinc {model_path} {dzn_file} --solver gecode -c --no-output-ozn --fzn {fzn_file}"],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if os.path.exists(parser_path):
        subprocess.run(
            [f"python {parser_path} {fzn_file} {graph_file}"],
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    if os.path.exists(fzn_file):
        os.remove(fzn_file)

    if os.path.exists(graph_file):
        with open(graph_file, "r+") as f:
            content = f.read()
            f.seek(0, 0)
            f.write(f"%nc: {nc}, nv: {nv}, sat: {str(sat).lower()}\n" + content)
        return idx
    return None


@torch.no_grad()
def generate_single_instance_line_by_line(model, tokenizer, gA_pyg, gB_pyg, device, max_new_tokens=2048, max_retries=5):
    """
    Dual-graph generation with statement-level syntax and constraint validation via MiniZinc cp-sat.
    """
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    semicolon_id = tokenizer.token_to_id(";")

    gA = gA_pyg.to(device)
    gB = gB_pyg.to(device)
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
    encoder_hidden_states = mean_emb.unsqueeze(0)
    encoder_attention_mask = torch.ones((1, max_N), dtype=torch.long, device=device)

    pid = os.getpid()

    for attempt in range(max_retries):
        input_ids = torch.full((1, 1), bos_id, dtype=torch.long, device=device)
        past_key_values = None
        curr_input_ids = input_ids
        is_valid_so_far = True

        for step in range(max_new_tokens):
            with torch.no_grad():
                outputs = model.decoder(
                    input_ids=curr_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask
                )
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

            if attempt == 0:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            token_val = next_token.item()
            input_ids = torch.cat([input_ids, next_token], dim=1)
            curr_input_ids = next_token

            if token_val == eos_id:
                break

            if token_val == semicolon_id:
                raw_prefix = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True).strip()
                formatted_prefix = format_flatzinc_text(raw_prefix)

                if "solve satisfy;" not in formatted_prefix:
                    eval_text = formatted_prefix.strip() + "\nsolve satisfy;\n"
                else:
                    eval_text = formatted_prefix

                temp_eval_path = f".cache/step_eval_{pid}_{attempt}.fzn"
                with open(temp_eval_path, "w") as f:
                    f.write(eval_text)

                sol_proc = subprocess.run(
                    ["minizinc", temp_eval_path, "--solver", "gecode", "-t", "5000"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )

                if os.path.exists(temp_eval_path):
                    os.remove(temp_eval_path)

                sol_err = sol_proc.stderr.decode()
                if sol_proc.returncode != 0 or "Error:" in sol_err or "error:" in sol_err or "syntax error" in sol_err.lower():
                    is_valid_so_far = False
                    break

        if not is_valid_so_far:
            continue

        final_raw_text = tokenizer.decode(input_ids[0].cpu().tolist(), skip_special_tokens=True).strip()
        nc_gen, nv_gen = parse_fzn_properties(final_raw_text)
        if nc_gen is None or nv_gen is None:
            continue

        formatted_final = format_flatzinc_text(final_raw_text)
        if "solve satisfy;" not in formatted_final:
            formatted_final = formatted_final.strip() + "\nsolve satisfy;\n"

        temp_eval_path = f".cache/final_eval_{pid}_{attempt}.fzn"
        with open(temp_eval_path, "w") as f:
            f.write(formatted_final)

        sol_proc = subprocess.run(
            ["minizinc", temp_eval_path, "--solver", "cp-sat", "-t", "5000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if os.path.exists(temp_eval_path):
            os.remove(temp_eval_path)

        sol_out = sol_proc.stdout.decode()
        sol_err = sol_proc.stderr.decode()

        if sol_proc.returncode == 0 and "Error:" not in sol_err and "error:" not in sol_err:
            sat_gen = "=====UNSATISFIABLE=====" not in sol_out
            return final_raw_text, formatted_final, nc_gen, nv_gen, sat_gen

    return None, None, None, None, None


@torch.no_grad()
def generate_web_app_pipeline(model, tokenizer, graph_pyg, domain="coloring", noise_std=0.15, max_new_tokens=1000, max_retries=3, device="cpu"):
    """
    Unified generation pipeline using noise-perturbed graph encoder embeddings and MiniZinc verification.
    """
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")

    g = graph_pyg.to(device)
    if not hasattr(g, 'batch') or g.batch is None:
        g.batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
    setattr(g, 'num_graphs', 1)

    node_embeddings = model.graph_encoder(g)
    pid = os.getpid()

    for attempt in range(max_retries):
        if noise_std > 0:
            noise = torch.randn_like(node_embeddings) * noise_std
            noisy_embeddings = node_embeddings + noise
        else:
            noisy_embeddings = node_embeddings

        encoder_hidden_states = noisy_embeddings.unsqueeze(0)
        encoder_attention_mask = torch.ones((1, noisy_embeddings.size(0)), dtype=torch.long, device=device)

        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

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

        generated_ids = model.decoder.generate(**kwargs)
        token_ids = generated_ids[0].cpu().tolist()
        final_raw_text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()

        formatted_final = format_flatzinc_text(final_raw_text)
        nc_gen, nv_gen = parse_fzn_properties(formatted_final)
        if nc_gen is None or nv_gen is None:
            continue

        temp_eval_path = f".cache/ui_eval_{pid}_{attempt}.fzn"
        os.makedirs(".cache", exist_ok=True)
        with open(temp_eval_path, "w") as f:
            f.write(formatted_final)

        eval_solver = "gecode" if (domain == "tsp" or "gecode_" in formatted_final) else "cp-sat"
        sol_proc = subprocess.run(
            ["minizinc", temp_eval_path, "--solver", eval_solver, "-t", "5000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if os.path.exists(temp_eval_path):
            os.remove(temp_eval_path)

        sol_out = sol_proc.stdout.decode()
        sol_err = sol_proc.stderr.decode()

        has_syntax_error = any(err_kw in sol_err.lower() for err_kw in ["syntax error", "type error", "undefined identifier", "unexpected"])
        if not has_syntax_error:
            sat_gen = "=====UNSATISFIABLE=====" not in sol_out
            return node_embeddings, noisy_embeddings, formatted_final, nc_gen, nv_gen, sat_gen

    return node_embeddings, noisy_embeddings, None, None, None, None
