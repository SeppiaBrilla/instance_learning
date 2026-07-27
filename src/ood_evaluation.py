import os
import re
import random
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_absolute_error, r2_score

# Ensure minizinc binary is in PATH
os.environ['PATH'] = '/work/minizinc_bundle/bin:' + os.environ.get('PATH', '')

from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from inference import load_checkpoint, convert_to_pyg_data
from generator import generate_coloring_instance

# Out-Of-Distribution (OOD) Parameter Ranges
OOD_NUM_VERTICES = [22, 24, 26, 28, 30]
OOD_EDGE_PROB = [0.85, 0.90, 0.95]
OOD_NUM_COLORS = [17, 19, 21, 25]

def generate_single_ood_instance(idx, out_dzn_dir=None, out_graph_dir=None):
    if out_dzn_dir is None:
        out_dzn_dir = "data/ood_test_instances" if os.path.exists("data/ood_test_instances") else "ood_test_instances"
    if out_graph_dir is None:
        out_graph_dir = "data/ood_test_graphs" if os.path.exists("data/ood_test_graphs") else "ood_test_graphs"
    os.makedirs(out_dzn_dir, exist_ok=True)
    os.makedirs(out_graph_dir, exist_ok=True)

    rng = random.Random(idx + 90000)
    nv = rng.choice(OOD_NUM_VERTICES)
    p = rng.choice(OOD_EDGE_PROB)
    nc = rng.choice(OOD_NUM_COLORS)

    dzn_file = os.path.join(out_dzn_dir, f"ood_instance_{idx}.dzn")
    graph_file = os.path.join(out_graph_dir, f"ood_instance_{idx}.graph")
    fzn_file = os.path.join(".cache", f"ood_{idx}.fzn")

    generate_coloring_instance(num_vertices=nv, edge_probability=p, num_colors=nc, filename=dzn_file)

    # 1. Solve with cp-sat (with 10 sec timeout for larger graphs)
    cp_res = subprocess.run(
        [f"minizinc model.mzn {dzn_file} --solver cp-sat -t 10000"],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout.decode()
    sat = "=====UNSATISFIABLE=====" not in cp_res

    # 2. Compile to FlatZinc and parse to graph format
    subprocess.run(
        [f"minizinc model.mzn {dzn_file} --solver gecode -c --no-output-ozn --fzn {fzn_file}"],
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    subprocess.run(
        [f"python /work/flatzinc_parser/flatzinc_parser.py {fzn_file} {graph_file}"],
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
    else:
        return None

def parse_header(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    if not first_line.startswith("%nc:"):
        return None
    match = re.search(r"nc:\s*(\d+),\s*nv:\s*(\d+),\s*sat:\s*(true|false)", first_line, re.IGNORECASE)
    if not match:
        return None
    return {
        "nc": int(match.group(1)),
        "nv": int(match.group(2)),
        "sat": match.group(3).lower() == "true"
    }

@torch.no_grad()
def extract_dataset_embeddings(graph_files, model, device):
    embeddings = []
    metadata_list = []

    for path in tqdm(graph_files):
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
        except Exception as e:
            continue

    X = np.array(embeddings)
    y_sat = np.array([m["sat"] for m in metadata_list], dtype=int)
    y_nv = np.array([m["nv"] for m in metadata_list], dtype=int)
    y_nc = np.array([m["nc"] for m in metadata_list], dtype=int)

    return X, y_sat, y_nv, y_nc

def main():
    print("=======================================================")
    print("  1. Generating Out-Of-Distribution (OOD) Test Cases   ")
    print("  (Higher Vertices: 22-30, Colors: 17-25, Prob: 0.85-0.95)")
    print("=======================================================")
    
    os.makedirs(".cache", exist_ok=True)
    out_dzn_dir = "data/ood_test_instances" if os.path.exists("data/ood_test_instances") else "ood_test_instances"
    out_graph_dir = "data/ood_test_graphs" if os.path.exists("data/ood_test_graphs") else "ood_test_graphs"
    os.makedirs(out_dzn_dir, exist_ok=True)
    os.makedirs(out_graph_dir, exist_ok=True)

    num_ood_samples = 250
    print(f"Generating {num_ood_samples} OOD test graph instances...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(generate_single_ood_instance, i, out_dzn_dir, out_graph_dir) for i in range(num_ood_samples)]
        for f in tqdm(as_completed(futures), total=num_ood_samples):
            f.result()

    ood_files = [os.path.join(out_graph_dir, f) for f in sorted(os.listdir(out_graph_dir)) if f.endswith(".graph")]
    print(f"Successfully generated {len(ood_files)} OOD test graphs.")

    print("\n=======================================================")
    print("  2. Loading Model & Extracting Embeddings             ")
    print("=======================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = train_and_get_tokenizer()

    sample_g = load_graph_to_features(ood_files[0])
    model = GraphToTextConditionalGeneration(
        len(sample_g["x"][0]), 
        len(sample_g["edge_attr"][0]), 
        10000, 
        tokenizer.token_to_id("[BOS]"), 
        tokenizer.token_to_id("[EOS]"), 
        tokenizer.token_to_id("[PAD]")
    ).to(device)

    model_path = "models/final_model_80m.pt" if os.path.exists("models/final_model_80m.pt") else "final_model_80m.pt"
    model, _, _, _, _, _ = load_checkpoint(model_path, model, None, None)
    model.eval()

    train_graph_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    train_graph_files = [os.path.join(train_graph_dir, f) for f in sorted(os.listdir(train_graph_dir)) if f.endswith(".graph")][:2500]
    
    print("\nExtracting embeddings for IN-DISTRIBUTION Training Set (2,500 graphs)...")
    X_train, y_sat_train, y_nv_train, y_nc_train = extract_dataset_embeddings(train_graph_files, model, device)

    print("\nExtracting embeddings for OUT-OF-DISTRIBUTION Test Set (250 graphs)...")
    X_ood, y_sat_ood, y_nv_ood, y_nc_ood = extract_dataset_embeddings(ood_files, model, device)

    print("\n=======================================================")
    print("  3. OOD Evaluation Results                            ")
    print("=======================================================")

    # A. SATISFIABILITY (Binary Classification)
    print("\n-------------------------------------------------------")
    print(" Target: SATISFIABILITY (sat) [OOD Test]")
    print("-------------------------------------------------------")
    lda_sat = LinearDiscriminantAnalysis()
    lda_sat.fit(X_train, y_sat_train)
    y_pred_sat = lda_sat.predict(X_ood)
    acc_sat = accuracy_score(y_sat_ood, y_pred_sat)
    print(f"OOD Test Accuracy: {acc_sat * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_sat_ood, y_pred_sat, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_sat_ood, y_pred_sat))

    # B. NUMBER OF VERTICES (Continuous Extrapolation via Ridge Linear Probe & Classification)
    print("\n-------------------------------------------------------")
    print(" Target: NUMBER OF VERTICES (nv) [OOD Test Range: 22 to 30]")
    print(" Training Range: 10 to 20")
    print("-------------------------------------------------------")
    # Linear Regression Extrapolation for continuous nv
    reg_nv = Ridge(alpha=1.0)
    reg_nv.fit(X_train, y_nv_train)
    y_pred_nv_reg = reg_nv.predict(X_ood)
    mae_nv = mean_absolute_error(y_nv_ood, y_pred_nv_reg)
    r2_nv = r2_score(y_nv_ood, y_pred_nv_reg)
    print(f"Linear Probe Extrapolation MAE: {mae_nv:.2f} vertices")
    print(f"Linear Probe Extrapolation R^2 Score: {r2_nv:.2f}")

    print("\nSample Extrapolation Predictions vs Ground Truth (nv):")
    for i in range(min(10, len(y_nv_ood))):
        print(f"  True nv: {y_nv_ood[i]} | Extrapolated Predicted nv: {y_pred_nv_reg[i]:.2f}")

    # C. NUMBER OF COLORS (Continuous Extrapolation via Ridge Linear Probe)
    print("\n-------------------------------------------------------")
    print(" Target: NUMBER OF COLORS (nc) [OOD Test Range: 17 to 25]")
    print(" Training Range: 3 to 15")
    print("-------------------------------------------------------")
    reg_nc = Ridge(alpha=1.0)
    reg_nc.fit(X_train, y_nc_train)
    y_pred_nc_reg = reg_nc.predict(X_ood)
    mae_nc = mean_absolute_error(y_nc_ood, y_pred_nc_reg)
    r2_nc = r2_score(y_nc_ood, y_pred_nc_reg)
    print(f"Linear Probe Extrapolation MAE: {mae_nc:.2f} colors")
    print(f"Linear Probe Extrapolation R^2 Score: {r2_nc:.2f}")

    print("\nSample Extrapolation Predictions vs Ground Truth (nc):")
    for i in range(min(10, len(y_nc_ood))):
        print(f"  True nc: {y_nc_ood[i]} | Extrapolated Predicted nc: {y_pred_nc_reg[i]:.2f}")

if __name__ == "__main__":
    main()
