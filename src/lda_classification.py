import os
import re
import torch
import numpy as np
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from inference import load_checkpoint, convert_to_pyg_data

def parse_header(file_path):
    """
    Parses metadata header like: %nc: 15, nv: 22, sat: true
    """
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    if not first_line.startswith("%nc:"):
        return None
    
    # Example format: %nc: 15, nv: 22, sat: true
    match = re.search(r"nc:\s*(\d+),\s*nv:\s*(\d+),\s*sat:\s*(true|false)", first_line, re.IGNORECASE)
    if not match:
        return None
    
    nc = int(match.group(1))
    nv = int(match.group(2))
    sat = match.group(3).lower() == "true"
    return {"nc": nc, "nv": nv, "sat": sat}

@torch.no_grad()
def extract_embeddings_and_labels(graph_dir=None, max_samples=3000):
    if graph_dir is None:
        graph_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model sample to initialize architecture
    tokenizer = train_and_get_tokenizer()
    
    all_files = [f for f in sorted(os.listdir(graph_dir)) if f.endswith(".graph")]
    valid_files = []
    metadata_list = []

    print("Scanning graph files for metadata...")
    for file in all_files:
        path = os.path.join(graph_dir, file)
        meta = parse_header(path)
        if meta is not None:
            valid_files.append(path)
            metadata_list.append(meta)
        if len(valid_files) >= max_samples:
            break

    print(f"Found {len(valid_files)} graphs with valid metadata.")
    
    sample_g = load_graph_to_features(valid_files[0])
    node_in_dim = len(sample_g["x"][0])
    edge_in_dim = len(sample_g["edge_attr"][0])

    model = GraphToTextConditionalGeneration(
        node_in_dim, 
        edge_in_dim, 
        10000, 
        tokenizer.token_to_id("[BOS]"), 
        tokenizer.token_to_id("[EOS]"), 
        tokenizer.token_to_id("[PAD]")
    ).to(device)

    model, _, _, _, _, _ = load_checkpoint("final_model_80m.pt", model, None, None)
    model.eval()

    embeddings = []
    y_sat = []
    y_nv = []
    y_nc = []

    print("Extracting encoder embeddings...")
    for path, meta in zip(tqdm(valid_files), metadata_list):
        graph_dict = load_graph_to_features(path)
        pyg_data = convert_to_pyg_data(graph_dict).to(device)
        if not hasattr(pyg_data, 'batch') or pyg_data.batch is None:
            pyg_data.batch = torch.zeros(pyg_data.num_nodes, dtype=torch.long, device=device)
        setattr(pyg_data, 'num_graphs', 1)

        node_embeddings = model.graph_encoder(pyg_data) # [num_nodes, hidden_dim]
        graph_embedding = node_embeddings.mean(dim=0).cpu().numpy() # Mean pooling

        embeddings.append(graph_embedding)
        y_sat.append(meta["sat"])
        y_nv.append(meta["nv"])
        y_nc.append(meta["nc"])

    X = np.array(embeddings)
    y_sat = np.array(y_sat, dtype=int)
    y_nv = np.array(y_nv, dtype=int)
    y_nc = np.array(y_nc, dtype=int)

    return X, y_sat, y_nv, y_nc

def evaluate_lda(X_train, y_train, X_test, y_test, target_name):
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

if __name__ == "__main__":
    train_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    test_dir = "data/test_graphs" if os.path.exists("data/test_graphs") else "test_graphs"

    print("Extracting training set embeddings...")
    X_train, y_sat_tr, y_nv_tr, y_nc_tr = extract_embeddings_and_labels(graph_dir=train_dir, max_samples=2500)

    print("\nExtracting held-out in-distribution test set embeddings...")
    X_test, y_sat_te, y_nv_te, y_nc_te = extract_embeddings_and_labels(graph_dir=test_dir, max_samples=1000)

    evaluate_lda(X_train, y_sat_tr, X_test, y_sat_te, "SATISFIABILITY (sat)")
    evaluate_lda(X_train, y_nv_tr, X_test, y_nv_te, "NUMBER OF VERTICES (nv)")
    evaluate_lda(X_train, y_nc_tr, X_test, y_nc_te, "NUMBER OF COLORS (nc)")
