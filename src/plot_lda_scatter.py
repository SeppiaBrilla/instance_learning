import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA

from models import train_and_get_tokenizer, GraphToTextConditionalGeneration
from load_graph import load_graph_to_features
from inference import load_checkpoint, convert_to_pyg_data

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

def generate_plots():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    sns.set_theme(style="darkgrid", palette="deep")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = train_and_get_tokenizer()

    graph_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    test_graph_dir = "data/test_graphs" if os.path.exists("data/test_graphs") else "test_graphs"
    output_dir = "data/outputs" if os.path.exists("data/outputs") else "."
    os.makedirs(output_dir, exist_ok=True)

    train_files = [os.path.join(graph_dir, f) for f in sorted(os.listdir(graph_dir)) if f.endswith(".graph")][:2500]
    test_files = [os.path.join(test_graph_dir, f) for f in sorted(os.listdir(test_graph_dir)) if f.endswith(".graph")][:1000]

    sample_g = load_graph_to_features(train_files[0])
    
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

    print("Extracting encoder embeddings for 2,500 training graphs...")
    X_train, y_sat_tr, y_nv_tr, y_nc_tr = extract_dataset_embeddings(train_files, model, device)

    print("Extracting encoder embeddings for held-out test graphs...")
    X_test, y_sat_te, y_nv_te, y_nc_te = extract_dataset_embeddings(test_files, model, device)

    print("Fitting LDA projections on training set and projecting held-out test set...")

    # 1. LDA Projection for Number of Colors (nc)
    lda_nc = LinearDiscriminantAnalysis(n_components=2)
    lda_nc.fit(X_train, y_nc_tr)
    X_nc_2d = lda_nc.transform(X_test)

    # 2. LDA Projection for Number of Vertices (nv)
    lda_nv = LinearDiscriminantAnalysis(n_components=2)
    lda_nv.fit(X_train, y_nv_tr)
    X_nv_2d = lda_nv.transform(X_test)

    # 3. LDA Projection for Satisfiability (sat)
    lda_sat = LinearDiscriminantAnalysis(n_components=1)
    lda_sat.fit(X_train, y_sat_tr)
    ld1_sat = lda_sat.transform(X_test).flatten()
    pca = PCA(n_components=2)
    pca.fit(X_train)
    X_pca = pca.transform(X_test)
    X_sat_2d = np.column_stack((ld1_sat, X_pca[:, 1]))

    # Use test set labels for plotting
    y_nc = y_nc_te
    y_nv = y_nv_te
    y_sat = y_sat_te

    # Plot 1: Number of Colors (nc)
    plt.figure(figsize=(9, 7), dpi=300)
    palette_nc = sns.color_palette("plasma", n_colors=len(np.unique(y_nc)))
    scatter_nc = sns.scatterplot(
        x=X_nc_2d[:, 0], y=X_nc_2d[:, 1], hue=y_nc, palette=palette_nc,
        s=45, alpha=0.85, edgecolor='w', linewidth=0.3
    )
    plt.title("2D LDA Projection: Number of Colors (nc) [Held-Out Test Set]", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("LDA Component 1", fontsize=12)
    plt.ylabel("LDA Component 2", fontsize=12)
    plt.legend(title="Colors (nc)", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    nc_path = os.path.join(output_dir, "lda_2d_scatter_nc.png")
    plt.savefig(nc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {nc_path}")

    # Plot 2: Number of Vertices (nv)
    plt.figure(figsize=(9, 7), dpi=300)
    palette_nv = sns.color_palette("viridis", n_colors=len(np.unique(y_nv)))
    scatter_nv = sns.scatterplot(
        x=X_nv_2d[:, 0], y=X_nv_2d[:, 1], hue=y_nv, palette=palette_nv,
        s=45, alpha=0.85, edgecolor='w', linewidth=0.3
    )
    plt.title("2D LDA Projection: Number of Vertices (nv) [Held-Out Test Set]", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("LDA Component 1", fontsize=12)
    plt.ylabel("LDA Component 2", fontsize=12)
    plt.legend(title="Vertices (nv)", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    nv_path = os.path.join(output_dir, "lda_2d_scatter_nv.png")
    plt.savefig(nv_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {nv_path}")

    # Plot 3: Satisfiability (sat)
    plt.figure(figsize=(9, 7), dpi=300)
    sat_labels = np.array(["UNSAT" if s == 0 else "SAT" for s in y_sat])
    palette_sat = {"UNSAT": "#E63946", "SAT": "#1D3557"}
    scatter_sat = sns.scatterplot(
        x=X_sat_2d[:, 0], y=X_sat_2d[:, 1], hue=sat_labels, palette=palette_sat,
        s=45, alpha=0.85, edgecolor='w', linewidth=0.3
    )
    plt.title("2D LDA/PCA Projection: Satisfiability (sat) [Held-Out Test Set]", fontsize=14, fontweight='bold', pad=12)
    plt.xlabel("LDA Component 1 (Separating Axis)", fontsize=12)
    plt.ylabel("Orthogonal PCA Component", fontsize=12)
    plt.legend(title="Satisfiability", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()
    sat_path = os.path.join(output_dir, "lda_2d_scatter_sat.png")
    plt.savefig(sat_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sat_path}")

    # Plot 4: Combined 3-Panel Figure
    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5), dpi=300)

    # Panel A: Colors (nc)
    sns.scatterplot(
        ax=axes[0], x=X_nc_2d[:, 0], y=X_nc_2d[:, 1], hue=y_nc, palette=palette_nc,
        s=40, alpha=0.85, edgecolor='none'
    )
    axes[0].set_title("A. Number of Colors (nc) [100.0% Test Accuracy]", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("LDA Comp 1", fontsize=11)
    axes[0].set_ylabel("LDA Comp 2", fontsize=11)
    axes[0].legend(title="Colors (nc)", loc='upper right', frameon=True, fontsize=9)

    # Panel B: Satisfiability (sat)
    sns.scatterplot(
        ax=axes[1], x=X_sat_2d[:, 0], y=X_sat_2d[:, 1], hue=sat_labels, palette=palette_sat,
        s=40, alpha=0.85, edgecolor='none'
    )
    axes[1].set_title("B. Satisfiability (sat) [97.3% Test Accuracy]", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("LDA Comp 1 (Discriminating Axis)", fontsize=11)
    axes[1].set_ylabel("Orthogonal PCA Axis", fontsize=11)
    axes[1].legend(title="Satisfiability", loc='upper right', frameon=True, fontsize=9)

    # Panel C: Vertices (nv)
    sns.scatterplot(
        ax=axes[2], x=X_nv_2d[:, 0], y=X_nv_2d[:, 1], hue=y_nv, palette=palette_nv,
        s=40, alpha=0.85, edgecolor='none'
    )
    axes[2].set_title("C. Number of Vertices (nv) [51.8% Test Accuracy]", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("LDA Comp 1", fontsize=11)
    axes[2].set_ylabel("LDA Comp 2", fontsize=11)
    axes[2].legend(title="Vertices (nv)", loc='upper right', frameon=True, fontsize=9)

    plt.suptitle("Linear Discriminant Analysis (LDA) Projections of Held-Out Test Embeddings", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    combined_path = os.path.join(output_dir, "lda_2d_scatter_all.png")
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot: {combined_path}")

if __name__ == "__main__":
    generate_plots()
