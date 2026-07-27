# Dual-Graph Latent Fusion & Graph-to-Text Constraint Generation

This repository contains the full source code, datasets, trained model weights, evaluation scripts, and analysis reports for **Dual-Graph Latent Space Fusion and Representation Learning** on Graph Coloring Constraint Problems.

---

## 📁 Repository Structure

```
instance_learning/
├── src/                                # Core Source Code & Executable Scripts
│   ├── models.py                       # GINE Encoder & GPT-2 Decoder Architecture
│   ├── load_graph.py                   # PyTorch Geometric Graph Loader & Parser
│   ├── generator.py                    # MiniZinc Graph Generator & FlatZinc Parser
│   ├── flattening.py                   # FlatZinc AST / String Utilities
│   ├── train.py                        # Model Training Loop Script
│   ├── inference.py                    # Dual-Graph Latent Mean-Pooling Inference
│   ├── lda_classification.py           # LDA Linear Discriminant Probes & Cross-Validation
│   ├── plot_lda_scatter.py             # 2D LDA Latent Space Scatter Generator
│   ├── ood_evaluation.py               # Out-of-Distribution (OOD) Extrapolation Benchmark
│   └── generate_100_annotated_instances.py # Annotated 100 Instance Generator & Analyzer
│
├── data/                               # Datasets & Generated Artifacts
│   ├── graphs/                         # 10,000 Training Graphs (.graph)
│   ├── flat/                           # 10,000 Training FlatZinc Instances (.fzn)
│   ├── instances/                      # MiniZinc Data Files (.mzn / .dzn)
│   ├── test_graphs/                    # In-Distribution Test Graphs (301 instances)
│   ├── test_instances/                 # In-Distribution Test FlatZinc Files
│   ├── ood_test_graphs/                # Out-of-Distribution Test Graphs (250 instances)
│   ├── ood_test_instances/             # Out-of-Distribution Test FlatZinc Files
│   ├── generated_100_instances/        # Raw Generated 100 Dual-Graph Instances
│   ├── annotated_100_instances/        # Fully Annotated 100 Dual-Graph Instances
│   └── outputs/                        # Statistics, CSV Summaries, and LDA Plots
│       ├── baseline_10k_distribution.csv
│       ├── generation_pattern_analysis.csv
│       ├── lda_2d_scatter_all.png
│       ├── lda_2d_scatter_nc.png
│       ├── lda_2d_scatter_nv.png
│       └── lda_2d_scatter_sat.png
│
├── models/                             # Checkpoints & Tokenizer
│   ├── final_model_80m.pt              # Trained 80M Parameter Checkpoint
│   └── tokenizer/                      # Trained BPE Tokenizer Vocabulary
│
└── graph_encoder_lda_report.md         # Full Technical & Statistical Report
```

---

## 🚀 Quick Start & Usage

Ensure MiniZinc is in your environment PATH:
```bash
export PATH="/work/minizinc_bundle/bin:$PATH"
export PYTHONPATH=src
```

### 1. Dual-Graph Latent Mean-Pooling Inference
Generate a constraint FlatZinc file by interpolating between two source graphs $G_1$ and $G_2$:
```bash
python src/inference.py data/graphs/instance_0.graph data/graphs/instance_1.graph
```

### 2. LDA Latent Representation Probing
Evaluate linear separability of GNN encoder representations for $nc$, $nv$, and $sat$:
```bash
python src/lda_classification.py
```

### 3. Generate 2D Scatter Projections
Generate 2D visualization scatter plots saved to `data/outputs/`:
```bash
python src/plot_lda_scatter.py
```

### 4. Out-of-Distribution (OOD) Benchmark Evaluation
Evaluate extrapolation on larger/denser graphs ($nv \in [22..30], nc \in [17..25]$):
```bash
python src/ood_evaluation.py
```

### 5. Generate Annotated Batch (100 Instances)
Generate 100 unique annotated FlatZinc instances with solver verification & statistical pattern analysis:
```bash
python src/generate_100_annotated_instances.py
```

---

## 📊 Summary of Results

- **Color Bounds ($nc$) Probing**: **100.00% Accuracy** (Linear separability across 7 classes).
- **Satisfiability ($sat$) Probing**: **96.68% Accuracy** (Vs. $50.00\%$ balanced accuracy of majority baseline, **91.20% UNSAT recall**).
- **OOD Monotonic Extrapolation**: Spearman **$\rho = 0.9677$** on unseen color bounds ($nc \in [17, 25]$).
- **Dual-Graph Generation**: **100% unique instances** (whitespace-insensitive) matching baseline SAT rate ($84.0\%$ vs $81.1\%$).
