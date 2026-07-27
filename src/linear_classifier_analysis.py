import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

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

def load_dataset(graph_dir, max_samples=None):
    if not os.path.exists(graph_dir):
        raise FileNotFoundError(f"Directory {graph_dir} not found.")

    all_files = [os.path.join(graph_dir, f) for f in sorted(os.listdir(graph_dir)) if f.endswith(".graph")]
    if max_samples:
        all_files = all_files[:max_samples]

    records = []
    print(f"Extracting tabular features from {len(all_files)} files in '{graph_dir}'...")
    for path in tqdm(all_files):
        feat = parse_graph_tabular_features(path)
        if feat is not None:
            records.append(feat)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} valid records. SAT distribution: {dict(df['sat'].value_counts())}")
    return df

def evaluate_classifier_performance(model, model_name, X_train, y_train, X_id, y_id, X_ood, y_ood, feature_names):
    print(f"\n========================================================================")
    print(f"  MODEL: {model_name}")
    print(f"  Features used: {feature_names}")
    print(f"========================================================================")

    # 1. 5-Fold Cross Validation on Training Set
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

    # 2. Fit model on full training set
    model.fit(X_train, y_train)

    # Output linear weights if available
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

    # 3. Evaluate on In-Distribution (ID) Test Set
    y_pred_id = model.predict(X_id)
    id_metrics = compute_metrics(y_id, y_pred_id, "In-Distribution (ID) Test Set")

    # 4. Evaluate on Out-of-Distribution (OOD) Test Set
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

def compute_metrics(y_true, y_pred, dataset_name):
    acc = accuracy_score(y_true, y_pred) * 100
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Class-specific recall & precision (UNSAT = 0, SAT = 1)
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

def main():
    train_dir = "data/graphs" if os.path.exists("data/graphs") else "graphs"
    id_test_dir = "data/test_graphs" if os.path.exists("data/test_graphs") else "test_graphs"
    ood_test_dir = "data/ood_test_graphs" if os.path.exists("data/ood_test_graphs") else "ood_test_graphs"

    print("Loading Datasets...")
    df_train = load_dataset(train_dir, max_samples=2500)
    df_id = load_dataset(id_test_dir, max_samples=1000)
    df_ood = load_dataset(ood_test_dir, max_samples=1000)

    # 1. Majority Classifier Baseline
    y_id = df_id['sat'].to_numpy()
    y_ood = df_ood['sat'].to_numpy()

    maj_pred_id = np.ones_like(y_id) # Predict SAT for all
    maj_pred_ood = np.ones_like(y_ood)

    print("\n========================================================================")
    print("  BASELINES: Majority Class Predictor (Always SAT)")
    print("========================================================================")
    compute_metrics(y_id, maj_pred_id, "Majority Baseline on ID Test Set")
    compute_metrics(y_ood, maj_pred_ood, "Majority Baseline on OOD Test Set")

    # Feature sets
    feature_sets = [
        ("Primary [num_nodes, num_edges, nc]", ["num_nodes", "num_edges", "nc"]),
        ("Alternative [nv, num_edges, nc]", ["nv", "num_edges", "nc"]),
        ("Full [nv, num_nodes, num_edges, nc]", ["nv", "num_nodes", "num_edges", "nc"])
    ]

    all_results = []

    for name, cols in feature_sets:
        X_train = df_train[cols].to_numpy()
        y_train = df_train['sat'].to_numpy()

        X_id = df_id[cols].to_numpy()
        X_ood = df_ood[cols].to_numpy()

        # Model 1: Logistic Regression (Standardized)
        lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(random_state=42))
        res_lr = evaluate_classifier_performance(
            lr_pipe, f"Logistic Regression ({name})",
            X_train, y_train, X_id, y_id, X_ood, y_ood, cols
        )
        all_results.append(res_lr)

        # Model 2: Linear Discriminant Analysis (LDA)
        lda_pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
        res_lda = evaluate_classifier_performance(
            lda_pipe, f"LDA ({name})",
            X_train, y_train, X_id, y_id, X_ood, y_ood, cols
        )
        all_results.append(res_lda)

        # Model 3: Linear SVM
        svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='linear', random_state=42))
        res_svm = evaluate_classifier_performance(
            svm_pipe, f"Linear SVM ({name})",
            X_train, y_train, X_id, y_id, X_ood, y_ood, cols
        )
        all_results.append(res_svm)

        # Model 4: Random Forest
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        res_rf = evaluate_classifier_performance(
            rf_clf, f"Random Forest ({name})",
            X_train, y_train, X_id, y_id, X_ood, y_ood, cols
        )
        # Print RF Feature Importances
        importances = rf_clf.feature_importances_
        imp_str = ", ".join([f"{col}: {imp*100:.2f}%" for col, imp in zip(cols, importances)])
        print(f"Random Forest Feature Importances: {imp_str}")
        all_results.append(res_rf)

    print("\n\n" + "="*90)
    print(" SUMMARY COMPARATIVE TABLE: TABULAR LINEAR CLASSIFIER vs BASELINES")
    print("="*90)
    summary_data = []
    
    # Add GNN + LDA reference stats from report
    gnn_lda_id_acc, gnn_lda_id_bal, gnn_lda_id_f1, gnn_lda_id_unsat_rec = 96.35, 95.10, 0.9710, 91.20
    gnn_lda_ood_acc, gnn_lda_ood_bal, gnn_lda_ood_f1, gnn_lda_ood_unsat_rec = 96.00, 93.50, 0.8200, 80.00 # Approx from LDA OOD

    for r in all_results:
        summary_data.append({
            "Classifier": r["model_name"],
            "5-Fold CV Acc": f"{r['cv_acc']:.2f}%",
            "ID Test Acc": f"{r['id_metrics']['acc']:.2f}%",
            "ID Bal Acc": f"{r['id_metrics']['bal_acc']:.2f}%",
            "ID UNSAT Rec": f"{r['id_metrics']['unsat_rec']:.2f}%",
            "ID Macro F1": f"{r['id_metrics']['macro_f1']:.4f}",
            "OOD Test Acc": f"{r['ood_metrics']['acc']:.2f}%",
            "OOD Bal Acc": f"{r['ood_metrics']['bal_acc']:.2f}%",
            "OOD UNSAT Rec": f"{r['ood_metrics']['unsat_rec']:.2f}%",
            "OOD Macro F1": f"{r['ood_metrics']['macro_f1']:.4f}",
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
