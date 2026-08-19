import os
import argparse
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import Dataset
from torch_geometric.nn.aggr import scaler
from models import GraphToTextConditionalGeneration, train_and_get_tokenizer
from tokenizers import Tokenizer
from tqdm import tqdm

def save_checkpoint(model, optimizer, scaler, epoch, step, loss, filepath="checkpoint.pt"):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        # 'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'loss': loss,
        'decoder_config': model.decoder_config # Saves architecture metadata
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, model, optimizer=None, device="cpu"):
    """
    Loads a saved model/checkpoint file to restore model weights, optimizer states, and epoch.
    Supports both checkpoint dictionaries and direct model state_dicts.
    """
    if not os.path.exists(filepath):
        # Search fallback relative paths
        for alt_path in [os.path.join("models", filepath), os.path.join("../models", filepath)]:
            if os.path.exists(alt_path):
                filepath = alt_path
                break
        else:
            raise FileNotFoundError(f"Checkpoint file not found: '{filepath}'")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    epoch = 0
    step = 0
    loss = float("inf")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Could not load optimizer state dict: {e}")
        epoch = checkpoint.get("epoch", 0)
        step = checkpoint.get("step", 0)
        loss = checkpoint.get("loss", float("inf"))
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
    print(f"Loaded checkpoint from '{filepath}' (Epoch: {epoch}, Prev Loss: {loss_str})")
    return epoch, step, loss

class GraphTextDataset(Dataset):

    def __init__(self, graph_list, text_list, tokenizer, max_length=2096):
        """graph_list: List of dictionaries returned by your parsing function.

        text_list: List of corresponding strings.
        """
        self.graphs = graph_list
        self.texts = text_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(length=self.max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        text = self.texts[idx].strip()
        if not text.startswith("[BOS]"):
            text = "[BOS] " + text
        if not text.endswith("[EOS]"):
            text = text + " [EOS]"

        # Wrap into PyTorch Geometric Data object
        graph_data = Data(
            x=torch.tensor(g["x"], dtype=torch.float),
            edge_index=torch.tensor(g["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(g["edge_attr"], dtype=torch.float),
        )
        output = self.tokenizer.encode(text)
        input_ids = torch.tensor(output.ids)
        attention_mask = torch.tensor(output.attention_mask)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return graph_data, input_ids, attention_mask, labels

def load_combined_dataset(datasets=None, max_per_dataset=None, val_ratio=0.10):
    """
    Loads dataset samples combining graph_coloring and knapsack (or any specified datasets).
    Splits the last `val_ratio` (e.g., 10%) of instances from each dataset into validation sets.
    """
    import random
    from load_graph import load_graph_to_features

    if datasets is None:
        datasets = [
            {
                "name": "graph_coloring",
                "flat_dirs": ["data/graph_coloring_flat", "graph_coloring_flat"],
                "graph_dirs": ["data/graph_coloring_graphs", "graph_coloring_graphs"]
            },
            {
                "name": "knapsack",
                "flat_dirs": ["data/knapsack_flat", "knapsack_flat"],
                "graph_dirs": ["data/knapsack_graphs", "knapsack_graphs"]
            },
            {
                "name": "tsp",
                "flat_dirs": ["data/tsp_flat", "tsp_flat"],
                "graph_dirs": ["data/tsp_graphs", "tsp_graphs"]
            }
        ]

    train_graphs, train_texts = [], []
    val_graphs, val_texts = [], []
    corpus_folders = []

    for ds in datasets:
        flat_dir = next((d for d in ds["flat_dirs"] if os.path.exists(d)), None)
        graph_dir = next((d for d in ds["graph_dirs"] if os.path.exists(d)), None)

        if not flat_dir or not graph_dir:
            print(f"Skipping dataset {ds['name']}: directories not found.")
            continue

        corpus_folders.append(flat_dir)
        files = [f for f in sorted(os.listdir(flat_dir)) if f.endswith(".fzn")]
        if max_per_dataset:
            files = files[:max_per_dataset]

        ds_graphs, ds_texts = [], []
        for file in files:
            graph_file = os.path.join(graph_dir, file.replace(".fzn", ".graph"))
            flat_file = os.path.join(flat_dir, file)
            if not os.path.exists(graph_file):
                continue
            try:
                graph = load_graph_to_features(graph_file)
                with open(flat_file, "r") as f:
                    text = f.read()
                ds_graphs.append(graph)
                ds_texts.append(text)
            except Exception:
                pass

        total_ds = len(ds_graphs)
        if total_ds == 0:
            continue

        val_count = int(total_ds * val_ratio)
        train_count = total_ds - val_count

        ds_train_g, ds_train_t = ds_graphs[:train_count], ds_texts[:train_count]
        ds_val_g, ds_val_t = ds_graphs[train_count:], ds_texts[train_count:]

        train_graphs.extend(ds_train_g)
        train_texts.extend(ds_train_t)
        val_graphs.extend(ds_val_g)
        val_texts.extend(ds_val_t)

        print(f"Loaded {ds['name']} ({flat_dir}): {len(ds_train_g)} train, {len(ds_val_g)} val (total {total_ds}).")

    # Shuffle training set and validation set independently
    random.seed(42)
    combined_train = list(zip(train_graphs, train_texts))
    random.shuffle(combined_train)
    if combined_train:
        train_graphs, train_texts = zip(*combined_train)
        train_graphs, train_texts = list(train_graphs), list(train_texts)

    combined_val = list(zip(val_graphs, val_texts))
    random.shuffle(combined_val)
    if combined_val:
        val_graphs, val_texts = zip(*combined_val)
        val_graphs, val_texts = list(val_graphs), list(val_texts)

    return (train_graphs, train_texts), (val_graphs, val_texts), corpus_folders


def train_generation_loop(train_data, val_data, epochs=5, batch_size=8, lr=2e-5, corpus_folders=None, resume_path=None):
    train_graphs, train_texts = train_data
    val_graphs, val_texts = val_data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    if corpus_folders is None:
        corpus_folders = ["data/graph_coloring_flat", "data/knapsack_flat", "data/tsp_flat"]

    tokenizer = train_and_get_tokenizer(10000, corpus_folders=corpus_folders, force_retrain=True)

    node_in_dim = len(train_graphs[0]["x"][0])
    edge_in_dim = len(train_graphs[0]["edge_attr"][0])

    model = GraphToTextConditionalGeneration(
        node_in_dim, 
        edge_in_dim, 
        10000, 
        tokenizer.token_to_id("[BOS]"), 
        tokenizer.token_to_id("[EOS]"), 
        tokenizer.token_to_id("[PAD]")
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}")
        start_epoch, _, prev_loss = load_checkpoint(resume_path, model, optimizer, device=device)
        if isinstance(prev_loss, (int, float)) and prev_loss != float("inf"):
            best_val_loss = prev_loss
        if start_epoch >= epochs:
            print(f"Notice: Checkpoint epoch ({start_epoch}) >= target epochs ({epochs}). Extending total epochs to {start_epoch + epochs}.")
            epochs = start_epoch + epochs

    enc_params = sum(p.numel() for p in model.graph_encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_params = enc_params + dec_params
    print(f'Encoder parameters: {enc_params / 1e6:.2f}M')
    print(f'Decoder parameters: {dec_params / 1e6:.2f}M')
    print(f'Total model parameters: {total_params / 1e6:.2f}M')
    
    train_dataset = GraphTextDataset(train_graphs, train_texts, tokenizer=tokenizer, max_length=model.decoder_config.n_positions)
    train_loader = PyGDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_graphs and val_texts:
        val_dataset = GraphTextDataset(val_graphs, val_texts, tokenizer=tokenizer, max_length=model.decoder_config.n_positions)
        val_loader = PyGDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(start_epoch, epochs):
        model.train()
        epoch_loss = 0.0
        for graph_batch, input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            graph_batch = graph_batch.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(
                graph_batch=graph_batch,
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for graph_batch, input_ids, attention_mask, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    graph_batch = graph_batch.to(device)
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)

                    outputs = model(
                        graph_batch=graph_batch,
                        decoder_input_ids=input_ids,
                        decoder_attention_mask=attention_mask,
                        labels=labels
                    )
                    val_loss += outputs.loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_checkpoint(model, optimizer, None, epoch + 1, 0, avg_val_loss, "best_model_mixed.pt")
                print(f"--> Saved best model checkpoint to best_model_mixed.pt (Val Loss: {best_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
            if (epoch + 1) % 10 == 0:
                save_checkpoint(model, optimizer, None, epoch + 1, 0, avg_train_loss, f"checkpoint_mixed_epoch_{epoch+1}.pt")

    save_checkpoint(model, optimizer, None, epochs, 0, avg_train_loss, "final_model_mixed.pt")
    print("Successfully saved final trained model to final_model_mixed.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or resume training GraphToText model.")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Path to saved model checkpoint to resume training from.")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of epochs to train (default: 100).")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training (default: 16).")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5).")
    parser.add_argument("--max_per_dataset", type=int, default=None, help="Max instances per dataset (default: None).")
    parser.add_argument("--val_ratio", type=float, default=0.10, help="Validation ratio (default: 0.10).")

    args = parser.parse_args()

    train_data, val_data, corpus_folders = load_combined_dataset(
        max_per_dataset=args.max_per_dataset, 
        val_ratio=args.val_ratio
    )
    print(f"Total instances loaded - Train: {len(train_data[0])}, Val: {len(val_data[0])}")
    if train_data[0]:
        train_generation_loop(
            train_data, 
            val_data, 
            epochs=args.epochs, 
            batch_size=args.batch_size, 
            lr=args.lr,
            corpus_folders=corpus_folders,
            resume_path=args.resume
        )