import torch
from torch.optim import optimizer
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
        'optimizer_state_dict': optimizer.state_dict(),
        # 'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'loss': loss,
        'decoder_config': model.decoder_config # Saves architecture metadata
    }
    torch.save(checkpoint, filepath)

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

def train_generation_loop(graph_list, text_list, epochs=5, batch_size=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    tokenizer = train_and_get_tokenizer(10000)

    node_in_dim = len(graph_list[0]["x"][0])
    edge_in_dim = len(graph_list[0]["edge_attr"][0])

    model = GraphToTextConditionalGeneration(node_in_dim, 
                                             edge_in_dim, 
                                             10000, 
                                             tokenizer.token_to_id("[BOS]"), 
                                             tokenizer.token_to_id("[EOS]"), 
                                             tokenizer.token_to_id("[PAD]")
                                             ).to(device)
    enc_params = sum(p.numel() for p in model.graph_encoder.parameters() if p.requires_grad)
    dec_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    total_params = enc_params + dec_params
    print(f'Encoder parameters: {enc_params / 1e6:.2f}M')
    print(f'Decoder parameters: {dec_params / 1e6:.2f}M')
    print(f'Total model parameters: {total_params / 1e6:.2f}M')

    dataset = GraphTextDataset(graph_list, text_list, tokenizer=tokenizer, max_length=model.decoder_config.n_positions)
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for graph_batch, input_ids, attention_mask, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
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
            
        avg_loss = epoch_loss / len(loader)
#        save_checkpoint(model, optimizer, None, epoch + 1, 0, avg_loss, f"checkpoint_epoch_{epoch+1}.pt")
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, None, epoch + 1, 0, avg_loss, f"checkpoint_epoch_{epoch+1}.pt")
        print(f"Epoch {epoch+1}/{epochs} - Generation Loss: {avg_loss:.4f}")

    save_checkpoint(model, optimizer, None, epochs, 0, avg_loss, "final_model_80m.pt")
    print("Successfully saved final trained model to final_model_80m.pt")

if __name__ == "__main__":
    import os 
    from load_graph import load_graph_to_features
    graph_list, text_list = [], []
    for file in sorted(os.listdir("flat")):
        if not file.endswith(".fzn"):
            continue
        try:
            graph = load_graph_to_features("graphs/" + file.replace(".fzn", ".graph"))
            with open("flat/" + file) as f:
                text = f.read()
            graph_list.append(graph)
            text_list.append(text)
#            if len(graph_list) >= 1000:
#                break
        except Exception as e:
            pass
    print(f"Loaded {len(graph_list)} FlatZinc models for training.")
    train_generation_loop(graph_list, text_list, epochs=100, batch_size=16)
