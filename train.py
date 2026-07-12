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

    def __init__(self, graph_list, text_list, tokenizer, max_length=512):
        """graph_list: List of dictionaries returned by your parsing function.

        text_list: List of corresponding strings.
        """
        self.graphs = graph_list
        self.texts = text_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(max_length=self.max_length)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        g = self.graphs[idx]
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
    total_params = sum(p.numel() for p in model.graph_encoder.parameters() if p.requires_grad)
    print(f'Total number of encoder parameters: {total_params}')
    total_params = sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)
    print(f'Total number of decoder parameters: {total_params}')
    # raise Exception()
    dataset = GraphTextDataset(graph_list, text_list, tokenizer=tokenizer, max_length=model.decoder_config.n_positions)
    loader = PyGDataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for graph_batch, input_ids, attention_mask, labels in tqdm(loader):
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
            
        save_checkpoint(model, optimizer, scaler, epoch, 0, loss, f"checkpoint_{epoch}.pt")
        print(f"Epoch {epoch+1} - Generation Loss: {epoch_loss / len(loader):.4f}")

if __name__ == "__main__":
    import os 
    from load_graph import load_graph_to_features
    graph_list, text_list = [], []
    for file in os.listdir("flat"):
        try:
            graph = load_graph_to_features("graphs/" + file.replace("fzn", "graph"))
            f = open("flat/" + file)
            text = f.read()
            f.close()
            graph_list.append(graph)
            text_list.append(text)
            if len(graph_list) >= 150:
                break
            # print(file)
            # raise Exception()
        except:
            pass
    # print("loaded all")
    # print(len(graph_list))
    train_generation_loop(graph_list, text_list, epochs=10)
