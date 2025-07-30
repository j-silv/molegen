
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch
from . import data as prepare_data
from .model import MoleGen

class DataFrameDataset(Dataset):
    """Wrapper class to use Pytorch DataLoader"""
    def __init__(self, df, colname="data"):
        self.data = df[colname].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
      
def main():
    
    ###################################################
    # Parameters
    DEBUG = True 
    embd = 16 # embedding size of a vocab indice
    num_layers = 1     # number of GCN layers
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-08
    epochs = 100
    lambda_boa = 0.5
    lambda_edge = 0.5
    batch_size = 32
    ###################################################
    
    torch.manual_seed(42)
    
    df, a2t, t2a, e2t, t2e, max_atoms = prepare_data.main()
    vocab_size = len(a2t)

    dataset = DataFrameDataset(df, "data")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    data = next(iter(loader))
    print(data)
    
    model = MoleGen(vocab_size=vocab_size, num_layers=num_layers, embd=embd, max_atoms=max_atoms)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(model)

    model.eval()
    for epoch in range(epochs):
        loss_avg = 0.0
        
        for idx, batch in enumerate(loader):
            boa, z, s = model(batch)
            
            if DEBUG:    
                boa_actual = batch.y_boa[0]
                boa_pred = torch.argmax(boa[0], dim=-1)
                
                for token, (count, pred_count) in enumerate(zip(boa_actual, boa_pred)):
                    print(f"{t2a[token]}({count}, {pred_count}) | ", end="")
                print(": ATOM(ACTUAL, PRED) - graph", idx)
                
                
                # select only edges that correspond to first graph 
                edge = batch.batch[batch.fc_edge_index] # (num_edges, )
                edge = torch.arange((edge[edge == 0]).shape[0]) # (num_edges_graph0, )
                edge_pred = s[edge] # (num_edges_graph0, C)
                edge_pred = torch.argmax(edge_pred, dim=-1) # (num_edges_graph0, )
                
                src_atoms = batch.fc_edge_index[0, edge] # (num_edges_graph0, )
                dest_atoms = batch.fc_edge_index[1, edge] # (num_edges_graph0, )
                src_atom_feats = batch.x[src_atoms].squeeze(-1) # (num_nodes_graph0, )
                dest_atom_feats = batch.x[dest_atoms].squeeze(-1) # (num_nodes_graph0, )

                # only show the edges that are not zero because we would have too many otherwise
                print("\n[ATOM1][ATOM2] (ACTUAL, PRED) - graph", idx)
                for num, (src, dest, actual, predicted) in enumerate(zip(src_atom_feats, dest_atom_feats, batch.y_fc_edge_attr, edge_pred)):
                    
                    if actual != 0 and predicted != 0:
                        print(f"[{t2a[src.item()]}][{t2a[dest.item()]}] ({actual}, {predicted})")
                     
            optimizer.zero_grad()
            
            # we have to permute because loss function expects (N, C, d1, d2, dK)
            # and we have a K-dimensional loss here
            loss_boa = loss_fn(torch.permute(boa, (0, 2, 1)), batch.y_boa.long())
            
            # we don't need to change input because s is already with shape (B, C) and
            # y_fc_edge_attr has shape (B,) with each value between 0 and C
            loss_edge = loss_fn(s, batch.y_fc_edge_attr.long())
            
            loss = lambda_boa*loss_boa + lambda_edge*loss_edge
            
            loss_avg += loss.item()
            
            loss.backward()
            
            optimizer.step()
        
        print()
        print(f"Avg loss: {loss_avg/len(loader):.5f} | Epoch {epoch}")
        print()
        
if __name__ == "__main__":
    main()