
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch
from . import data as prepare_data
from .model import MoleGen


class DataFrameDataset(Dataset):
    def __init__(self, df, colname="data"):
        self.data = df[colname].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
      
def main():
    DEBUG = False
    
    df, a2t, t2a, e2t, t2e, max_atoms = prepare_data.main()

    torch.manual_seed(42) # reproducibility
    dataset = DataFrameDataset(df, "data")
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    data = next(iter(loader))
    print(data)
    
    # import code; code.interact(local=locals())
    # import pdb; pdb.set_trace()

    ###################################################
    embd = 16 # embedding size of a vocab indice
    vocab_size = len(a2t)
    num_layers = 1     # number of GCN layers
    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-08
    epochs = 100
    ###################################################
    
    model = MoleGen(vocab_size=vocab_size, num_layers=num_layers, embd=embd, max_atoms=max_atoms)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(model)

    model.eval()
    for epoch in range(epochs):
        loss_avg = 0.0
        
        for idx, batch in enumerate(loader):
            boa, z, s = model(batch)
            
            # import code; code.interact(local=locals())
            
            optimizer.zero_grad()
            
            # we have to permute because loss function expects (N, C, d1, d2, dK)
            # and we have a K-dimensional loss here
            loss_boa = loss_fn(torch.permute(boa, (0, 2, 1)), batch.y_boa.long())
            
            # we don't need to change input because s is already with shape (B, C) and
            # y_fc_edge_attr has shape (B,) with each value between 0 and C
            loss_edge = loss_fn(s, batch.y_fc_edge_attr.long())
            
            # TODO add loss from other functions
            loss = loss_boa + loss_edge
            
            loss_avg += loss.item()
            
            loss.backward()
            
            optimizer.step()
            
        print(f"Avg loss: {loss_avg/len(loader):.5f} | Epoch {epoch}")
        
        
        
        if DEBUG:    
                
            for token, count in enumerate(batch.y[0]):
                print(f"{t2a[token]:<2}({count:>3}) ", end="")
            print("   EXPECTED (1st graph) - batch", idx)
            
            for token, logits in enumerate(boa[0]):
                count = torch.argmax(logits)
                print(f"{t2a[token]:<2}({count:>3}) ", end="")
            print("   PREDICTED (1st graph) - batch", idx)  
    
    
    

if __name__ == "__main__":
    main()