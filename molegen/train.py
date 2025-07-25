
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
    ###################################################
    
    model = MoleGen(vocab_size=vocab_size, num_layers=num_layers, embd=embd, max_atoms=max_atoms)
    


    print(model)

    # data = dataset[0]
    # result = model(data)

    for batch in loader:
        boa, z = model(batch)
    
    

if __name__ == "__main__":
    main()