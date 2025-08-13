
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
import torch
from . import data as prepare_data
from .model import MoleGen
from collections import defaultdict

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
    num_layers = 3    # number of GCN layers
    lr = 0.002
    betas = (0.9, 0.999)
    eps = 1e-08
    epochs = 2000
    lambda_boa = 0.05
    lambda_edge = 0.45
    lambda_kl = 0.5
    batch_size = 128
    shuffle = False
    ###################################################
    
    torch.manual_seed(42)
    
    df, a2t, t2a, e2t, t2e, max_atoms = prepare_data.main()
    vocab_size = len(a2t)

    dataset = DataFrameDataset(df, "data")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    data = next(iter(loader))
    print(data)
    
    model = MoleGen(vocab_size=vocab_size, num_layers=num_layers, embd=embd, max_atoms=max_atoms)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(model)
    
    torch.serialization.add_safe_globals([float])
    checkpoint = torch.load("molegen.ckpt", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    model.train()
    MAX_SMILES_STRING = 50
    for epoch in range(epochs):
        loss = dict()
        avg_loss = defaultdict(float)
        
        for idx, batch in enumerate(loader):
            boa, z, s, kl = model(batch)
            
            if DEBUG:    
                smiles = df['canonical_smiles'].iloc[batch.graph_id[0].item()]
                
                print(f"({idx:<2}) {smiles[:MAX_SMILES_STRING]:<{MAX_SMILES_STRING}}{'...' if len(smiles) > MAX_SMILES_STRING else '   '} | ", end="")
                
                boa_actual = batch.y_boa[0]
                boa_pred = torch.argmax(boa[0], dim=-1)
                
                for token, (count, pred_count) in enumerate(zip(boa_actual, boa_pred)):
                    print(f"{t2a[token]}({count}, {pred_count}) | ", end="")
                
                
                # select only edges that correspond to first graph 
                edge = batch.batch[batch.fc_edge_index[0]] # (num_edges, )
                edge = torch.arange((edge[edge == 0]).shape[0]) # (num_edges_graph0, )
                edge_pred = s[edge] # (num_edges_graph0, C)
                edge_pred = torch.argmax(edge_pred, dim=-1) # (num_edges_graph0, )
                
                src_atoms = batch.fc_edge_index[0, edge] # (num_edges_graph0, )
                dest_atoms = batch.fc_edge_index[1, edge] # (num_edges_graph0, )
                src_atom_feats = batch.x[src_atoms].squeeze(-1) # (num_nodes_graph0, )
                dest_atom_feats = batch.x[dest_atoms].squeeze(-1) # (num_nodes_graph0, )

                # only show the edges that are not zero because we would have too many otherwise
                correct_edges = 0
                num_edges = 0
                for num, (src, dest, actual, predicted) in enumerate(zip(src_atom_feats, dest_atom_feats, batch.y_fc_edge_attr, edge_pred)):
                    if actual != 0:
                        num_edges += 1
                        if actual == predicted:
                            correct_edges += 1
                            
                print(f"edges {correct_edges}/{num_edges}")
                     
            optimizer.zero_grad()
            
            # we have to permute because loss function expects (N, C, d1, d2, dK)
            # and we have a K-dimensional loss here
            loss['boa'] = loss_fn(torch.permute(boa, (0, 2, 1)), batch.y_boa.long())
            
            # we don't need to change input because s is already with shape (B, C) and
            # y_fc_edge_attr has shape (B,) with each value between 0 and C
            loss['edge'] = loss_fn(s, batch.y_fc_edge_attr.long())
            
            loss['kl'] = kl
            
            loss['total'] = lambda_boa*loss['boa'] + lambda_edge*loss['edge'] + lambda_kl*loss['kl']
            loss['total'].backward()
            
            # TODO: add torch inference mode wrapper here?
            avg_loss['total'] += loss['total'].item()
            avg_loss['boa'] += loss['boa'].item()
            avg_loss['edge'] += loss['edge'].item()
            avg_loss['kl'] += loss['kl'].item()
            
            optimizer.step()
        
        print()
        print(f"Epoch {epoch} ------ average losses ------- | ", end="")
        for name,value in avg_loss.items():
            print(f"{name}: {value/len(loader):.5f} | ", end="")
        print()
        print()
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'avg_loss': avg_loss
        }, "molegen.ckpt")

if __name__ == "__main__":
    main()