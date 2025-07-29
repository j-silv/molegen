import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.utils import scatter
from torch_geometric.utils import dense_to_sparse
import torch

class MLPAtom(nn.Module):
    """Multi-layer perceptron which gets soft bag of atom
    
    Implementation of Eq. 9 in paper
    """
    def __init__(self, embd=16, vocab_size=8, max_atoms=100):
        super().__init__()   
        
        self.vocab_size = vocab_size
        self.max_atoms = max_atoms
        
        # note that we have an input vector of size (num_graphs, embd)
        # we need an output vector of size (vocab_size, R) where vocab_size
        # is 'm' in the paper which means the different atoms we can select
        # and R is basically a one-hot vector that goes up to the maximum number
        # of total atoms in any one molecule in the training set
        #
        # (small optimization here is to not do the total number of atoms in training set
        # but simply the highest number of any particular atom, i.e. take C02:
        # we would only need to have R == 2, since even though we have 3 atoms,
        # the max number for any single atom is only 2, thus we don't need R == 3.
        # i think they just simplified it in the paper)
        #
        # I'm not sure how to construct this though. I could do like a concatenation thing though
        # and have m linear layers with all different weights and then just concatenate the output
        #
        # ah okay the intuition here is that we are going to output a R*vocab_size vector,
        # and then we will reshape it afterwards. 
        
        # not a lot of info on the structure of a simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(embd, embd),
            nn.ReLU(),
            nn.Linear(embd, vocab_size*max_atoms),
        )
        
    def forward(self, z):
        boa = self.mlp(z)
        
        return boa.view(-1, self.vocab_size, self.max_atoms)


class MLPBond(nn.Module):
    """Multi-layer perceptron which predicts which bond for each edge
    
    Implementation of Eq. 11 in paper
    """
    def __init__(self, embd=16, num_bonds=4):
        super().__init__()   
        
        self.num_bonds = num_bonds
        
        self.mlp = nn.Sequential(
            nn.Linear(embd, embd),
            nn.ReLU(),
            nn.Linear(embd, num_bonds),
        )
        
    def forward(self, e):
        bonds = self.mlp(e)
        return bonds

        
class GCNAtomLayer(nn.Module):
    """Intermediate class which performs ConvNet, BN, Relu, and Residual for atoms

    Implementation of Eq. 4 in paper
    """
    def __init__(self, embd=16):
        super().__init__()

        self.gcn = ResGatedGraphConv(in_channels=embd, out_channels=embd, edge_dim=embd)
        self.bn = nn.BatchNorm1d(embd)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.gcn(x, edge_index, edge_attr)
        h = self.bn(h)
        h = self.relu(h)
        h = x + h # resid
        return h

class GCNBondLayer(nn.Module):
    """Intermediate class which performs linear matrix multiply for bonds

    Implementation of Eq. 5 in paper
    """

    def __init__(self, embd=16):
        super().__init__()
        self.v1 = nn.Linear(embd, embd)
        self.v2 = nn.Linear(embd, embd)
        self.v3 = nn.Linear(embd, embd)
        self.bn = nn.BatchNorm1d(embd)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_src = x[edge_index[0]] # oh... apparently we get this for free?
        h_dest = x[edge_index[1]] # apparently pytorch geometric already handles large batch

        e = self.v1(edge_attr) + self.v2(h_src) + self.v3(h_dest)

        e = self.bn(e)
        e = self.relu(e)

        e = edge_attr + e # resid
        return e

class MoleGen(nn.Module):
    """Main model for VAE model generating"""
    def __init__(self, vocab_size=8, embd=16, num_layers=4, max_atoms=100, num_bonds=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embd = embd
        self.num_layers = num_layers

        self.aembs = nn.Embedding(vocab_size, embd) # atom embeddings
        self.bembs = nn.Embedding(vocab_size, embd) # bond embeddings

        # GCN atom encoder
        self.gcn_aenc = nn.ModuleList([GCNAtomLayer(embd) for i in range(num_layers)])

        # GCN bond encoder
        self.gcn_benc = nn.ModuleList([GCNBondLayer(embd) for i in range(num_layers)])

        # GCN atom decoder
        self.gcn_adec = nn.ModuleList([GCNAtomLayer(embd) for i in range(num_layers)])

        # GCN bond decoder
        self.gcn_bdec = nn.ModuleList([GCNBondLayer(embd) for i in range(num_layers)])

        self.a = nn.Linear(embd, embd)
        self.b = nn.Linear(embd, embd)
        self.c = nn.Linear(embd, embd)
        self.d = nn.Linear(embd, embd)
        
        self.u = nn.Linear(embd, embd)

        self.sig = nn.Sigmoid()
        
        self.mlp_atom = MLPAtom(embd, vocab_size, max_atoms)
        self.mlp_bond = MLPBond(embd, num_bonds)

    def forward(self, input_data):

        x, edge_index, edge_attr  = input_data.x, input_data.edge_index, input_data.edge_attr


        # we look up the embeddings from our table
        # data.x.shape         = (num_atoms, 1)
        # aemb.shape           = (num_atoms, embedding_dim)
        # data.edge_attr.shape = (num_bonds)
        # bemb.shape           = (num_bonds, embedding_dim)
        aemb = self.aembs(x.view(-1)) # this is used later for the bond generation
        bemb = self.bembs(edge_attr)

        # we run GCN
        h = aemb
        e = bemb

        # import pdb; pdb.set_trace()
        for i in range(self.num_layers):
            data = Data(x=h, edge_index=edge_index, edge_attr=e)

            h = self.gcn_aenc[i](data)
            e = self.gcn_benc[i](data)

        h_src = h[edge_index[0]]
        h_dest = h[edge_index[1]]


        z = (self.a(e) + self.b(h_src) + self.c(h_dest))*self.d(e)
        
        # the .batch attribute only maps to nodes
        # to get a mapping to the edge -> graph (which is what we need)
        # we simply index into the batch to get the indexes of which
        # edge corresponds to which graph 
        batch_edge = input_data.batch[edge_index[0]]
        
        # now batch_edge will be of shape (num_edges),
        # and cruicially it will ressemble .batch but now have 0s for the first
        # graph's edges, 1s for the second graph's edges, etc.
        # this is because the edge_index is automatically incremented,
        # imagine we had a graph with 25 nodes (N) and 30 edges (M)
        # then edge_index[0, 0:M] will only contain values between 0 and 24 inclusive.
        # thus, we will convert the 0 to 24 inclusive into a 0 to 29 inclusive 
        # since we have M entries 
        # then once we have a similar looking .batch for edges, we can do the
        # scatter operation to get a per graph output
        z = scatter(z, batch_edge, dim=0, reduce='sum') # (num_graphs, embedding_dim)
        
        boa = self.mlp_atom(z) # (num_graphs, vocab_size, max_atoms)


        ######################################################################################
        # VVVVVVVVVVVVV  untested VVVVVVVVVVVVVVVVVVVVVV
        ######################################################################################

        fc_bemb = self.u(z) # (num_graphs, embedding_dim) -> this needs to be applied to each edge_attr 
        
        # with this, we can create a new embedding attr matrix
        # then we will index into the fc_bemb which has e.g. 32 graphs, and we want to apply fc_bemb[0] to 
        # the total number of edges in the first graph, fc_bemb[1] to total number of edges in second graph,
        # etc... so this trick turns the node indexes into edge_indexes 
        fc_batch_edge = input_data.batch[input_data.fc_edge_index[0]]
        
        # and then we simply index into our bond embeddings to get the bond embeddings
        # for each edge per graph! 
        fc_edge_attr = fc_bemb[fc_batch_edge] #  (num_fc_edges, embedding_dim)
        
        # (num_atoms, embedding_dim)
        fc_aemb = aemb # same as original because the fc graph still has the same nodes in it, just with different bonds
            
        h = fc_aemb
        e = fc_edge_attr
        for i in range(self.num_layers):
            data = Data(x=h, edge_index=input_data.fc_edge_index, edge_attr=e)
            h = self.gcn_adec[i](data)
            e = self.gcn_bdec[i](data)  
        
        s = self.mlp_bond(e)
        
        
        return boa, z, s

