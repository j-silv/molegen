import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.utils import scatter
import torch
import torch.nn.functional as F

class AtomMLP(nn.Module):
    """Multi-layer perceptron (MLP) which predicts BOA for each graph
    
    Implementation of Eq. 9 in paper. This MLP takes in a graph level embedding 'z'
    and produces a BOA where we essentially get an integer distribution of the predicted atoms

    Args:
        embd       : embedding dimension for latent space
        vocab_size : number of unique atoms in training set
        max_atoms  : max number of total atoms in any molecule
    """
    def __init__(self, embd=16, vocab_size=8, max_atoms=100):
        
        super().__init__()   
        
        self.vocab_size = vocab_size
        self.max_atoms = max_atoms
        
        # note there isn't much information on the paper on this MLP but I am assuming we
        # use a nn.ReLU for the activation
        self.mlp = nn.Sequential(
            nn.Linear(embd, embd),
            nn.ReLU(),
            nn.Linear(embd, vocab_size*max_atoms),
        )
        
    def forward(self, z):
        """Forward pass of Atom MLP
        
        We compute an output vector of size (num_graphs, vocab_size, max_atoms) with
        the last dimension essentially representing a one-hot vector that has a dimension size
        up to the maximum number of atoms. We perform the MLP with a flattened last dimension
        (vocab_size*max_atoms) and then view it as a separate dimension for the loss calculation.
        
        Note that a small optimization here 
        is to not do the total number of atoms in the training set but simply the
        highest number of any particular atom instead.
        
        Args:
            z   : graph level embedding (num_graphs, embd)
            
        Returns:
            boa : graph level BOA (num_graphs, vocab_size, max_atoms)
        """
        boa = self.mlp(z)
        
        return boa.view(-1, self.vocab_size, self.max_atoms)


class BondMLP(nn.Module):
    """Multi-layer perceptron which predicts which bond for each edge
    
    Implementation of Eq. 11 in paper. This MLP takes in edge embeddings 'e'
    and produces an integer distribution of the predicted bond types

    Args:
        embd       : embedding dimension for latent space
        num_bonds  : number of unique atoms in training set
        num_bonds  : number of different types of bonds for classification
        
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
        """Forward pass of Bond MLP
        
        We compute an output vector of size (num_edges, num_bonds) with
        the last dimension representing a one-hot vector that has a dimension size
        up to the different types of bonds.
        
        Args:
            e   : edge embedding (num_edges, embd)
            
        Returns:
            bonds : graph level BOA (num_edges, num_bonds)
        """
        bonds = self.mlp(e)
        return bonds

        
class AtomGCNLayer(nn.Module):
    """Updates atom embeddings with Conv->BN->Relu->Residual

    Implementation of Eq. 4 in paper

    Args:
        embd       : embedding dimension for latent space    

    """
    def __init__(self, embd=16):
        super().__init__()

        self.gcn = ResGatedGraphConv(in_channels=embd, out_channels=embd, edge_dim=embd)
        self.bn = nn.BatchNorm1d(embd)
        self.relu = nn.ReLU()

    def forward(self, data):
        """Perform forward pass for atom GCN
        
        Args:
            data  : PyG Data() object (possibly batched) with the following attributes:
                x           : atom token, e.g. x[0] is the embedding vector for atom/node 0 (num_nodes, embd)
                edge_index  : bond connectivity, e.g. atom at edge_index[0,0] connects to edge_index[1, 0] (2, num_edges)
                edge_attr   : bond token, e.g. edge_attr[0] is the embedding vector for bond/edge 0 (num_edges, embd)
                
        Returns:
            h     :   Updated node embedding (num_nodes, embd)
            
        """ 
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h = self.gcn(x, edge_index, edge_attr)
        h = self.bn(h)
        h = self.relu(h)
        h = x + h # residual pathway
        return h

class BondGCNLayer(nn.Module):
    """Updates bond embeddings with Linear->BN->Relu->Residual

    Implementation of Eq. 5 in paper
    
    Args:
        embd       : embedding dimension for latent space

    """

    def __init__(self, embd=16):
        super().__init__()
        self.v = nn.ModuleList([nn.Linear(embd,embd) for _ in range(3)])
        self.bn = nn.BatchNorm1d(embd)
        self.relu = nn.ReLU()

    def forward(self, data):
        """Perform forward pass for bond GCN
        
        Args:
            data  : PyG Data() object (possibly batched) with the following attributes:
                x           : atom token, e.g. x[0] is the embedding vector for atom/node 0 (num_nodes, embd)
                edge_index  : bond connectivity, e.g. atom at edge_index[0,0] connects to edge_index[1, 0] (2, num_edges)
                edge_attr   : bond token, e.g. edge_attr[0] is the embedding vector for bond/edge 0 (num_edges, embd)
                
        Returns:
            e     :   Updated edge embedding (num_edges, embd)
            
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h_src = x[edge_index[0]] # oh... apparently we get this for free?
        h_dest = x[edge_index[1]] # apparently pytorch geometric already handles large batch

        e = self.v[0](edge_attr) + self.v[1](h_src) + self.v[2](h_dest)

        e = self.bn(e)
        e = self.relu(e)

        e = edge_attr + e # residual pathway
        return e

class MoleGen(nn.Module):
    """Main model for VAE molecular generation
    
    This model is an implementation of the paper:
    A Two-Step Graph Convolutional Decoder for Molecule Generation
    by Bresson et Laurent (2019). A molecule graph is first encoded 
    into a latent representation 'z', which is used to produce a 
    'Bag of Atoms' (BOA). This BOA tells us how many of each atom
    we have in the predicted molecule, ignoring connectivity. 
    The second stage takes 'z' and the original input formula and
    decodes the edge feature connectivity from a fully connected network.
    
    Args:
        vocab_size : number of unique atoms in training set
        embd       : embedding dimension for latent space
        num_layers : number of GNN layers (message passing/k-hop distance)
        max_atoms  : max number of total atoms in any molecule
        num_bonds  : number of different types of bonds for classification
    """
    def __init__(self, vocab_size=8, embd=16, num_layers=4, max_atoms=100, num_bonds=4):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embd = embd
        self.num_layers = num_layers
        self.max_atoms = max_atoms

        self.bond_embeddings = nn.Embedding(vocab_size, embd) 
        self.atom_encoder = nn.ModuleList([AtomGCNLayer(embd) for _ in range(num_layers)])
        self.bond_encoder = nn.ModuleList([BondGCNLayer(embd) for _ in range(num_layers)])
        self.atom_decoder = nn.ModuleList([AtomGCNLayer(embd) for _ in range(num_layers)])
        self.bond_decoder = nn.ModuleList([BondGCNLayer(embd) for _ in range(num_layers)])

        self.linear = nn.ModuleDict(dict(
            atom_embedding=nn.Linear(max_atoms + vocab_size, embd),
            a=nn.Linear(embd, embd),
            b=nn.Linear(embd, embd),
            c=nn.Linear(embd, embd),
            d=nn.Linear(embd, embd),
            u=nn.Linear(embd, embd)
        ))

        self.sigmoid = nn.Sigmoid()
        
        self.atom_mlp = AtomMLP(embd, vocab_size, max_atoms)
        self.bond_mlp = BondMLP(embd, num_bonds)
        
    def forward(self, input_data):
        """Forward pass for MoleGen
        
        First we look up the atom and bond embeddings, and then apply GCN layers iteratively.
        Afterwards we reduce the node and edge embeddings into a single graph latent vector 'z'.
        We apply MLP to 'z' to produce a BOA. Then we use the original 'x' tokens and the 'z' vector
        to produce an edge probability matrix, which we again apply an MLP to to get 's' which is
        the predicted bond tokens for each edge of a fully connected network.
        
        Args:
            input_data  : PyG Data() object (possibly batched) with the following attributes:
                x           : atom token, e.g. x[0] is the integer token for atom/node 0 (num_nodes, 1)
                edge_index  : bond connectivity, e.g. atom at edge_index[0,0] connects to edge_index[1, 0] (2, num_edges)
                edge_attr   : bond token, e.g. edge_attr[0] is the integer token for bond/edge 0 (num_edges, )
                
        Returns:
            boa     :   Bag of Atoms prediction (num_graphs, vocab_size, max_atoms)
            z       :   Per graph latent representation (num_graphs, embd)
            s       :   Edge probability matrix (num_fc_edges, num_bonds)
        """


        ######################################################################################
        # Encoding step
        ######################################################################################
        
        # Break symmetry and add positional embeddings to atom embeddings
        one_hot_positions = F.one_hot(input_data.pos.view(-1).long(), num_classes=self.max_atoms) # view avoids extra dimension added
        one_hot_atom_tokens = F.one_hot(input_data.x.view(-1).long(), num_classes=self.vocab_size)
        one_hot_concat = torch.cat((one_hot_positions, one_hot_atom_tokens), 1).to(dtype=torch.float32)
        atom_embedding = self.linear['atom_embedding'](one_hot_concat)
        
        # Get bond embeddings
        bond_embedding = self.bond_embeddings(input_data.edge_attr)

        # Apply GCN layers iteratively (encoding)
        h = atom_embedding
        e = bond_embedding
        for i in range(self.num_layers):
            data = Data(x=h, edge_index=input_data.edge_index, edge_attr=e)
            h = self.atom_encoder[i](data)
            e = self.bond_encoder[i](data)

        # Extract source and destination atom features 
        h_src = h[input_data.edge_index[0]]
        h_dest = h[input_data.edge_index[1]]

        # Apply linear and activation before reduction step
        z = (self.sigmoid(self.linear['a'](e) + self.linear['b'](h_src) + self.linear['c'](h_dest)))*self.linear['d'](e)
        
        # the .batch attribute only maps to nodes
        # to get a mapping to the edge -> graph we simply index
        # into the batch to get the indexes of which edge corresponds to which graph 
        batch_edge = input_data.batch[input_data.edge_index[0]] # (num_edges, )
        
        # now batch_edge will ressemble input_data.batch but for edges instead of nodes
        # apply scatter operation to get a per graph output
        z = scatter(z, batch_edge, dim=0, reduce='sum') # (num_graphs, embd)
        
        boa = self.atom_mlp(z) # (num_graphs, vocab_size, max_atoms)


        ######################################################################################
        # Decoding step
        ######################################################################################

        # in the bond generation step, each edge gets the same initial feature vector
        fc_bond_embedding = self.linear['u'](z) # (num_graphs, embd) 
        
        # same trick as before where we convert node mappings to edge mappings
        fc_batch_edge = input_data.batch[input_data.fc_edge_index[0]]
        
        # we index into our bond embeddings to get the bond embeddings
        # for each edge per graph, since fc_bond_embedding is batched
        fc_edge_attr = fc_bond_embedding[fc_batch_edge] # (num_fc_edges, embd)
        
        # re-use original atom embedding since the fc graph has the same nodes just with different bonds
        fc_atom_embedding = atom_embedding # (num_atoms, embd)
        
        # Apply GCN layers iteratively (decoding)
        h = fc_atom_embedding
        e = fc_edge_attr
        for i in range(self.num_layers):
            data = Data(x=h, edge_index=input_data.fc_edge_index, edge_attr=e)
            h = self.atom_decoder[i](data)
            e = self.bond_decoder[i](data)  
        
        # now take each edge and apply MLP to predict bond type for each
        s = self.bond_mlp(e)
        
        return boa, z, s

