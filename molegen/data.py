import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdmolops
from tqdm.auto import tqdm
from torch_geometric.data import Data
from torch.utils.data import Dataset
import torch
from torch_geometric.utils import dense_to_sparse
from rdkit.Chem.rdchem import BondType
from PIL import Image
from io import BytesIO
from rdkit.Chem.Draw import rdMolDraw2D



class PytorchDataset(Dataset):
    """Wrapper class to use Pytorch DataLoader"""
    def __init__(self, df, colname="data"):
        self.data = df[colname].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

class DataframeDataset:
    """Pandas dataframe for molecules wrapped in a class"""   
    
    def __init__(self,
                 zinc_url="https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
                 n_samples=10,
                 img_size=(300, 300),
                 darkmode=False,
                 bw=False,
                 verbose=False):
        """Load dataframe and prepare for PyG training"""
        
        self.n_samples = n_samples
        
        self.load_zinc_dataset(zinc_url, n_samples)
        self.get_node_mappings(verbose=verbose)
        self.get_edge_mappings(verbose=verbose)
        self.create_pyg_data(verbose=verbose)
        self.init_draw(img_size, darkmode, bw)

    def load_zinc_dataset(self, zinc_url, n_samples):
        """Download ZINC dataset from Kaggle, save as dataframe and canonicalize SMILES"""
        
        tqdm.pandas() # enable progress bars in pandas
        df = pd.read_csv(zinc_url, nrows=n_samples, usecols=["smiles"])

        df['mol'] = df.smiles.progress_apply(Chem.MolFromSmiles)
        df['canonical_smiles'] = df.mol.progress_apply(Chem.MolToSmiles)

        self.df = df


    def get_boa(self, mol, verbose=False):
        """Return bag of atoms representation for the molecule 'mol'"""
        
        # we have an extra dimension here so that PyG batch mechanism
        # concatenates all graphs along row 0
        boa = torch.zeros(1, len(self.n2t), dtype=torch.int32)
        
        atoms = self.get_atom_info(mol)
        
        for atom in atoms:
            symbol, _, _, _ = atom
            boa[:, self.n2t[symbol]] += 1

        if verbose:        
            print("BAG_OF_ATOMS_LABEL:")
            print(boa)
            print()
        
        return boa
            

    def get_node_features(self, mol, verbose=False):
        """Returns a node feature matrix for PyTorch"""
        atoms = self.get_atom_info(mol)

        node_features = torch.zeros(len(atoms), 1, dtype=torch.int32)

        for atom in atoms:
            symbol, idx, _, _ = atom
            node_features[idx] = self.n2t[symbol]

        if verbose:
            print("NODE_FEATURE_MATRIX:")
            print(node_features)
            print()

        return node_features

    def get_pos_features(self, mol, verbose=False):
        """Returns positional features for molecule 'mol'
        
        SMILES -> atom index mapping, i.e. smile2atom[0]
        means that the first (zeroth) element of the smiles string
        is the atom index smile2atom[0]
        we flip this around to get atom2smile[0] means atom index 0 appears
        appears at the position atom2smile[0]
        note that this is kind of unnecessary since we already canonicalize the SMILES 
        and so it will always be in the right order... thus we simply need to convert the position
        into one hot vector. we add it for completeness... it would otherwise just be torch.arange(len(atoms))    
        """
        atoms = self.get_atom_info(mol)
        
        # this assures that the _smilesAtomOutputOrder prop is populated
        try:
            mol.GetProp("_smilesAtomOutputOrder") 
        except(KeyError):
            Chem.MolToSmiles(mol)
            
        smile2atom = mol.GetProp("_smilesAtomOutputOrder") 
        smile2atom = [int(x) for x in smile2atom.strip('[]').split(',')]
        atom2smile = torch.zeros(len(atoms), 1, dtype=torch.int32)
        
        for smile_pos, atom_idx in enumerate(smile2atom):
            atom2smile[atom_idx] = smile_pos

        if verbose:
            print("POS_FEATURE:")
            print(atom2smile)
            print()
            
        return atom2smile, smile2atom
    
    def parse_mol(self, mol, mol_id=0, verbose=False): 
        """Pandas apply function applied on each mol row in dataframe
        
        Return PyG Data object
        """       
        x = self.get_node_features(mol, verbose=verbose)
        pos = self.get_pos_features(mol, verbose=verbose)
        y_boa = self.get_boa(mol, verbose=verbose)
        
        edge_index, edge_attr = self.get_connectivity(mol, verbose=verbose)
        edge_attr = self.convert_edge_features(edge_attr, verbose=verbose)
        
        fc_edge_index, y_fc_edge_attr = self.get_fc_connectivity(mol, verbose=verbose)
        y_fc_edge_attr = self.convert_edge_features(y_fc_edge_attr, verbose=verbose)
        
        # note that PyG will automatically increment fc_edge_index as per:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html#torch_geometric.data.Batch
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y_boa=y_boa,
                    fc_edge_index=fc_edge_index,
                    y_fc_edge_attr=y_fc_edge_attr,
                    pos=pos,
                    graph_id = torch.tensor([mol_id])) # to keep track of this data object and map to SMILES later
        data.validate(raise_on_error=True)
        
        return data  
    
    def create_pyg_data(self, verbose=False):
        """Calls Pandas apply on mol objects with apply_create_pyg_data"""        
        parse_mol_wrapper = lambda row: self.parse_mol(row['mol'], row.name, verbose=verbose)
        self.df["data"] = self.df.progress_apply(parse_mol_wrapper, axis=1)
        

    def get_edge_mappings(self, verbose=False):
        """Create edge-token, and token-edge mappings
        
        We need to create a mapping from index to atom type, and atom type back to index so that we
        can decode the atoms. Think of it like a tokenizer encoder/decoder. This will be a dictionary.
        Same thing for the edges. We will scan through our dataset and create our vocabulary
        which maps atom symbol to indice, and vice versa    
        """
        e2t = dict() # edge to token
        t2e = dict() # token to edge

        # RDKit has a lot of options here, but we will constrain to these 4
        # and maybe make a note of which molecules have bonds we wouldn't expect
        # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetIdx
        # note that we are treating aromatic as a "double" bond
        BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
        for idx, bond_type in enumerate(BOND_TYPES):
            e2t[bond_type] = idx
            t2e[idx] = bond_type
        
        if verbose:
            print("Edge to token:", e2t)
            print("Token to edge:", t2e)
        
        self.e2t = e2t
        self.t2e = t2e


    def get_node_mappings(self, verbose=False):
        """Create node-token, token-node mappings
        
        We need to create a mapping from index to atom type, and atom type back to index so that we
        can decode the atoms. Think of it like a tokenizer encoder/decoder. This will be a dictionary.
        Same thing for the edges. We will scan through our dataset and create our vocabulary
        which maps atom symbol to indice, and vice versa    
        """
        
        n2t = dict() # node (atom) to token
        t2n = dict() # token to node (atom)
        
        # for a given molecule, what is the maximum amount of atoms which are present.
        # used to set the output matrix size when doing softmax
        max_atoms = float('-inf') 

        for mol in list(self.df.mol):
            max_atoms = max(max_atoms, mol.GetNumAtoms())
            
            for atom in mol.GetAtoms():
                
                symbol = atom.GetSymbol()
                
                if symbol not in n2t:
                    t = len(n2t)
                    n2t[symbol] = t
                    t2n[t] = symbol

        if verbose:
            print("Node to token:", n2t)
            print("Token to node:", t2n)
            print("Max number of atoms in any given molecule:", max_atoms)
            
        self.n2t = n2t
        self.t2n = t2n
        self.max_atoms = max_atoms

    
    def init_draw(self, img_size=(300, 300), darkmode=False, bw=False):
        """Initialize drawing canvas for RDKit molecules"""
        self.d2d = rdMolDraw2D.MolDraw2DCairo(*img_size)
        
        if darkmode:
            rdMolDraw2D.SetDarkMode(self.d2d)
        
        if bw:
            self.d2d.drawOptions().useBWAtomPalette()
            if darkmode:
                self.d2d.drawOptions().updateAtomPalette({-1: (1, 1, 1)})
            

    def get_img(self, mol, legend=''):
        """Draw the mol and return a PIL image object"""
        self.d2d.ClearDrawing()
        self.d2d.DrawMolecule(mol, legend=legend)
        self.d2d.FinishDrawing()
        bio = BytesIO(self.d2d.GetDrawingText())
        return Image.open(bio)

            
    
    ##############################################################################
    # Static methods
    ##############################################################################

    @staticmethod
    def convert_edge_features(adj_matrix, verbose=False):
        """Convert RDKit bond types to our edge type set
        
        RDKit has a lot of options, but we are squashing them down to 4
        https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetIdx
        """

        new_adj_matrix = adj_matrix.clone()

        # only account for single, double, and triple bonds
        # RDKit bond type values are increasing so this works
        new_adj_matrix[adj_matrix > BondType.TRIPLE] = BondType.SINGLE
        new_adj_matrix[adj_matrix == BondType.AROMATIC] = BondType.DOUBLE

        # RDKit gives a non-zero value for unconnected bond, but in get_fc_connectivity
        # we set all unconnected bonds to this bondtype. thus we revert back to 0
        new_adj_matrix[adj_matrix == BondType.ZERO] = 0.0

        if verbose:
            print("NEW ADJ MATRIX:")
            print(new_adj_matrix)
            print()

        return new_adj_matrix
    

    @staticmethod
    def get_atom_info(mol, verbose=False):
        """Extract RDKit info from mol object's atoms
        
        Iterate through the atoms in `mol` and return information such as
        the atom symbol, atomic number, bond degree, etc.
        """
        atoms = []

        if verbose:
            print("ATOM_INFO:")
            print("SYMBOL INDEX NUMBER DEGREE")
            
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            idx = atom.GetIdx()
            num = atom.GetAtomicNum()
            deg = atom.GetDegree()

            atoms.append((symbol, idx, num, deg))

            if verbose:
                print(*atoms[-1])

        if verbose:
            print()

        return atoms

    @staticmethod
    def get_bond_info(mol, verbose=False):
        """Extract RDKit info from mol object's bonds
        
        Iterate through the bonds in `mol` and return information
        such as which atoms each bond is connected to and the type
        of bond (`bond_num`)
        """
        bonds = []

        if verbose:
            print("BOND_INFO:")
            print("INDEX BEGIN_ATOM END_ATOM BOND_TYPE BOND_NUM")

        for bond in mol.GetBonds():
            idx = bond.GetIdx()
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()
            bond_num = bond.GetBondTypeAsDouble()  # TODO: need to convert this

            bonds.append((idx, atom1, atom2, bond_type, bond_num))

            if verbose:
                print(*bonds[-1])

        if verbose:
            print()

        return bonds
    

    @staticmethod
    def get_connectivity(mol, verbose=False):
        """Get graph connection info (sparse adjacency matrix) from mol object
        
        First create an adjacency matrix using RDKit function
        and then convert it to a sparse matrix for PyTorch Geometric
        to batch process. Return PyTorch sparse matrices
        """

        rdkit_adj_matrix = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        adj_matrix_tensor = torch.tensor(rdkit_adj_matrix, dtype=torch.int32)
        sparse_matrix = dense_to_sparse(adj_matrix_tensor)

        if verbose:
            print("RDKIT_ADJ_MATRIX:")
            print(rdkit_adj_matrix)
            print()

            print("SPARSE_MATRIX:")
            print(sparse_matrix)
            print()

        return sparse_matrix


    @staticmethod
    def get_fc_connectivity(mol, verbose=False):
        """Get fully connected graph connection info (sparse adjacency matrix) from mol object
        
        First create an adjacency matrix using RDKit function
        and then convert it to a sparse matrix for PyTorch Geometric
        to batch process. Return PyTorch sparase matrix which is fully-connected
        
        note it is inefficient to completely store the sparse matrix instead of just the dense matrix
        because by definition the fully connected dense matrix has a connection between every node 
        to mitigate passing this around, we could create the dense matrix on the fly in model.py,
        but it just makes things more complicated as the adjacency matrices will be different sizes
        depending on the num of atoms in the molecule
        """

        rdkit_adj_matrix = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
        fc_adj_matrix = torch.tensor(rdkit_adj_matrix, dtype=torch.int32)

        if torch.any(fc_adj_matrix == BondType.ZERO):
            raise ValueError("adj_matrix has some BondType.ZERO edges which we override! Unexpected results will occur")
        if torch.any(fc_adj_matrix.diag() > 0):
            raise ValueError("Adjacency matrix has non-zero diagonal elements which leads to unexpected results")
            
        # we set these to a non-zero value so that later in 'get_edge_features'
        # we can include them for our bond prediction loss
        fc_adj_matrix[fc_adj_matrix == 0] = BondType.ZERO
        fc_adj_matrix.fill_diagonal_(0) # removes self-loops
        fc_sparse_matrix = dense_to_sparse(fc_adj_matrix)
        
        if verbose:
            print("FC_ADJ_MATRIX:")
            print(fc_adj_matrix)
            print()
            
            print("FC_SPARSE_MATRIX:")
            print(fc_sparse_matrix)
            print()

        return fc_sparse_matrix


        
        
