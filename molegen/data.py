import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdmolops
from tqdm.auto import tqdm
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import dense_to_sparse
from rdkit.Chem.rdchem import BondType

def atom_info(mol, verbose=False):
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


def bond_info(mol, verbose=False):
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


def connect_info(mol, verbose=False):
    """Get graph connection info from mol object
    
    First create an adjacency matrix using RDKit function
    and then convert it to a sparse matrix for PyTorch Geometric
    to batch process
    """

    rdkit_adj_matrix = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    adj_matrix_tensor = torch.tensor(rdkit_adj_matrix, dtype=torch.int32)
    sparse_matrix = dense_to_sparse(adj_matrix_tensor)
    # note it is inefficient to completely store the sparse matrix instead of just the dense matrix
    # because by definition the fully connected dense matrix has a connection between every node 
    # to mitigate passing this around, we could create the dense matrix on the fly in model.py,
    # but it just makes things more complicated as the adjacency matrices will be different sizes
    # depending on the num of atoms in the molecule
    fc_adj_matrix = adj_matrix_tensor
    if torch.any(fc_adj_matrix == BondType.ZERO):
        raise ValueError("adj_matrix has some BondType.ZERO edges which we override! Unexpected results will occur")
    if torch.any(fc_adj_matrix.diag() > 0):
        raise ValueError("Adjacency matrix has non-zero diagonal elements which leads to unexpected results")
        
    # we set these to a non-zero value so that later we can include them for our bond prediction loss
    fc_adj_matrix[fc_adj_matrix == 0] = BondType.ZERO
    fc_adj_matrix.fill_diagonal_(0) # removes self-loops
    fc_sparse_matrix = dense_to_sparse(fc_adj_matrix)
    
    if verbose:
        print("RDKIT_ADJ_MATRIX:")
        print(rdkit_adj_matrix)
        print()

        print("SPARSE_MATRIX:")
        print(sparse_matrix)
        print()

        print("FC_ADJ_MATRIX:")
        print(fc_adj_matrix)
        print()
        
        print("FC_SPARSE_MATRIX:")
        print(fc_sparse_matrix)
        print()

    return sparse_matrix, fc_sparse_matrix


def node_feature_matrix(mol, a2t, verbose=False):
    """Returns a node feature matrix for PyTorch

    Also return a label for the number of atoms of each token
    in the molecule
    """
    atoms = atom_info(mol)

    matrix = torch.zeros(len(atoms), 1, dtype=torch.int32)
    # we have an extra dimension here so that PyG working concatenates all graphs along row 0
    boa = torch.zeros(1, len(a2t), dtype=torch.int32)
    
    for atom in atoms:
        symbol, idx, _, _ = atom
        matrix[idx] = a2t[symbol]
        boa[:, a2t[symbol]] += 1

    if verbose:
        print("NODE_FEATURE_MATRIX:")
        print(matrix)
        print()
        
        print("BAG_OF_ATOMS_LABEL:")
        print(boa)
        print()


    return matrix, boa


def map_rdkit_bond_types(rdkit_edge_types, verbose=False):
    """Convert RDKit bond types to our edge type set"""
    # RDKit has a lot of options, but we are squashing them down to 4
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetIdx

    our_edge_types = rdkit_edge_types.clone()

    # convert all bond types other than single, double, and triple to single
    # TODO we can use classes/enum here to make this more explicit
    our_edge_types[rdkit_edge_types > BondType.TRIPLE] = BondType.SINGLE

    # specially set aromatic to double
    our_edge_types[rdkit_edge_types == BondType.AROMATIC] = BondType.DOUBLE

    # set zerobond type to 0 (earlier we set all unconnected bonds to this bondtype)
    our_edge_types[rdkit_edge_types == BondType.ZERO] = 0.0

    if verbose:
        print("OUR_EDGE_TYPES:")
        print(our_edge_types)
        print()

    return our_edge_types


def main():
    tqdm.pandas() # enable progress bars in pandas
    df = pd.read_csv("https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv", nrows=1000)

    df['mol'] = df.smiles.progress_apply(Chem.MolFromSmiles)
    df['canonical_smiles'] = df.mol.progress_apply(Chem.MolToSmiles)

    print("Unique non-canonical smiles:", len(df.smiles.unique()))
    print("Unique canonical smiles:", len(df.canonical_smiles.unique()))

    # strip needed because a \n is added to the 'smiles' field due to dataset
    print("Number of changes:", len(df[(df.smiles.str.strip() != df.canonical_smiles)]))

    Draw.MolsToGridImage(df.mol[0:4],molsPerRow=2,legends=list(df.canonical_smiles[0:4]))

    a2t = dict() # atom to token
    t2a = dict() # token to atom
    e2t = dict() # edge to token
    t2e = dict() # token to edge
    
    # for a given molecule, what is the maximum amount of atoms which are present 
    # used to set the output matrix size when doing softmax
    max_atoms = float('-inf') 

    # RDKit has a lot of options here, but we will constrain to these 4
    # and maybe make a note of which molecules have bonds we wouldn't expect
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetIdx
    # note that we are treating aromatic as a "double" bond
    BOND_TYPES = ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
    for idx, bond_type in enumerate(BOND_TYPES):
        e2t[bond_type] = idx
        t2e[idx] = bond_type

    bond_types_in_dataset = set()

    for mol in list(df.mol):
        max_atoms = max(max_atoms, mol.GetNumAtoms())
        
        for atom in mol.GetAtoms():
            
            symbol = atom.GetSymbol()
            
            if symbol not in a2t:
                t = len(a2t)
                a2t[atom.GetSymbol()] = t
                t2a[t] = atom.GetSymbol()

            for bond in atom.GetBonds():
                bond_types_in_dataset.add(bond.GetBondType().name)

    print("Atom to token:", a2t)
    print("Token to atom:", t2a)
    print("Edge to token:", e2t)
    print("Token to edge:", t2e)
    print("Bond types in dataset:", bond_types_in_dataset)
    print("Max number of atoms in any given molecule:", max_atoms)

    ##################################################
    # Sanity check
    test_mol = Chem.MolFromSmiles("C(O)=CN")
    plt.imshow(Draw.MolToImage(test_mol))


    atom_info(test_mol, verbose=True)
    bond_info(test_mol, verbose=True)
    (_, edge_attr), (_, fc_edge_attr) = connect_info(test_mol, verbose=True)
    node_feature_matrix(test_mol, a2t, verbose=True )
    map_rdkit_bond_types(edge_attr, verbose=True)
    map_rdkit_bond_types(fc_edge_attr, verbose=True)
    ###################################################

    def prepare_df(mol):
        (edge_index, edge_attr), (fc_edge_index, y_fc_edge_attr)  = connect_info(mol)
        # y_boa is y labels for bag-of-atoms ->
        # a (vocab_size,) shape labels that shows the number of atoms for each atom 
        x, y_boa = node_feature_matrix(mol, a2t) 
        edge_attr = map_rdkit_bond_types(edge_attr)
        y_fc_edge_attr = map_rdkit_bond_types(y_fc_edge_attr)
        
        # note that PyG will automatically increment fc_edge_index
        # as per the following doc:
        # https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Batch.html#torch_geometric.data.Batch
        data = Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y_boa=y_boa,
                    fc_edge_index=fc_edge_index,
                    y_fc_edge_attr=y_fc_edge_attr)
        data.validate(raise_on_error=True)
        return data

    df["data"] = df.mol.progress_apply(prepare_df)
    
    return df, a2t, t2a, e2t, t2e, max_atoms
    
if __name__ == "__main__":
    main()