import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm.auto import tqdm
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from .utils import atom_info, bond_info, connect_info, node_feature_matrix, map_rdkit_bond_types


    
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

    test_mol = Chem.MolFromSmiles("C(O)=CN")
    plt.imshow(Draw.MolToImage(test_mol))


    atom_info(test_mol, verbose=True)
    bond_info(test_mol, verbose=True)
    adj_matrix, (edge_index, rdkit_edge_types) = connect_info(test_mol, verbose=True)
    x = node_feature_matrix(test_mol, a2t, verbose=True )
    edge_attr = map_rdkit_bond_types(rdkit_edge_types, verbose=True)


    def prepare_df(mol):
        adj_matrix, (edge_index, rdkit_edge_types) = connect_info(mol)
        x = node_feature_matrix(mol, a2t)
        edge_attr = map_rdkit_bond_types(rdkit_edge_types)
        
        # need to create the y labels

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.validate(raise_on_error=True)
        return data

    df["data"] = df.mol.progress_apply(prepare_df)
    
    return df, a2t, t2a, e2t, t2e, max_atoms
    
if __name__ == "__main__":
    main()