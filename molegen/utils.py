
from rdkit.Chem import rdmolops
import torch
from torch_geometric.utils import dense_to_sparse


def atom_info(mol, verbose=False):
  """Extract RDKit info from Atoms"""
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
  """Extract RDKit info from bonds"""
  bonds = []

  if verbose:
    print("BOND_INFO:")
    print("INDEX BEGIN_ATOM END_ATOM BOND_TYPE BOND_NUM")

  for bond in mol.GetBonds():
    idx = bond.GetIdx()
    atom1 = bond.GetBeginAtomIdx()
    atom2 = bond.GetEndAtomIdx()
    bond_type = bond.GetBondType()
    bond_num = bond.GetBondTypeAsDouble() # TODO: need to convert this

    bonds.append((idx, atom1, atom2, bond_type, bond_num))

    if verbose:
      print(*bonds[-1])

  if verbose:
    print()

  return bonds

def connect_info(mol, verbose=False):
  """Get adjacency matrix from mol"""

  adj_matrix = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
  sparse_matrix = dense_to_sparse(torch.tensor(adj_matrix, dtype=torch.int32))

  if verbose:
    print("ADJ_MATRIX:")
    print(adj_matrix)
    print()

    print("SPARSE_MATRIX:")
    print(sparse_matrix)
    print()

  return adj_matrix, sparse_matrix

def node_feature_matrix(mol, a2t, verbose=False):
  """Returns a node feature matrix for PyTorch"""
  atoms = atom_info(mol)

  matrix = torch.zeros(len(atoms), 1, dtype=torch.int32)

  for atom in atoms:
    symbol, idx, _, _  = atom

    matrix[idx] = a2t[symbol]

  if verbose:
    print("NODE_FEATURE_MATRIX:")
    print(matrix)
    print()

  return matrix

def map_rdkit_bond_types(rdkit_edge_types, verbose=False):
  """Convert RDKit bond types to our edge type set"""
  # RDKit has a lot of options, but we are squashing them down to 4
  # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.Bond.GetIdx

  our_edge_types = rdkit_edge_types.clone()

  # convert all bond types other than single, double, and triple to single
  # TODO we can use classes/enum here to make this more explicit
  our_edge_types[rdkit_edge_types > 3] = 1.0

  # specially set aromatic to double
  our_edge_types[rdkit_edge_types == 12] = 2.0

  # set zerobond type to 0
  our_edge_types[rdkit_edge_types == 21] = 0.0

  if verbose:
    print("OUR_EDGE_TYPES:")
    print(our_edge_types)
    print()


  return our_edge_types