__author__ = "Bowen Liu"
__copyright__ = "Copyright 2018, Stanford University"

# import networkx as nx
import numpy as np
from rdkit import Chem  # for debug

class MoleculeEnvironment():
  def __init__(self, possible_atoms, possible_bonds):
    self.mol = Chem.RWMol()
    self.possible_atom_types = np.array(possible_atoms)  # dim d_n. Array that
    # contains the possible atom symbols strs
    self.possible_bond_types = np.array(possible_bonds, dtype=object)  # dim
    # d_e. Array that contains the possible rdkit.Chem.rdchem.BondType objects
    self.current_atom_idx = None
    self.total_atoms = 0
    self.total_bonds = 0

  def reset(self):
    self.mol = Chem.RWMol()
    self.current_atom_idx = None
    self.total_atoms = 0
    self.total_bonds = 0

  def step(self, action, action_type):
    """
    Perform a given action
    :param action:
    :param action_type:
    :return: reward of 1 if resulting molecule graph does not exceed valency,
    -1 if otherwise
    """
    if action_type == 'add_atom':
      self._add_atom(action)
    elif action_type == 'modify_bond':
      self._modify_bond(action)
    else:
      raise ValueError('Invalid action')

    # calculate rewards
    if self.check_valency():
      return 1  # arbitrary choice
    else:
      return -1 # arbitrary choice

  def _add_atom(self, action):
    """
    Adds an atom
    :param action: one hot np array of dim d_n, where d_n is the number of
    atom types
    :return:
    """
    assert action.shape == (len(self.possible_atom_types), )
    atom_type_idx = np.argmax(action)
    atom_symbol = self.possible_atom_types[atom_type_idx]
    self.current_atom_idx = self.mol.AddAtom(Chem.Atom(atom_symbol))
    self.total_atoms += 1

  def _modify_bond(self, action):
    """
    Adds or modifies a bond (currently no deletion is allowed)
    :param action: np array of dim N-1 x d_e, where N is the current total
    number of atoms, d_e is the number of bond types
    :return:
    """
    assert action.shape == (self.current_atom_idx, len(self.possible_bond_types))
    other_atom_idx = int(np.argmax(action.sum(axis=1))) # b/c
    # GetBondBetweenAtoms fails for np.int64
    bond_type_idx = np.argmax(action.sum(axis=0))
    bond_type = self.possible_bond_types[bond_type_idx]

    # if bond exists between current atom and other atom, modify the bond
    # type to new bond type. Otherwise create bond between current atom and
    # other atom with the new bond type
    bond = self.mol.GetBondBetweenAtoms(self.current_atom_idx, other_atom_idx)
    if bond:
      bond.SetBondType(bond_type)
    else:
      self.mol.AddBond(self.current_atom_idx, other_atom_idx, order=bond_type)
      self.total_bonds += 1

  def get_num_atoms(self):
    return self.total_atoms

  def get_num_bonds(self):
    return self.total_bonds

  # TODO(Bowen): check
  def check_chemical_validity(self):
    """
    Checks the chemical validity of the mol object. Existing mol object is
    not modified
    :return: True if chemically valid, False otherwise
    """
    s = Chem.MolToSmiles(self.mol, isomericSmiles=True)
    m = Chem.MolFromSmiles(s) # implicitly performs sanitization
    if m:
      return True
    else:
      return False

  # TODO(Bowen): check
  def check_valency(self):
    """
    Checks that no atoms in the mol have exceeded their possible
    valency
    :return: True if no valency issues, False otherwise
    """
    try:
      Chem.SanitizeMol(self.mol,
                            sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
      return True
    except ValueError:
      return False

  def get_matrices(self):
    """
    Get the adjacency matrix, edge feature matrix and, node feature matrix
    of the current molecule graph
    :return: np arrays: adjacency matrix, dim n x n; edge feature matrix,
    dim n x n x d_e; node feature matrix, dim n x d_n
    """
    A_no_diag = Chem.GetAdjacencyMatrix(self.mol)
    A = A_no_diag + np.eye(*A_no_diag.shape)

    n = A.shape[0]

    d_n = len(self.possible_atom_types)
    F = np.zeros((n, d_n))
    for a in self.mol.GetAtoms():
      atom_idx = a.GetIdx()
      atom_symbol = a.GetSymbol()
      float_array = (atom_symbol == self.possible_atom_types).astype(float)
      assert float_array.sum() != 0
      F[atom_idx, :] = float_array

    d_e = len(self.possible_bond_types)
    E = np.zeros((n, n, d_e))
    for b in self.mol.GetBonds():
      begin_idx = b.GetBeginAtomIdx()
      end_idx = b.GetEndAtomIdx()
      bond_type = b.GetBondType()
      float_array = (bond_type == self.possible_bond_types).astype(float)
      assert float_array.sum() != 0
      E[begin_idx, end_idx, :] = float_array
      E[end_idx, begin_idx, :] = float_array

    return A, E, F

# for testing get_matrices
def matrices_to_mol(A, E, F, node_feature_list, edge_feature_list):
  """
  Converts matrices A, E, F to rdkit mol object
  :param A: adjacency matrix, numpy array, dim k x k. Entries are either 0 or 1
  :param E: edge attribute matrix, numpy array, dim k x k x de. Entries are
  edge wise probabilities
  :param F: node attribute matrix, numpy array, dim k x dn. Entries are node
  wise probabilities
  :param node_feature_list: list of d_n elements that specifies possible
  atomic symbols
  :param edge_feature_list: list of d_e elements that specifies possible rdkit
  bond types
  :return: rdkit mol object
  """
  k = A.shape[0]

  rw_mol = Chem.RWMol()

  matrix_atom_idx_to_mol_atom_idx = {}
  for l in range(k):
    if A[l, l] == 1.0:
      atom_feature = F[l, :]
      atom_symbol = node_feature_list[np.argmax(atom_feature)]
      atom = Chem.Atom(atom_symbol)
      mol_atom_idx = rw_mol.AddAtom(atom)
      matrix_atom_idx_to_mol_atom_idx[l] = mol_atom_idx

  matrix_atom_idxes = matrix_atom_idx_to_mol_atom_idx.keys()
  for i in range(len(matrix_atom_idxes) - 1):
    for j in range(i + 1, len(matrix_atom_idxes)):
      if A[i, j] == 1.0:
        bond_feature = E[i, j, :]
        bond_type = edge_feature_list[np.argmax(bond_feature)]
        begin_atom_idx = matrix_atom_idx_to_mol_atom_idx[i]
        end_atom_idx = matrix_atom_idx_to_mol_atom_idx[j]
        rw_mol.AddBond(begin_atom_idx, end_atom_idx, order=bond_type)

  return rw_mol.GetMol()

# TESTS

# molecule construction test
possible_atoms = ['C', 'N', 'O']
possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE]

env = MoleculeEnvironment(possible_atoms, possible_bonds)
# add carbon
env.step(np.array([1, 0, 0]), 'add_atom')
# add carbon
env.step(np.array([1, 0, 0]), 'add_atom')
# add double bond between carbon 1 and carbon 2
env.step(np.array([[0, 1, 0]]), 'modify_bond')
# add carbon
env.step(np.array([1, 0, 0]), 'add_atom')
# add single bond between carbon 2 and carbon 3
env.step(np.array([[0, 0, 0], [1, 0, 0]]), 'modify_bond')
# add oxygen
env.step(np.array([0, 0, 1]), 'add_atom')
# add single bond between carbon 3 and oxygen
env.step(np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]), 'modify_bond')

assert Chem.MolToSmiles(env.mol, isomericSmiles=True) == 'C=CCO'

# test get_matrices 1
A, E, F = env.get_matrices()
assert Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
                                        possible_bonds), isomericSmiles=True) \
       == 'C=CCO'

# molecule check valency test
env = MoleculeEnvironment(possible_atoms, possible_bonds)
# add carbon
r = env.step(np.array([1, 0, 0]), 'add_atom')
# add oxygen
r = env.step(np.array([0, 0, 1]), 'add_atom')
# add single bond between carbon and oxygen 1
r = env.step(np.array([[1, 0, 0]]), 'modify_bond')
# add oxygen
r = env.step(np.array([0, 0, 1]), 'add_atom')
# add single bond between carbon and oxygen 2
r = env.step(np.array([[1, 0, 0], [0, 0, 0]]), 'modify_bond')
# add oxygen
r = env.step(np.array([0, 0, 1]), 'add_atom')
# add single bond between carbon and oxygen 3
r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), 'modify_bond')
# add oxygen
r = env.step(np.array([0, 0, 1]), 'add_atom')
# add single bond between carbon and oxygen 4
r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
           'modify_bond')
# add oxygen
r = env.step(np.array([0, 0, 1]), 'add_atom')
assert r == 1
# add single bond between carbon and oxygen 4. This exceeds valency on C
r = env.step(np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
         'modify_bond')
assert r == -1

# test get_matrices 2
A, E, F = env.get_matrices()
assert Chem.MolToSmiles(matrices_to_mol(A, E, F, possible_atoms,
                                        possible_bonds), isomericSmiles=True) \
       == 'OC(O)(O)(O)O'

# # TODO(Bowen): unit test this. Esp molecules that have differing
# # chiral centers? Check if this accounts for differing E/Z bonds, or between
# # E/Z bonds and unspecified bonds.
# def check_smiles_match(smiles_1, smiles_2, ignore_stereochemistry=False):
#   """
#   Checks whether SMILES of mol 1 is the same as SMILES of mol 2. Uses InChI
#   comparison to account for different smiles forms. smiles_1 and smiles_2
#   must contain the same number of molecules, but can be > 1
#   :param mol_1:  SMILES string of molecule 1
#   :param mol_2:  SMILES string of molecule 2
#   :param ignore_stereochemistry: If True, then ignores R/S and or E/Z
#   configuration
#   :return:  True if mol_1 and mol_2 are valid molecules and are the same.
#   False if otherwise
#   """
#   mol_1 = Chem.MolFromSmiles(smiles_1)
#   mol_2 = Chem.MolFromSmiles(smiles_2)
#   if mol_1 and mol_2:
#     if not ignore_stereochemistry:  # account for stereocenters
#       mol_1_inchi = Chem.MolToInchi(mol_1)
#       mol_2_inchi = Chem.MolToInchi(mol_2)
#     else:  # ignore stereocenters. Clear stereochemistry by converting mol to
#       # smiles with no stereocenters, and then back to inchi
#       mol_1_inchi = Chem.MolToInchi(Chem.MolFromSmiles(
#         Chem.MolToSmiles(mol_1, isomericSmiles=False)))
#       mol_2_inchi = Chem.MolToInchi(Chem.MolFromSmiles(
#         Chem.MolToSmiles(mol_2, isomericSmiles=False)))
#     return mol_1_inchi == mol_2_inchi
#   else:  # either mol_1 or mol_2 is a None obj, which implies an invalid molecule
#     return False
#
# # TODO(Bowen): check
# def mol_to_graph(mol):
#   """
#   Convert an rdkit mol object to a nx graph. Very simple atm:
#   node attributes only include atom symbol {C, N, O, S, Cl} and CIP symbol {
#   None, R, S},
#   edge attributes only include bond type {SINGLE, DOUBLE, TRIPLE}
#   and E/Z symbol {None, E, Z}
#   :param mol: rdkit mol object
#   :return: nx graph object
#   """
#   # TODO(Bowen): refactor with dicts
#
#   # for correctness checks
#   possible_atoms = ['C', 'N', 'O', 'S', 'Cl', 'F']
#   possible_atom_stereo = [None, 'R', 'S']
#   possible_bonds = ['SINGLE', 'DOUBLE', 'TRIPLE'] # assume we kekulize the
#   # mol so no aromatic bonds
#   # possible_bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
#   possible_bond_stereo = ['STEREONONE', 'STEREOE', 'STEREOZ']
#
#   Chem.Kekulize(mol)  # kekulize the mol to convert any aromatic bonds to
#   # single or double bonds
#
#   # atom_symbol_list = []
#   # atom_stereo_list = []
#   atom_symbol_dict = {}
#   atom_stereo_dict = {}
#   for a in mol.GetAtoms():
#     atom_idx = a.GetIdx()
#     atom_symbol = a.GetSymbol()
#     assert atom_symbol in possible_atoms
#     atom_symbol_dict[atom_idx] = atom_symbol
#     # atom_symbol_list.append(atom_symbol)
#     if a.HasProp('_ChiralityPossible'):
#       atom_stereo = a.GetProp('_CIPCode')
#     else:
#       atom_stereo = None
#     assert atom_stereo in possible_atom_stereo
#     atom_stereo_dict[atom_idx] = atom_stereo
#     # atom_stereo_list.append(atom_stereo)
#
#   edge_list = []
#   # bond_type_list = []
#   # bond_stereo_list = []
#   bond_type_dict = {}
#   bond_stereo_dict = {}
#   for b in mol.GetBonds():
#     bond = (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
#     edge_list.append(bond)
#     bond_type = str(b.GetBondType())
#     assert bond_type in possible_bonds
#     bond_type_dict[bond] = bond_type
#     # bond_type_list.append(bond_type)
#     bond_stereo = str(b.GetStereo())
#     assert bond_stereo in possible_bond_stereo
#     bond_stereo_dict[bond] = bond_stereo
#     # bond_stereo_list.append(bond_stereo)
#
#   # create nx graph
#   G = nx.Graph()
#   G.add_edges_from(edge_list)
#   nx.set_node_attributes(G, 'atom_symbol', atom_symbol_dict)
#   nx.set_node_attributes(G, 'atom_stereo', atom_stereo_dict)
#   nx.set_edge_attributes(G, 'bond_type', bond_type_dict)
#   nx.set_edge_attributes(G, 'bond_stereo', bond_stereo_dict)
#
#   return G
#
# # TODO(Bowen): finish and check. The atom and bond stereo is incorrect. Wrong
# #  approach to do this. See 01/03/18
# def graph_to_mol(G):
#   """
#   Convert a nx graph object into a rdkit mol object. Very simple atm:
#   node attributes only include atom symbol {C, N, O, S, Cl} and CIP symbol {
#   None, R, S},
#   edge attributes only include bond type {SINGLE, DOUBLE, TRIPLE}
#   and E/Z symbol {None, E, Z}
#   :param G: nx graph object
#   :return: rdkit mol object
#   """
#   # for correctness checks
#   possible_atoms = ['C', 'N', 'O', 'S', 'Cl', 'F']
#   possible_atom_stereo = [None, 'R', 'S']
#   possible_bonds = ['SINGLE', 'DOUBLE', 'TRIPLE']  # assume we kekulize the
#   # mol so no aromatic bonds
#   # possible_bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
#   possible_bond_stereo = ['STEREONONE', 'STEREOE', 'STEREOZ']
#
#   rw_mol = Chem.RWMol()
#
#   # add atoms
#   for node in G.nodes(data=True):
#     atom_stereo = node[1]['atom_stereo']
#     assert atom_stereo in possible_atom_stereo
#     atom_symbol = node[1]['atom_symbol']
#     assert atom_symbol in possible_atoms
#
#     atom = Chem.Atom(atom_symbol)
#     if atom_stereo:
#       atom.SetProp('_CIPCode', atom_stereo) # just a string property
#     rw_mol.AddAtom(atom)
#
#   # add bonds
#   for edge in G.edges(data=True):
#     begin_atom_idx = edge[0]
#     end_atom_idx = edge[1]
#     bond_stereo = edge[2]['bond_stereo']  # use to set bond_stereo
#     assert bond_stereo in possible_bond_stereo
#     bond_type = edge[2]['bond_type']  # use to convert to
#     # 'rdkit.Chem.rdchem.BondType'
#     assert bond_type in possible_bonds
#
#     rdkit_bond_stereo = getattr(Chem.rdchem.BondStereo, bond_stereo)
#     rdkit_bond_type = getattr(Chem.rdchem.BondType, bond_type)
#
#     rw_mol.AddBond(begin_atom_idx, end_atom_idx, order=rdkit_bond_type)
#
#     # set the bond stereo property
#     rw_mol.GetBondBetweenAtoms(begin_atom_idx, end_atom_idx).SetStereo(rdkit_bond_stereo)
#
#   return rw_mol.GetMol()
#
# # test
# s = 'O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C' # lipitor
# mol = Chem.MolFromSmiles(s)
# Chem.Kekulize(mol)
# g = mol_to_graph(mol)
# # nx.drawing.nx_pylab.draw_networkx(g)
# mol_new = graph_to_mol(g)
# assert check_smiles_match(s, Chem.MolToSmiles(mol_new),
#                           ignore_stereochemistry=True)
#
# s_2 = 'C/C=C/[C@H](Cl)C(O)=O'
# mol_2 = Chem.MolFromSmiles(s_2)
# Chem.Kekulize(mol_2)
# g_2 = mol_to_graph(mol_2)
# mol_2_new = graph_to_mol(g_2)
# assert check_smiles_match(s_2, Chem.MolToSmiles(mol_2_new),
#                           ignore_stereochemistry=True)