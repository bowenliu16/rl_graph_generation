import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from rdkit import Chem
import gym_molecule


class MoleculeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        possible_atoms = ['C', 'N', 'O']
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE]
        self.mol = Chem.RWMol()
        self.possible_atom_types = np.array(possible_atoms)  # dim d_n. Array that
        # contains the possible atom symbols strs
        self.possible_bond_types = np.array(possible_bonds, dtype=object)  # dim
        # d_e. Array that contains the possible rdkit.Chem.rdchem.BondType objects
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
            return -1  # arbitrary choice

    def reset(self):
        self.mol = Chem.RWMol()
        self.current_atom_idx = None
        self.total_atoms = 0
        self.total_bonds = 0

    def render(self, mode='human', close=False):
        return

    def _add_atom(self, action):
        """
        Adds an atom
        :param action: one hot np array of dim d_n, where d_n is the number of
        atom types
        :return:
        """
        assert action.shape == (len(self.possible_atom_types),)
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
        other_atom_idx = int(np.argmax(action.sum(axis=1)))  # b/c
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
        m = Chem.MolFromSmiles(s)  # implicitly performs sanitization
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


if __name__ == '__main__':
    env = gym.make('molecule-v0') # in gym format
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
    A,E,F = env.get_matrices()
    print('A',A)
    print('E',E)
    print('F',F)
    print(env.check_valency())
    print(env.check_chemical_validity())
