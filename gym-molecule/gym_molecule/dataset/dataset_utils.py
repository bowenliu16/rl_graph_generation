__author__ = "Bowen Liu"
__copyright__ = "Copyright 2018, Stanford University"

import pandas as pd
from rdkit import Chem

def load_dataset(path):
  """
  Loads gdb13 dataset from path to pandas dataframe
  :param path:
  :return:
  """
  df = pd.read_csv(path, header=None, names=['smiles'])
  return df

class gdb_dataset:
  """
  Simple object to contain the gdb13 dataset
  """
  def __init__(self, path):
    self.df = load_dataset(path)

  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, item):
    """
    Returns an rdkit mol object
    :param item:
    :return:
    """
    smiles = self.df['smiles'][item]
    mol = Chem.MolFromSmiles(smiles)
    return mol, smiles

# TESTS
path = 'gdb13.rand1M.smi.gz'
dataset = gdb_dataset(path)

print(len(dataset))
print(dataset[0])