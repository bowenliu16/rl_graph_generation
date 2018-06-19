# Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation
This repository is the official Tensorflow implementation of "Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation".

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, Bowen Liu\*, [Rex Ying](https://cs.stanford.edu/people/rexy/), [Vijay Pande](https://pande.stanford.edu/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation](https://arxiv.org/abs/1806.02473)

## Installation
- Install rdkit, please refer to the offical website for further details, using anaconda is recommended:
```bash
conda create -c rdkit -n my-rdkit-env rdkit
```
- Install mpi4py, networkx:
```bash
conda install mpi4py
pip install networkx=1.11
```
- Install OpenAI baseline dependencies:
```bash
cd rl-baselines
pip install -e .
```
- Install customized molecule gym environment:
```bash
cd gym-molecule
pip install -e.
```


## Code description
There are 4 important files:
- `run_molecule.py` is the main code for running the program. You may tune all kinds of hyper-parameters there.
- The molecule environment code is in `gym-molecule/gym_molecule/envs/molecule.py`.
- RL related code is in `rl-baselines/baselines/ppo1` folder: `gcn_policy.py` is the GCN policy network; `pposgd_simple_gcn.py` is the PPO algorithm specifically tuned for GCN policy.



## Run
- single process run
```bash
python run_molecule.py
```
- mutiple processes run
```bash
mpirun -np 8 python run_molecule.py 2>/dev/null
```
`2>/dev/null` will hide the warning info provided by rdkit package.

We highly recommend using tensorboard to monitor the training process. To do this, you may run
```bash
tensorboard --logdir runs
```

All the generated molecules along the training process are stored in the `molecule_gen` folder, each run configuration is stored in a different csv file. Molecules are stored using SMILES strings, along with the desired properties scores.