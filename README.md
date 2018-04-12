# rl_graph_generation

## Installation

- Install rdkit, please refer to the offical website, using anaconda is recommended
- Install mpi4py, `conda install mpi4py`, install networkx `pip install networkx=1.11`, install opencv `conda install opencv` (if needed).
- Install OpenAI baseline dependencies, run `pip install -e .` in the `rl-baseline` folder
- Install customized molecule gym environment, run `pip install -e .` in the `gym-molecule` folder



## Code structure

- Molecule environment code is in `gym-molecule/gym_molecule/envs/molecule.py`
- RL related code is in `rl-baselines/baselines/ppo1` folder. `gcn_policy.py` is the GCN policy network; `pposgd_simple_gcn.py` is the PPO algorithm specifically tuned for GCN policy; `run_molecule.py` is the main code for running the program.



## Run

- cd into `rl-baselines/baselines/ppo1`, then run `run_molecule.py`
- The code is using CPU solely. To allow for multiple processing, please run `mpirun -np 8 python run_molecule.py 2>/dev/null` (8 processes, and don't show warning info!!).