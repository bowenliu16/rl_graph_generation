#!/bin/bash
conda install -y mpi4py
conda install -y -c rdkit rdkit
pip install networkx==1.11
pip install matplotlib==3.5.3
pip install -e rl-baselines
pip install -e gym-molecule