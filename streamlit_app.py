import streamlit as st

"""
# _Current work in progress... check back soon_

# MoleGen - a generative Machine Learning model for chemical molecules

In this post, I will be replicating the model and results from the following 2019 paper: [A Two-Step Graph Convolutional Decoder for Molecule Generation](https://arxiv.org/pdf/1906.03412).

## Preparation - paper read

We start by reading the paper. As a first pass, we can make the following comments:

- The authors used the ZINC dataset
- The very first input to the model is a canonical SMILES representation -> this needs to be converted into a graph ([see page 13/26 of this paper](https://arxiv.org/pdf/1610.02415) for more info on the encoding process). We will also need to extract the position information
- We use a graph convolutional network (GCN) encoder to aggregate information from neighbors for each node and edge
- We then have a simple MLP to convert the output to a matrix and create an output such that we have a one-hot vector essentially of how many atoms of each atom type, and we choose the index that has the maximum
- Then we need to decode this and so we start with a fully connected graph (I'm assuming the edge types are randomized).

As a first step, we want to look at our data and build the encoder.

## Dataset exploration

Next, we will load the ZINC dataset and explore it. We will only load a subset first.

Note that we can't use the builtin ZINC dataset from PyTorch because we lose the SMILES representation and we don't know the mapping from integer encoding to the atom and edge type. I think this is [somewhere explained here though](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559) or [here](https://zinc15.docking.org/catalogs/home/). However, we don't need the entire ZINC. Only a small subset.

We will use the [ZINC-250k dataset](https://www.kaggle.com/datasets/basu369victor/zinc250k) which is available on Kaggle. Note for debugging we will only load 1000 molecules.

We will use [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html) to convert these SMILES strings into canonical strings and to extract the connectivity for PyTorch geometric. But first, we have to load in our data. We will use Pandas dataframes for this. Note that most of these libraries are already pre-installed on Google colab

"""