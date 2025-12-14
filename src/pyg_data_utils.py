import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
import torch_geometric.transforms as T
from pyg_features import atom_featurizer,bond_featurizer

def smiles2graph(smiles_or_mol):
    """
    Converts SMILES string or rdkit mol to graph Data object
    :input: SMILES string (str) or rdkit mol
    :return: graph object
    smiles2graph is responsible for converting SMILES string or RDKit molecule object (rdchem.Mol) to graph data object
    """

    # changed this function to work with mols too
    if isinstance(smiles_or_mol, str):
        mol = Chem.MolFromSmiles(smiles_or_mol)
    else:
        assert isinstance(smiles_or_mol, rdchem.Mol)
        mol = smiles_or_mol

    # mol = Chem.AddHs(mol)

    # atoms

    node_features = np.array([atom_featurizer(atom) for atom in mol.GetAtoms()])
    # x= torch.tensor(node_features, dtype=torch.float32)
    x = np.array(node_features, dtype=np.float32)
    # bond
    num_bond_features = 6
    if len(mol.GetBonds()) > 0:
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_features = bond_featurizer(bond)
            edges_list.append((start, end))
            edge_features_list.append(bond_features)
            edges_list.append((end, start))
            edge_features_list.append(bond_features)

        edge_index = np.array(edges_list, dtype=np.int64).T
        edge_attr = np.array(edge_features_list, dtype=np.float32)
    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype=np.float32)

        

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def graph2data(graph):
    """ taken from process() in https://github.com/snap-stanford/ogb/blob/master/ogb/lsc/pcqm4mv2_pyg.py """
    # graph2data converts a graph dictionary structure to PyTorch Geometric Data object
    data = Data()
    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.float32)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.float32)
    data.y = torch.Tensor([-1])  # dummy

    # Add random walk encoding
    transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')  # Set random walk length
    data = transform(data)

    return data


def pyg_preprocess(mol,idx):
    graph = smiles2graph(mol) # smiles2graph is responsible for converting SMILES string or RDKit molecule object (rdchem.Mol) to graph data object
    data = graph2data(graph) # graph2data converts graph dictionary to PyTorch Geometric Data object
    data.idx = idx
    # item = preprocess_item(data) 
    return data
