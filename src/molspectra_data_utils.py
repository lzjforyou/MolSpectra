import torch
import os
import pickle
from rdkit import Chem
from rdkit.Chem import rdchem
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from communities.algorithms import louvain_method,girvan_newman
from torch_geometric.data import Data
import numpy as np
from massformer.pyg_features import atom_featurizer,bond_featurizer
from copy import deepcopy
from torch import nn

def trans_to_adj(graph): # graph: A NetworkX format graph object
    # Convert networkx format graph to adjacency matrix
    graph.remove_edges_from(nx.selfloop_edges(graph)) # Remove all self-loop edges from the graph
    nodes = range(len(graph.nodes)) # Get node range
    return np.array(nx.to_numpy_array(graph, nodelist=nodes)) # Generate adjacency matrix in node order, return NumPy array representation of adjacency matrix

num = 0
maxlen=0

def compute_posenc_stats(data, pe_types=None, is_undirected=True,hdse=3):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph One or more PE types to compute
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    global num
    # num += 1
    # if num==101:
    #     s()
    global maxlen

    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.

    Maxnode = 500
    flag = 0
    if(data.x.shape[0]>=Maxnode):
        print(data)
        raise Exception("Maxnode exceed")

    G = to_networkx(data, to_undirected=True) # Convert graph to undirected graph
    Maxdis = 10
    Maxspd = 30
    SPD = [[0 for j in range(Maxnode)] for i in range(N)] # Initialize shortest path distance matrix
    dist = [[0 for j in range(Maxnode)] for i in range(N)] # Initialize distance matrix
    G_all = G
    # print(list(G.nodes()))
    # adj_matrix = nx.to_numpy_array(G_all)
    if(hdse==0):
        # pass Use METIS graph partitioning algorithm to partition graph into specified parts (nparts=5)
        # _, communities = nxmetis.partition(G, nparts=5)
        # print(communities)
        pass
    #Spectral clustering
    elif(hdse==1): # Spectral clustering
        # print(G.nodes(),G.edges())
        # adj_matrix = nx.to_numpy_array(G) # Convert graph adjacency matrix to NumPy array
        # communities = spectral_clustering(adj_matrix, 5) # Use spectral clustering algorithm to partition graph into 5 communities
        # print(communities)
        pass
    #Spectral-cut
    elif(hdse==2): # This code implements community detection based on Spectral-cut. Through spectral analysis and cutting operations on the graph, the graph is gradually divided into multiple subgraphs
        # graph = pygsp.graphs.Graph(nx.adjacency_matrix(G).todense()) # Input is a NetworkX graph G, converted to dense matrix via nx.adjacency_matrix as input for spectral cut
        # Cut the input graph to generate multi-level graph structure, K=10: number of graph layers after cutting, r=0.9: cutting ratio, controls node reduction rate after each cut, method='algebraic_JC': use algebraic method for cutting
        # C: cutting matrix, describes mapping relationship between original graph and cut graph. Gc: cut graph. Third and fourth return values are cutting metadata, not used
        # C, Gc, _, _ = coarsen(graph, K=10, r=0.9, method='algebraic_JC')
        # dense_matrix = C.toarray() # Convert cutting matrix C to NumPy dense matrix form
        # row_indices, col_indices = dense_matrix.nonzero() # Get row and column indices of non-zero elements in cutting matrix
        # max_cluster = np.max(row_indices) + 1 # Calculate maximum cluster number (i.e., number of communities max_cluster)
        # cluster_list = [[] for _ in range(max_cluster)] # Initialize an empty list with length equal to number of communities, each sublist represents a community
        # for i in range(len(row_indices)): # Traverse indices of non-zero elements, assign node numbers to corresponding communities:
        #     cluster = col_indices[i] # cluster represents node number
        #     point = row_indices[i] # point represents cluster number
        #     cluster_list[point].append(cluster)
        # communities = cluster_list # communities is a list where each sublist represents nodes contained in a community
        pass
    #Girvan-Newman, Girvan-Newman algorithm is a community structure detection algorithm based on edge betweenness. It iteratively removes edges with highest betweenness value in the graph, gradually dividing the graph to form different communities
    elif(hdse==3):  
        adj_matrix = nx.to_numpy_array(G)
        communities, _ = girvan_newman(adj_matrix)
    #Louvains Method, Louvain algorithm is a fast heuristic method for detecting community structure, based on modularity optimization
    # It generates nested community partitions by repeatedly merging nodes, ultimately obtaining community partition with maximum modularity
    elif(hdse==4):
        adj_matrix = nx.to_numpy_array(G)
        communities, _ = louvain_method(adj_matrix)
    # Use METIS graph partitioning algorithm to partition graph into 10 subgraphs, this method aims to minimize edge weight (cut cost) between partition subgraphs while maintaining connectivity within subgraphs
    elif(hdse==10):
        # _, communities = nxmetis.partition(G, nparts=10)
        pass
        
    # This code implements further operations after community detection, including constructing quotient graph based on communities, calculating shortest path length (SPD) between nodes in graph, and limiting maximum path length
    # if(cfg.dataset.name=='ogbg-molhiv' or cfg.dataset.name=='subset' or cfg.dataset.name=='ogbg-molpcba'):
    #     communities, _ = girvan_newman(adj_matrix)
    # else:
    #     communities, _ = louvain_method(adj_matrix)

    # Generate quotient graph based on community partition, quotient graph nodes are communities, each community corresponds to a node, if there is connection between two communities, there is also an edge between nodes in quotient graph
    # G_all: original graph. communities: community partition result, indicates which nodes belong to same community. relabel=True: renumber quotient graph nodes to 0, 1, 2, ...
    # Quotient graph M is a higher-level abstraction used to represent connection relationships between communities
    M = nx.quotient_graph(G_all, communities, relabel=True)
    # print(communities)
    # if(len(communities)>80):
    #     print(len(communities))
    #     raise Exception("communities exceed")
    dict_graph = {} # Build a dictionary dict_graph that maps each node to its community number
    for i in range(len(communities)):
        for j in communities[i]:
            dict_graph[j] = i
    length = dict(nx.all_pairs_shortest_path_length(G_all)) # Calculate shortest path length between all node pairs in graph G_all, return a nested dictionary
    # print(length)
    # Construct an N×N shortest path length matrix (SPD)
    for i in range(N):
        for j in range(N):
            if(j in length[i]):
                SPD[i][j] = length[i][j]
                if(SPD[i][j]>=Maxspd):
                    SPD[i][j] = Maxspd
                maxlen = max(SPD[i][j],maxlen) # (after limitation)
            else:
                SPD[i][j] = Maxspd
    
    G = M # Quotient graph M represents community structure of original graph G_all, viewing communities as single nodes
    length = dict(nx.all_pairs_shortest_path_length(G)) # Calculate shortest path length between communities in quotient graph
    for i in range(N):
        for j in range(N):
            if(dict_graph[j] in length[dict_graph[i]]):
                dist[i][j] = length[dict_graph[i]][dict_graph[j]]
                if(dist[i][j]>=Maxdis):
                    dist[i][j] = Maxdis
            else:
                dist[i][j] = Maxdis
    laplacian_norm_type = 'none'
    
    # laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower() # Read Laplacian matrix normalization type from configuration parameters
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        # Convert calculated Laplacian matrix to SciPy sparse matrix format
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        # L.toarray(): Convert sparse matrix L to dense matrix for using NumPy's eigendecomposition function
        evals, evects = np.linalg.eigh(L.toarray())
        
        if 'LapPE' in pe_types:
            max_freqs=8
            eigvec_norm="L2"
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=8
            eigvec_norm="L2"
        # Extract Laplacian eigenvalues and eigenvectors
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)
        abs_pe = data.EigVecs

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = list(range(1, 21))
        if len(kernel_param) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing # (Num nodes) x (K steps)
        abs_pe = rw_landing

    # data.SPD = torch.tensor(SPD).long()
    # data.dist = torch.tensor(dist).long()
    SPD = torch.tensor(SPD).long() # Shortest path length between node pairs
    dist = torch.tensor(dist).long() # Distance between community pairs

    complete_edge_index_dist = dist[:N,:N] # Extract first N×N part of community distance matrix
    complete_edge_index_dist = complete_edge_index_dist.reshape(-1) # One-dimensional tensor
    complete_edge_index_SPD = SPD[:N,:N]
    complete_edge_index_SPD = complete_edge_index_SPD.reshape(-1)
    data.complete_edge_dist = complete_edge_index_dist
    data.complete_edge_SPD = complete_edge_index_SPD
    s = torch.arange(N)
    # s.repeat_interleave(N): Repeat each node number N times. s.repeat(N): Repeat entire sequence N times. Stack two 1D tensors row-wise to generate a 2×(N×N) matrix
    data.complete_edge_index = torch.vstack((s.repeat_interleave(N), s.repeat(N)))

    return data


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


def graph2data(graph, hdse=3):
    """
    Convert a graph dictionary structure to PyTorch Geometric Data object, and compute RWSE encoding.

    Args:
        graph: Graph dictionary structure containing edge_index, edge_feat, node_feat, num_nodes, etc.
        is_undirected: Whether to treat graph as undirected.
        hdse: Algorithm option for community detection (default is 3, i.e., Girvan-Newman algorithm).
        
    Returns:
        data: Extended PyTorch Geometric Data object.
    """
    # Create PyTorch Geometric Data object
    data = Data()
    assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
    assert (len(graph['node_feat']) == graph['num_nodes'])
    data.__num_nodes__ = int(graph['num_nodes'])
    data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.float32)
    data.x = torch.from_numpy(graph['node_feat']).to(torch.float32)
    data.y = torch.Tensor([-1])  # dummy

    # Compute RWSE and EquivStableLapPE encoding
    pe_types = ['RWSE', 'EquivStableLapPE']
    # pe_types = ['RWSE']
    data = compute_posenc_stats(data, pe_types=pe_types, is_undirected=True, hdse=hdse)


    return data


def Mol_preprocess(mol,idx):
    graph = smiles2graph(mol) # smiles2graph is responsible for converting SMILES string or RDKit molecule object (rdchem.Mol) to graph data object
    data = graph2data(graph) # graph2data converts a graph dictionary structure to PyTorch Geometric Data object
    data.idx = idx
    # item = preprocess_item(data) # Calculate additional feature data for Graphormer
    return data








def test_pyg_preprocess():
    """
    Test the functionality of pyg_preprocess function.
    """
    # Define a test molecule (SMILES string)
    mol = "CCO"  # Ethanol molecule
    idx = 0  # Data index

    # Call pyg_preprocess function
    
    data = Mol_preprocess(mol, idx)


    # Print generated Data object
    print("Generated Data object:")
    print(data)

    # Verify whether key attributes of Data object exist
    required_attributes = ['x', 'edge_index', 'edge_attr', 'pestat_RWSE', 'complete_edge_dist', 'complete_edge_SPD']
    missing_attributes = [attr for attr in required_attributes if not hasattr(data, attr)]
    
    if len(missing_attributes) > 0:
        print("Test failed, missing following attributes:", missing_attributes)
    else:
        print("Test passed, all expected attributes exist!")

    # Check basic content of Data object
    print("\nDetailed information of Data object:")
    print(f"x (node features): {data.x.shape}")
    print(f"edge_index (edge indices): {data.edge_index.shape}")
    print(f"edge_attr (edge features): {data.edge_attr.shape}")
    print(f"pestat_RWSE (RWSE encoding): {data.pestat_RWSE.shape}")
    print(f"complete_edge_dist (edge distance matrix): {data.complete_edge_dist.shape}")
    print(f"complete_edge_SPD (shortest path distance matrix): {data.complete_edge_SPD.shape}")
    print(f"complete_edge_index: {data.complete_edge_index.shape}")










def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    # Normalize eigenvectors according to normalization method specified by eigvec_norm (e.g., L2 normalization)
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs: # If number of nodes is less than max_freqs, need to pad eigenvectors to ensure consistent output dimensions
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2) # Shape (num_nodes, max_freqs, 1)
    # EigVecs: A tensor with shape (num_nodes, max_freqs)
    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.
    This code implements calculation of random walk landing probabilities. Based on input graph and specified random walk steps k, it generates a matrix representing probability of each node returning to itself at these time steps

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


# If the script is run directly, execute the test function
if __name__ == "__main__":
    test_pyg_preprocess()
