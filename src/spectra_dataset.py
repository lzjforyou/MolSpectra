import torch 
import torch.utils.data as th_data
import pandas as pd
import numpy as np
import os
from pprint import pprint
import dgl
import dgllife.utils as chemutils
import torch_geometric.data
from tqdm import tqdm
import itertools
from sklearn.decomposition import LatentDirichletAllocation
import ast
import data_utils
import molspectra_data_utils as molspectra_data_utils
import pyg_data_utils as pyg_data_utils
from misc_utils import EPS, np_temp_seed, np_one_hot, flatten_lol, none_or_nan


def np_one_hot(input, num_classes=None):
    """Numpy wrapper for one_hot encoding"""
    th_input = torch.as_tensor(input, device="cpu")
    th_oh = torch.nn.functional.one_hot(th_input, num_classes=num_classes)
    oh = th_oh.numpy()
    return oh


def data_to_device(data_d, device, non_blocking):
    """
    The main function of this function is to migrate tensors or graph data in a data dictionary to a specified device (such as GPU or CPU) to accelerate computation
    
    Parameters:
        data_d: Data dictionary
        device: Target device (cuda or cpu)
        non_blocking: Whether to use non-blocking transfer
    
    Returns:
        new_data_d: Data dictionary after device transfer
    """
    new_data_d = {}
    for k, v in data_d.items():
        if isinstance(v, torch.Tensor) or isinstance(v, dgl.DGLGraph) or isinstance(v, torch_geometric.data.Data):
            new_data_d[k] = v.to(device, non_blocking=non_blocking)
        elif isinstance(v, dict):
            new_v = {}
            for kk, vv in v.items():
                new_v[kk] = vv.to(device, non_blocking=non_blocking)
            new_data_d[k] = new_v
        else:
            new_data_d[k] = v
    return new_data_d


class TrainSubset(th_data.Subset):
    """Training subset wrapper class"""
    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])


class BaseDataset(th_data.Dataset):
    """Base dataset class for molecular spectroscopy data"""

    def __init__(self, *dset_types, **kwargs):
        """
        Initialize base dataset
        
        Parameters:
            *dset_types: Dataset types (hdse, fp, graph, pyg_data, dtnn, mat, smiles)
            **kwargs: Other configuration parameters
        """
        # Set dataset type flags
        self.is_hdse_data_dst = "hdse" in dset_types
        self.is_fp_dset = "fp" in dset_types
        self.is_graph_dset = "graph" in dset_types
        self.is_pyg_data_dset = "pyg_data" in dset_types
        self.is_dtnn_data_dset = "dtnn" in dset_types
        self.is_mat_data_dset = "mat" in dset_types
        self.is_smiles_dset = "smiles" in dset_types  # New SMILES dataset flag
        
        assert (self.is_pyg_data_dset or self.is_fp_dset or self.is_hdse_data_dst or 
                self.is_dtnn_data_dset or self.is_mat_data_dset or self.is_smiles_dset)
        
        # Dynamically set passed keyword arguments as class attributes for flexible configuration
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        # If SMILES dataset, initialize vocabulary-related parameters
        if self.is_smiles_dset:
            self._setup_smiles_vocab()
        
        # Read data file, data file path specified by datapath attribute
        if self.datapath.endswith('.csv'):
            self.df = pd.read_csv(self.datapath)
        elif self.datapath.endswith('.pkl'):
            self.df = pd.read_pickle(self.datapath)
        else:
            raise ValueError('Unsupported file format, only .csv and .pkl supported')
        
        # ========== New: Filtering logic based on draw_attn ==========
        if self.draw_attn:  # If draw_attn is True, perform filtering
            common_smiles_path = 'YOUR_PATH/draw_attention.csv'  # Replace with your actual path
            # Check if file exists
            if not os.path.exists(common_smiles_path):
                raise FileNotFoundError(f"Filter file does not exist: {common_smiles_path}")
            
            # Read common SMILES list
            common_df = pd.read_csv(common_smiles_path)
            
            # Check if column name exists
            if 'canonical_smiles' not in common_df.columns:
                raise ValueError(f"Filter file missing 'canonical_smiles' column, current columns: {common_df.columns.tolist()}")
            
            # Extract common SMILES set
            common_smiles_set = set(common_df['canonical_smiles'].dropna().unique())
            
            # Check if self.df has smiles column
            if not hasattr(self, 'smiles') or self.smiles not in self.df.columns:
                raise ValueError(f"Dataset missing SMILES column, current columns: {self.df.columns.tolist()}")
            
            # Count before filtering
            original_count = len(self.df)
            
            # Filter: only keep samples in common SMILES set
            self.df = self.df[self.df[self.smiles].isin(common_smiles_set)].reset_index(drop=True)
            
            # Count after filtering
            filtered_count = len(self.df)
            
            # Print filtering information
            print(f"[Data Filtering] draw_attn=True, SMILES filtering enabled")
            print(f"  - Common SMILES count: {len(common_smiles_set)}")
            print(f"  - Samples before filtering: {original_count}")
            print(f"  - Samples after filtering: {filtered_count}")
            print(f"  - Retention rate: {filtered_count/original_count*100:.2f}%")
        else:
            print(f"[Data Filtering] draw_attn=False, skipping SMILES filtering, using all data")
        
        # Configure metadata dictionary to map some string data to numeric indices
        self._setup_spec_metadata_dicts()

    def _setup_spec_metadata_dicts(self):
        """Setup spectrum metadata dictionaries"""
        state_type_list = self.all_state_type
        self.state_type_c2i = {string: i for i, string in enumerate(state_type_list)}
        self.state_type_i2c = {i: string for i, string in enumerate(state_type_list)}
        self.num_state_type = len(state_type_list)

    def __getitem__(self, idx):
        """Get single sample"""
        spec_entry = self.df.iloc[idx]
        data = self.process_entry(spec_entry)
        return data

    def __len__(self):
        """Get dataset length"""
        return self.df.shape[0]

    def normalize(self, s):
        """Normalize input sequence s to 0~1 range and return normalized list"""
        maxval = max(s)             # Find maximum value in sequence
        scale = 1 / maxval          # Calculate scaling factor
        if(maxval == 0):            # If maximum value is 0, avoid division by zero
            scale = 0
        return([j * scale for j in s])  # Multiply all elements by scale to complete normalization

    def floor_out(self, x):
        """Set lower threshold 0.01 for sequence x, values <= 0.01 set to 0, reducing spectrum noise"""
        return([j if j > 0.01 else 0 for j in x])
    
    def transform(self, spec):
        """
        Normalize and threshold process IR spectrum vector
        :param spec: list or np.ndarray, spectrum vector
        :return: normalized and denoised spectrum vector (list)
        """
        # Normalize to 0~1
        normed = self.normalize(spec)
        # Threshold processing, values <= 0.01 set to 0
        floored = self.floor_out(normed)
        return floored

    def get_split_masks(self, val_frac, test_frac, split_key, split_seed):
        """
        Split data by split_key ('inchikey_s' or 'scaffold') grouping, output train, validation, test masks.
        
        Parameters:
            val_frac: float, validation set proportion
            test_frac: float, test set proportion
            split_key: str, column name for splitting basis
            split_seed: int, random seed
        
        Returns:
            train_mask, val_mask, test_mask (all bool numpy arrays)
        """
        assert split_key in ["inchikey_s", "scaffold"], split_key

        # 1. Get independent groups
        groups = self.df[split_key].dropna().unique()
        rng = np.random.RandomState(split_seed)
        groups = rng.permutation(groups)  # Shuffle order

        n = len(groups)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_test - n_val

        # 2. Divide groups
        train_groups = set(groups[:n_train])
        val_groups = set(groups[n_train:n_train + n_val])
        test_groups = set(groups[n_train + n_val:])

        # 3. Generate masks
        group_col = self.df[split_key]
        train_mask = group_col.isin(train_groups).values
        val_mask = group_col.isin(val_groups).values
        test_mask = group_col.isin(test_groups).values

        return train_mask, val_mask, test_mask

    def _setup_smiles_vocab(self):
        """Setup SMILES vocabulary"""
        # Default SMILES character set (can be extended as needed)
        self.smiles_chars = [
            'C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se', 'Te',
            'c', 'n', 'o', 's', 'p', 'b',  # Aromatic atoms
            '(', ')', '[', ']', '=', '#', '-', '+', '\\', '/', '@',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            '.', '%', 'H'  # Other symbols
        ]
        
        # Add special tokens
        self.smiles_chars = ['<PAD>', '<UNK>', '<START>', '<END>'] + self.smiles_chars
        
        # Create character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(self.smiles_chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.smiles_chars)}
        self.vocab_size = len(self.smiles_chars)
        
        # Set maximum sequence length (can be obtained from configuration)
        self.max_smiles_len = getattr(self, 'max_smiles_len', 128)

    def smiles_to_tokens(self, smiles_str):
        """Convert SMILES string to token sequence"""
        tokens = []
        i = 0
        while i < len(smiles_str):
            # Handle two-character atoms (like Cl, Br, etc.)
            if i < len(smiles_str) - 1:
                two_char = smiles_str[i:i+2]
                if two_char in self.char_to_idx:
                    tokens.append(self.char_to_idx[two_char])
                    i += 2
                    continue
            
            # Handle single character
            char = smiles_str[i]
            if char in self.char_to_idx:
                tokens.append(self.char_to_idx[char])
            else:
                tokens.append(self.char_to_idx['<UNK>'])  # Unknown character
            i += 1
        
        return tokens

    def pad_smiles_tokens(self, tokens):
        """Pad or truncate SMILES token sequence"""
        if len(tokens) > self.max_smiles_len:
            tokens = tokens[:self.max_smiles_len]
        else:
            # Pad with <PAD>
            pad_length = self.max_smiles_len - len(tokens)
            tokens = tokens + [self.char_to_idx['<PAD>']] * pad_length
        
        return tokens

    def safe_to_array(self, val):
        """Safely convert value to numpy array"""
        # Already ndarray or list, convert directly
        if isinstance(val, np.ndarray):
            return val.astype(np.float32)
        if isinstance(val, list):
            return np.array(val, dtype=np.float32)
        # String form of list
        if isinstance(val, str):
            try:
                arr = ast.literal_eval(val)
                return np.array(arr, dtype=np.float32)
            except Exception as e:
                raise ValueError(f"Cannot parse Spectra string: {val}, error: {e}")
        raise TypeError(f"Unknown Spectra data type: {type(val)}, content: {val}")

    def get_spec_feats(self, spec_entry):
        """
        Extract spectrum features
        
        Parameters:
            spec_entry: Single spectrum entry
        
        Returns:
            spec_feats: Dictionary containing spectrum and metadata
        """
        # Extract spectrum vector (e.g., IR intensity)
        spectra = spec_entry[self.Spectra]
        spectra = self.safe_to_array(spectra)
        # (Assuming one-dimensional vector) shape: [N] -> [1,N], suitable for batch
        spec = torch.tensor(spectra, dtype=torch.float32).unsqueeze(0)
        
        # Get phase information
        state = spec_entry['state']
        state_type_idx = self.state_type_c2i[state]
        state_type_meta = torch.as_tensor(np_one_hot(state_type_idx, num_classes=self.num_state_type), dtype=torch.float32)
        spec_meta = state_type_meta.unsqueeze(0)
        
        spec_feats = {
            "spec": spec,
            "spec_meta": spec_meta,
        }
        return spec_feats

    def get_dataloaders(self, run_d):
        """
        Get data loaders for train, validation, and test sets
        
        Parameters:
            run_d: Run configuration dictionary
        
        Returns:
            dl_dict: Data loader dictionary
            split_id_dict: Split ID dictionary
        """
        val_frac = run_d["val_frac"]
        test_frac = run_d["test_frac"]
        split_key = run_d["split_key"]  # Molecular scaffold splitting method
        split_seed = run_d["split_seed"]
        assert run_d["batch_size"] % run_d["grad_acc_interval"] == 0
        batch_size = run_d["batch_size"] // run_d["grad_acc_interval"]
        num_workers = run_d["num_workers"]
        pin_memory = run_d["pin_memory"] if run_d["device"] != "cpu" else False

        # Generate masks for train, val, test sets and count molecules and samples
        train_mask, val_mask, test_mask = self.get_split_masks(val_frac, test_frac, split_key, split_seed)
        
        # Print counts for each split
        print(f"Training set count: {np.sum(train_mask)}")
        print(f"Validation set count: {np.sum(val_mask)}")
        print(f"Test set count: {np.sum(test_mask)}")

        all_idx = np.arange(len(self))
        train_ss = TrainSubset(self, all_idx[train_mask])
        val_ss = th_data.Subset(self, all_idx[val_mask])
        test_ss = th_data.Subset(self, all_idx[test_mask])

        collate_fn = self.get_collate_fn()
        
        if len(train_ss) > 0:
            train_dl = th_data.DataLoader(
                train_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=True,
                drop_last=True  # Prevent single data batches that mess with batchnorm
            )
            train_dl_2 = th_data.DataLoader(
                train_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
        else:
            train_dl = train_dl_2 = None
            
        if len(val_ss) > 0:
            val_dl = th_data.DataLoader(
                val_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
        else:
            val_dl = None
            
        if len(test_ss) > 0:
            test_dl = th_data.DataLoader(
                test_ss,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=False,
                drop_last=False
            )
        else:
            test_dl = None

        # Setup dl_dict - construct data loader dictionary
        dl_dict = {}
        dl_dict["train"] = train_dl
        dl_dict["primary"] = {
            "train": train_dl_2,
            "val": val_dl,
            "test": test_dl
        }

        # Setup split_id_dict - construct split ID dictionary, generate corresponding sample spectrum ID list for each split
        split_id_dict = {}
        split_id_dict["primary"] = {}
        split_id_dict["primary"]["train"] = self.df.iloc[all_idx[train_mask]]["id"].to_numpy()
        split_id_dict["primary"]["val"] = self.df.iloc[all_idx[val_mask]]["id"].to_numpy()
        split_id_dict["primary"]["test"] = self.df.iloc[all_idx[test_mask]]["id"].to_numpy()
        
        return dl_dict, split_id_dict

    def get_track_dl(self, idx, num_rand_idx=0, topk_idx=None, bottomk_idx=None, other_idx=None, spec_ids=None):
        """
        Get tracking data loaders
        
        Parameters:
            idx: Input sample index list for constructing tracking data loader
            num_rand_idx: Number of samples
            topk_idx: Indices of highest and lowest similarity samples in validation set
            bottomk_idx: Bottom-k indices
            other_idx: Other sample indices to track
            spec_ids: Specific spectrum IDs
        
        Returns:
            track_dl_dict: Tracking data loader dictionary
        """
        track_seed = 520
        track_dl_dict = {}  # Initialize empty dictionary to store different categories of tracking data loaders
        collate_fn = self.get_collate_fn()
        
        if num_rand_idx > 0:  # Construct random sample data loader
            with np_temp_seed(track_seed):
                rand_idx = np.random.choice(idx, size=num_rand_idx, replace=False)  # Randomly select num_rand_idx samples from input indices idx
            rand_dl = th_data.DataLoader(
                th_data.Subset(self, rand_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["rand"] = rand_dl  # Store random sample data loader in tracking dictionary
            
        if not (topk_idx is None):  # Construct Top-K sample loader
            topk_idx = idx[topk_idx]
            topk_dl = th_data.DataLoader(
                th_data.Subset(self, topk_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["topk"] = topk_dl
            
        if not (bottomk_idx is None):  # Construct Bottom-K sample loader
            bottomk_idx = idx[bottomk_idx]
            bottomk_dl = th_data.DataLoader(
                th_data.Subset(self, bottomk_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["bottomk"] = bottomk_dl
            
        if not (other_idx is None):  # Construct other index sample loader
            other_idx = idx[other_idx]
            other_dl = th_data.DataLoader(
                th_data.Subset(self, other_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["other"] = other_dl
            
        if not (spec_ids is None):  # Construct loader based on Spec IDs
            # Preserves order
            spec_idx = []
            for spec_id in spec_ids:
                spec_idx.append(int(self.df[self.df["id"] == spec_id].index[0]))  # Find corresponding sample index from dataset based on input spec_ids
            spec_idx = np.array(spec_idx)
            spec_dl = th_data.DataLoader(
                th_data.Subset(self, spec_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["spec"] = spec_dl
            
        return track_dl_dict

    def get_data_dims(self):
        """
        Get data dimensions
        
        Returns:
            dim_d: Dimension dictionary
        """
        data = self.__getitem__(0)
        dim_d = {}
        
        if self.is_fp_dset:  # If fingerprint dataset (is_fp_dset), extract fingerprint feature dimension fp_dim from data
            fp_dim = data["fp"].shape[1]
        else:
            fp_dim = -1

        # New SMILES dimension handling
        if self.is_smiles_dset:
            smiles_vocab_size = self.vocab_size
            smiles_max_len = self.max_smiles_len
        else:
            smiles_vocab_size = -1
            smiles_max_len = -1
        
        if self.is_graph_dset:
            if self.atom_feature_mode == "pretrain":
                n_dim = -1
            else:
                n_dim = data["graph"].ndata['h'].shape[1]
            if self.bond_feature_mode == "none":
                e_dim = 0
            elif self.bond_feature_mode == "pretrain":
                e_dim = -1
            else:
                e_dim = data["graph"].edata['h'].shape[1]
        else:
            n_dim = e_dim = -1
            
        c_dim = l_dim = -1
        
        if self.spec_meta_global:
            g_dim = data["spec_meta"].shape[1]  # If includes spectrum metadata (spec_meta_global), get meta feature dimension g_dim from data
        else:
            g_dim = 0
            
        o_dim = data["spec"].shape[1]  # Get spectrum feature dimension o_dim from data, i.e., dimension of original spectrum data

        dim_d = {
            "fp_dim": fp_dim,
            "smiles_vocab_size": smiles_vocab_size,  # New
            "smiles_max_len": smiles_max_len,        # New
            "n_dim": n_dim,
            "e_dim": e_dim,
            "c_dim": c_dim,
            "l_dim": l_dim,
            "g_dim": g_dim,
            "o_dim": o_dim
        }
        return dim_d

    def get_collate_fn(self):
        """
        Custom data collation function, responsible for integrating batch data into model input format
        This collation function will be used as DataLoader's collate_fn parameter for data integration in each batch
        """
        def _collate(data_ds):
            # Check for rebatching
            if isinstance(data_ds[0], list):  # Check if first item of current batch is a list. If so, call flatten_lol to flatten nested list into single-layer flat list
                data_ds = flatten_lol(data_ds)
            assert isinstance(data_ds[0], dict)
            
            batch_data_d = {k: [] for k in data_ds[0].keys()}  # Initialize batch dictionary batch_data_d with sample keys as keys and empty lists as values
            
            for data_d in data_ds:  # Traverse each sample, collect data by key item by item, store each key's value in corresponding list in batch dictionary
                for k, v in data_d.items():
                    batch_data_d[k].append(v)
                    
            for k, v in batch_data_d.items():  # Execute merge operation for each key's data based on different data types
                if isinstance(data_ds[0][k], torch.Tensor):
                    batch_data_d[k] = torch.cat(v, dim=0)
                elif isinstance(data_ds[0][k], list):
                    batch_data_d[k] = flatten_lol(v)
                elif isinstance(data_ds[0][k], dgl.DGLGraph):
                    batch_data_d[k] = dgl.batch(v)
                elif k == "hdse" and isinstance(data_ds[0][k], torch_geometric.data.Data):
                    batch_data_d[k] = torch_geometric.data.Batch.from_data_list(v)
                elif k == "pyg_data" and isinstance(data_ds[0][k], torch_geometric.data.Data):
                    # New handling for pyg_data, using PyG's Batch.from_data_list method
                    batch_data_d[k] = torch_geometric.data.Batch.from_data_list(v)
                elif k == "dtnn_data":
                    # Filter out None
                    v_filtered = [x for x in v if x is not None]
                    if len(v_filtered) == 0:
                        raise ValueError("All dtnn_data in batch are None!")
                    atom_types = torch.stack([x.atom_types for x in v_filtered], dim=0)
                    dist_basis = torch.stack([x.dist_basis for x in v_filtered], dim=0)
                    mask = torch.stack([x.mask for x in v_filtered], dim=0)
                    num_atoms = torch.tensor([x.num_atoms for x in v_filtered], dtype=torch.long)
                    batch_data_d[k] = dict(
                        atom_types=atom_types,
                        dist_basis=dist_basis,
                        mask=mask,
                        num_atoms=num_atoms
                    )
                elif k == "mat_data" and isinstance(data_ds[0][k], dict):
                    batch_mat = {kk: [] for kk in data_ds[0][k].keys()}
                    num_atoms_list = [dd['x'].shape[0] for dd in v]
                    max_num_atoms = max(num_atoms_list)
                    
                    for dd in v:
                        current_atoms = dd['x'].shape[0]
                        
                        for kk, vv in dd.items():
                            if isinstance(vv, torch.Tensor):
                                shape = list(vv.shape)
                                
                                # Atom feature handling
                                if len(shape) == 2 and kk == 'x':
                                    feature_dim = shape[1]
                                    # Support 26 and 28 dimensional features
                                    assert feature_dim in [26, 28], f"Atom feature dimension not supported, current is {feature_dim}, should be 26 or 28"
                                    
                                    if shape[0] < max_num_atoms:
                                        pad_size = max_num_atoms - shape[0]
                                        vv = torch.nn.functional.pad(vv, (0, 0, 0, pad_size))
                                
                                # Square matrix handling (adjacency matrix, distance matrix, etc.)
                                elif len(shape) == 2 and shape[0] == shape[1]:
                                    if shape[0] < max_num_atoms:
                                        pad_size = max_num_atoms - shape[0]
                                        if kk in ['dist', 'distance']:
                                            # Special handling for distance matrix
                                            vv_padded = torch.nn.functional.pad(vv, (0, pad_size, 0, pad_size))
                                            if pad_size > 0:
                                                vv_padded[shape[0]:, :] = 999.0
                                                vv_padded[:, shape[0]:] = 999.0
                                            vv = vv_padded
                                        else:
                                            # Other matrices padded with 0
                                            vv = torch.nn.functional.pad(vv, (0, pad_size, 0, pad_size))
                                
                                # 3D tensor handling (edge features, etc.)
                                elif len(shape) == 3 and shape[0] == shape[1]:
                                    if shape[0] < max_num_atoms:
                                        pad_size = max_num_atoms - shape[0]
                                        vv = torch.nn.functional.pad(vv, (0, 0, 0, pad_size, 0, pad_size))
                                
                                # 1D vector handling
                                elif len(shape) == 1:
                                    if shape[0] < max_num_atoms:
                                        pad_size = max_num_atoms - shape[0]
                                        vv = torch.nn.functional.pad(vv, (0, pad_size))
                                        
                            batch_mat[kk].append(vv)
                    
                    # Convert to batch tensors
                    for kk, vv in batch_mat.items():
                        if isinstance(vv[0], torch.Tensor):
                            batch_mat[kk] = torch.stack(vv, dim=0)
                        elif isinstance(vv[0], np.ndarray):
                            batch_mat[kk] = torch.tensor(np.stack(vv, axis=0), dtype=torch.float32)
                        else:
                            batch_mat[kk] = vv
                    
                    batch_data_d[k] = batch_mat
                else:
                    raise ValueError(f"{type(data_ds[0][k])} is not supported")
                    
            return batch_data_d

        return _collate

    def process_entry(self, spec_entry):
        """
        Process single spectrum entry
        Returns a dictionary including spectrum vector spec, metadata spec_meta (already one-hot processed), pyg format data, and other information
        
        Parameters:
            spec_entry: Single spectrum entry
        
        Returns:
            data: Processed data dictionary
        """
        # Pass to get_spec_feats (assuming get_spec_feats internally calls self.transform)
        spec_feats = self.get_spec_feats(spec_entry)  # Returns a dictionary including spectrum vector, metadata, etc.
        data = {**spec_feats}
        
        mol = data_utils.mol_from_smiles(spec_entry[self.smiles], standardize=True)  # Get molecule object from molecule SMILES string
        smile = data_utils.mol_to_smiles(mol)
        data[self.smiles] = [smile]
  
            
        if self.is_hdse_data_dst:
            HiDeeST_data = molspectra_data_utils.HDSE_preprocess(mol, spec_entry["id"])
            data["hdse"] = HiDeeST_data
            
        if self.is_pyg_data_dset:
            pyg_data = pyg_data_utils.pyg_preprocess(mol, spec_entry["id"])
            data["pyg_data"] = pyg_data
            
  
     
        
        return data

    def get_atom_featurizer(self):
        """
        This code defines a method get_atom_featurizer that returns corresponding atom feature extractor (featurizer) based on different atom feature modes (atom_feature_mode).
        This method is mainly used to generate atom features required for molecular graph neural networks (Graph Neural Network, GNN)
        """
        assert self.is_graph_dset
        if self.atom_feature_mode == "canonical":
            return chemutils.CanonicalAtomFeaturizer()
        elif self.atom_feature_mode == "pretrain":
            return chemutils.PretrainAtomFeaturizer()
        elif self.atom_feature_mode == 'light':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                data_utils.atom_type_one_hot
            ])
        elif self.atom_feature_mode == 'full':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                data_utils.atom_type_one_hot,
                data_utils.atom_bond_type_one_hot,
                chemutils.atom_degree_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_explicit_valence_one_hot,
                chemutils.atom_implicit_valence_one_hot,
                chemutils.atom_hybridization_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_formal_charge_one_hot,
                chemutils.atom_num_radical_electrons_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
                chemutils.atom_chiral_tag_one_hot
            ])
        elif self.atom_feature_mode == 'medium':
            atom_featurizer_funs = chemutils.ConcatFeaturizer([
                chemutils.atom_mass,
                data_utils.atom_type_one_hot,
                data_utils.atom_bond_type_one_hot,
                chemutils.atom_total_degree_one_hot,
                chemutils.atom_total_num_H_one_hot,
                chemutils.atom_is_aromatic_one_hot,
                chemutils.atom_is_in_ring_one_hot,
            ])
        else:
            raise ValueError(f"Invalid atom_feature_mode: {self.atom_feature_mode}")

        return chemutils.BaseAtomFeaturizer({"h": atom_featurizer_funs})

    def get_bond_featurizer(self):
        """
        This code defines a method get_bond_featurizer that returns corresponding bond feature extractor (featurizer) based on different bond feature modes (bond_feature_mode).
        This method is mainly used to generate bond features required for molecular graph neural networks (Graph Neural Network, GNN).
        """
        assert self.is_graph_dset
        if self.bond_feature_mode == "canonical":
            return chemutils.CanonicalBondFeaturizer()
        elif self.bond_feature_mode == "pretrain":
            return chemutils.PretrainBondFeaturizer()
        elif self.bond_feature_mode == 'light':
            return chemutils.BaseBondFeaturizer(
                featurizer_funcs={'h': chemutils.ConcatFeaturizer([
                    chemutils.bond_type_one_hot
                ])}, self_loop=self.self_loop
            )
        elif self.bond_feature_mode == 'full':
            return chemutils.CanonicalBondFeaturizer(
                bond_data_field='h', self_loop=self.self_loop
            )
        else:
            assert self.bond_feature_mode == 'none'
            return None

    def batch_from_smiles(self, smiles_list, ref_spec_entry):
        """Create batch from SMILES list"""
        data_list = []
        for smiles in smiles_list:
            mol = data_utils.mol_from_smiles(smiles, standardize=True)
            assert not none_or_nan(mol)
            data = self.process_entry(ref_spec_entry, mol)
            data_list.append(data)
        collate_fn = self.get_collate_fn()
        batch_data = collate_fn(data_list)
        return batch_data

    def load_all(self, keys):
        """Load all data for specified keys"""
        collate_fn = self.get_collate_fn()
        dl = th_data.DataLoader(
            self,
            batch_size=100,
            collate_fn=collate_fn,
            num_workers=min(10, len(os.sched_getaffinity(0))),
            pin_memory=False,
            shuffle=False,
            drop_last=False
        )
        all_ds = []
        for b_idx, b in tqdm(enumerate(dl), total=len(dl), desc="> load_all"):
            b_d = {}
            for k in keys:
                b_d[k] = b[k]
            all_ds.append(b_d)
        all_d = collate_fn(all_ds)
        return all_d


class LMDataset(th_data.Dataset):
    """Library matching dataset class"""
    
    def __init__(self, base_ds, *dset_types, **kwargs):
        """
        Initialize library matching dataset
        
        Parameters:
            base_ds: Base dataset
            *dset_types: Dataset types
            **kwargs: Other configuration parameters
        """
        self.base_ds = base_ds
        self.df = base_ds.df
        self.transform = base_ds.transform
        self.spectrum_normalization = base_ds.spectrum_normalization
        
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __len__(self):
        """Get dataset length"""
        return len(self.df)

    def __getitem__(self, idx):
        """Get single sample"""
        return self.base_ds[idx]

    def get_dataloader(self, run_d, mode, group_id=None):
        """
        Get data loader
        
        Parameters:
            run_d: Run configuration
            mode: Mode ('spec' or 'group')
            group_id: Group ID (optional)
        
        Returns:
            Data loader
        """
        batch_size = run_d.get("lm_batch_size", 32)
        num_workers = run_d.get("num_workers", 0)
        pin_memory = run_d.get("pin_memory", False) if run_d["device"] != "cpu" else False
        
        if mode == "spec":
            # Return all spectrum samples
            indices = np.arange(len(self))
        elif mode == "group":
            if group_id is None:
                # Return all candidate samples
                indices = np.arange(len(self))
            else:
                # Return samples for specific group
                indices = self.df[self.df['group_id'] == group_id].index.to_numpy()
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        subset = th_data.Subset(self, indices)
        collate_fn = self.base_ds.get_collate_fn()
        
        dataloader = th_data.DataLoader(
            subset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False
        )
        
        return dataloader


def get_dset_types(embed_types):
    """
    Determine which types of datasets should be loaded based on model embedding types (embed_types)
    
    Parameters:
        embed_types: List of embedding types
    
    Returns:
        dset_types: List of dataset types
    """
    dset_types = set()
    for embed_type in embed_types:
        if embed_type == "fp":
            dset_types.add("fp")
        elif embed_type in ["gat", "wln", "gin_pt"]:
            dset_types.add("graph")
        elif embed_type in ["hdse"]:
            dset_types.add("hdse")
        elif embed_type in ["pyg_data"]:
            dset_types.add("pyg_data")
        elif embed_type == "dtnn":
            dset_types.add("dtnn")
        elif embed_type in ["mat"]:
            dset_types.add("mat")
        elif embed_type == "smiles":  # New SMILES support
            dset_types.add("smiles")
        else:
            raise ValueError(f"invalid embed_type {embed_type}")
    dset_types = list(dset_types)
    return dset_types


def get_default_ds(data_d_ow=dict(), model_d_ow=dict(), run_d_ow=dict()):
    """
    get_default_ds function loads default configurations (data configuration data_d, model configuration model_d, run configuration run_d),
    and allows users to modify these default settings through override dictionaries data_d_ow, model_d_ow, and run_d_ow
    
    Parameters:
        data_d_ow: Data configuration override dictionary
        model_d_ow: Model configuration override dictionary
        run_d_ow: Run configuration override dictionary
    
    Returns:
        data_d: Data configuration
        model_d: Model configuration
        run_d: Run configuration
    """
    from runner import load_config
    template_fp = "config/template.yml"
    custom_fp = None
    device_id = None
    checkpoint_name = None
    
    # Call load_config function to load default configuration from template file template_fp
    _, _, _, data_d, model_d, run_d = load_config(
        template_fp, 
        custom_fp, 
        device_id,
        checkpoint_name
    )
    
    # Override default configuration
    for k, v in data_d_ow.items():
        if k in data_d:
            data_d[k] = v
    for k, v in model_d_ow.items():
        if k in model_d:
            model_d[k] = v
    for k, v in run_d_ow.items():
        if k in run_d:
            run_d[k] = v
            
    return data_d, model_d, run_d


def get_dataloader(data_d_ow=dict(), model_d_ow=dict(), run_d_ow=dict()):
    """
    Get data loader
    
    Parameters:
        data_d_ow: Data configuration override dictionary
        model_d_ow: Model configuration override dictionary
        run_d_ow: Run configuration override dictionary
    
    Returns:
        ds: Dataset
        dl_dict: Data loader dictionary
        data_d: Data configuration
        model_d: Model configuration
        run_d: Run configuration
    """
    data_d, model_d, run_d = get_default_ds(
        data_d_ow=data_d_ow,
        model_d_ow=model_d_ow,
        run_d_ow=run_d_ow
    )
    dset_types = get_dset_types(model_d["embed_types"])
    ds = BaseDataset(*dset_types, **data_d)
    dl_dict, split_id_dict = ds.get_dataloaders(run_d)
    return ds, dl_dict, data_d, model_d, run_d
