import torch as th
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

from misc_utils import EPS, np_temp_seed, np_one_hot, flatten_lol, none_or_nan
import data_utils as data_utils
import spec_utils as spec_utils
import molspectra_data_utils as molspectra_data_utils

def data_to_device(data_d, device, non_blocking):
    # The main function of this function is to transfer tensors or graph data in a data dictionary to a specified device (such as GPU or CPU) to accelerate computation
    new_data_d = {}
    for k, v in data_d.items():
        if isinstance(v, th.Tensor) or isinstance(v, dgl.DGLGraph) or isinstance(v, torch_geometric.data.Data):
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

    def __getitem__(self, idx):
        return self.dataset.__getitem__(self.indices[idx])


class BaseDataset(th_data.Dataset):

    def __init__(self, *dset_types, **kwargs):

        self.is_hdse_data_dst = "hdse" in dset_types
        assert self.is_hdse_data_dst
        for k, v in kwargs.items(): # Dynamically set the passed keyword arguments as class attributes, allowing flexible configuration of class behavior
            setattr(self, k, v)
        assert os.path.isdir(self.proc_dp), self.proc_dp # Verify that the configured data processing directory exists to ensure the validity of subsequent file operations
        self.spec_df = pd.read_pickle(os.path.join(self.proc_dp, "spec_23_EI_df.pkl"))
        self.mol_df = pd.read_pickle(os.path.join(self.proc_dp, "mol_23_EI_df.pkl"))
        # Filter data based on draw_attn parameter
        if self.draw_attn:

            print(f"[Data Filtering] draw_attn=True, enabling SMILES filtering")
            

            common_smiles_path = ''
            
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
            print(f"  - Number of common SMILES: {len(common_smiles_set)}")
            
            # Determine SMILES column name (usually 'smiles')
            smiles_col = getattr(self, 'smiles', 'smiles')  # If not set, default to 'smiles'
            
            # Check if mol_df has SMILES column
            if smiles_col not in self.mol_df.columns:
                raise ValueError(
                    f"Dataset missing SMILES column '{smiles_col}', "
                    f"current columns: {self.mol_df.columns.tolist()}"
                )
            
            # Count before filtering
            original_count = len(self.mol_df)
            
            # Filter: only keep samples in common SMILES set
            self.mol_df = self.mol_df[
                self.mol_df[smiles_col].isin(common_smiles_set)
            ].reset_index(drop=True)
            
            # Count after filtering
            filtered_count = len(self.mol_df)
            
            # Print filtering information
            print(f"  - Sample count before filtering: {original_count}")
            print(f"  - Sample count after filtering: {filtered_count}")
            print(f"  - Retention ratio: {filtered_count/original_count*100:.2f}%")
            print(f"  - Removed samples: {original_count - filtered_count}")
        else:
            print(f"[Data Filtering] draw_attn=False, skipping SMILES filtering, using all data")
        self._select_spec() # Select specific spectral and molecular data based on data_d configuration parameters, insert grouping information, and collect statistics
        # use mol_id as index for speedy access
        self.mol_df = self.mol_df.set_index("mol_id", drop=False).sort_index().rename_axis(None) # Set mol_df index to mol_id for quick data access by molecule ID. Use drop=False to keep original column


    def _select_spec(self):
        # The masks list stores boolean sequences (Series), each sequence is generated from a DataFrame (spec_df) based on specific filtering conditions. These boolean sequences are used to indicate whether each row in the DataFrame meets specific conditions
        # If a row of data meets the condition, the corresponding position will be True; otherwise False. Finally, AND operation is performed to filter out rows that meet the conditions
        masks = []
        # maximum mz allowed Maximum mass-to-charge ratio (m/z) mask
        mz_mask = self.spec_df["peaks"].apply(lambda peaks: max(peak[0] for peak in peaks) < self.mz_max)
        masks.append(mz_mask)
        # precursor mz Precursor mass-to-charge ratio mask
        prec_mz_mask = ~self.spec_df["prec_mz"].isna()
        masks.append(prec_mz_mask)
        # single molecule Single molecule mask (single_mol_mask): Exclude molecular records whose SMILES symbols contain periods (indicating molecular mixtures), ensuring each record in the dataset corresponds to only one molecule
        multi_mol_ids = self.mol_df[self.mol_df["smiles"].str.contains("\\.")]["mol_id"]
        single_mol_mask = ~self.spec_df["mol_id"].isin(multi_mol_ids)
        masks.append(single_mol_mask)
        # neutral molecule Neutral molecule mask Purpose is to select uncharged molecules
        charges = self.mol_df["mol"].apply(data_utils.mol_to_charge)
        charged_ids = self.mol_df[charges != 0]["mol_id"]
        neutral_mask = ~self.spec_df["mol_id"].isin(charged_ids)
        # print(neutral_mask.sum())
        masks.append(neutral_mask)  
        # put them together
        all_mask = masks[0]
        for mask in masks:
            all_mask = all_mask & mask
        if np.sum(all_mask) == 0:
            raise ValueError("select removed all items")
        self.spec_df = self.spec_df[all_mask].reset_index(drop=True) # Apply final mask all_mask to original spec_df dataframe to keep records that meet conditions, and reset index
        self.spec_df = self.spec_df.drop_duplicates(subset=['mol_id'], keep='first').reset_index(drop=True)
        prec_type_counts = self.spec_df['prec_type'].value_counts()
        print(f"prec_type_counts:{prec_type_counts}")
        # only keep mols with spectra
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.spec_df["mol_id"])]
        self.mol_df = self.mol_df.reset_index(drop=True)
        
        # Only keep molecules containing ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F'] elements
        allowed_elems = set(['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F'])
        def mol_only_allowed_elems(mol):
            # Get symbols of all atoms in molecule
            return all(atom.GetSymbol() in allowed_elems for atom in mol.GetAtoms())
        allowed_mol_mask = self.mol_df["mol"].apply(mol_only_allowed_elems)
        self.mol_df = self.mol_df[allowed_mol_mask].reset_index(drop=True)
        self.spec_df = self.spec_df[self.spec_df["mol_id"].isin(self.mol_df["mol_id"])].reset_index(drop=True)



    def __getitem__(self, idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        # mol_entry = self.mol_df[self.mol_df["mol_id"] == mol_id].iloc[0]
        mol_entry = self.mol_df.loc[mol_id]
        data = self.process_entry(spec_entry, mol_entry["mol"])
        return data

    def __len__(self):

        return self.spec_df.shape[0]

    def bin_func(self, mzs, ints, return_index=False):

        assert self.ints_thresh == 0., self.ints_thresh
        return spec_utils.bin_func(
            mzs,
            ints,
            self.mz_max,
            self.mz_bin_res,
            self.ints_thresh,
            return_index)

    def transform_func(self, spec):


        spec = spec_utils.process_spec(
                th.as_tensor(spec),
                self.transform,
                self.spectrum_normalization)
        spec = spec.numpy()
        return spec

    def get_split_masks(
        self,
        val_frac,
        test_frac,
        split_key,
        split_seed):

        assert split_key in ["inchikey_s", "scaffold"], split_key

        # Get main dataset molecules
        mask = self.spec_df["dset"].isin(self.dset)
        mol_id = self.spec_df[mask]["mol_id"].unique()
        key = set(self.mol_df[self.mol_df["mol_id"].isin(mol_id)][split_key])
        key_list = sorted(list(key))

        # Calculate split quantities
        test_num = round(len(key_list) * test_frac)
        val_num = round(len(key_list) * val_frac)

        # Random split
        with np_temp_seed(split_seed):
            test_key = set(np.random.choice(key_list, size=test_num, replace=False))
            train_val_key = key - test_key
            val_key = set(np.random.choice(sorted(list(train_val_key)), size=val_num, replace=False))
            train_key = train_val_key - val_key

        # Get molecule IDs and masks
        train_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(list(train_key))].unique()
        val_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(list(val_key))].unique()
        test_mol_id = self.mol_df["mol_id"][self.mol_df[split_key].isin(list(test_key))].unique()

        train_mask = self.spec_df["mol_id"].isin(train_mol_id)
        val_mask = self.spec_df["mol_id"].isin(val_mol_id)
        test_mask = self.spec_df["mol_id"].isin(test_mol_id)
        mask = train_mask | val_mask | test_mask
        all_mol_id = pd.Series(list(set(train_mol_id) | set(val_mol_id) | set(test_mol_id)))

        assert (train_mask & val_mask & test_mask).sum() == 0

        print("> primary")
        print("splits: train, val, test, total")
        print(f"spec: {train_mask.sum()}, {val_mask.sum()}, {test_mask.sum()}, {mask.sum()}")
        print(f"mol: {len(train_mol_id)}, {len(val_mol_id)}, {len(test_mol_id)}, {len(all_mol_id)}")

        return train_mask, val_mask, test_mask




    def get_spec_feats(self, spec_entry):

        # convert to a dense vector
        mol_id = th.tensor(spec_entry["mol_id"]).unsqueeze(0)
        spec_id = th.tensor(spec_entry["spec_id"]).unsqueeze(0)
        mzs = [peak[0] for peak in spec_entry["peaks"]]
        ints = [peak[1] for peak in spec_entry["peaks"]]
        prec_mz = spec_entry["prec_mz"]
        prec_mz_bin = self.bin_func([prec_mz], None, return_index=True)[0] # Precursor mass-to-charge ratio bin index, i.e., precursor mass-to-charge ratio mass
        prec_diff = max(mz - prec_mz for mz in mzs) # Calculate maximum difference between all mass-to-charge ratios and precursor mass-to-charge ratio, used to capture maximum dispersion in spectrum
        num_peaks = len(mzs)
        bin_spec = self.transform_func(self.bin_func(mzs, ints)) # Binned spectrum vector
        spec = th.as_tensor(bin_spec, dtype=th.float32).unsqueeze(0)
        # same as prec_mz_bin but tensor Convert precursor mass-to-charge ratio bin index prec_mz_bin to tensor, and use min function to ensure index does not exceed spectrum dimension range
        prec_mz_idx = th.tensor(min(prec_mz_bin, spec.shape[1] - 1)).unsqueeze(0)
        assert prec_mz_idx < spec.shape[1], (prec_mz_bin, prec_mz_idx, spec.shape)
        spec_feats = {
            "spec": spec,
            "prec_mz": [prec_mz],
            "prec_mz_bin": [prec_mz_bin],
            "prec_diff": [prec_diff],
            "num_peaks": [num_peaks],
            "mol_id": mol_id,
            "spec_id": spec_id,
            "prec_mz_idx": prec_mz_idx
        }
        return spec_feats

    def get_dataloaders(self, run_d):
        val_frac = run_d["val_frac"]
        test_frac = run_d["test_frac"]
        split_key = run_d["split_key"] # Molecular scaffold split method
        split_seed = run_d["split_seed"]
        assert run_d["batch_size"] % run_d["grad_acc_interval"] == 0
        batch_size = run_d["batch_size"] // run_d["grad_acc_interval"]
        num_workers = run_d["num_workers"]
        pin_memory = run_d["pin_memory"] if run_d["device"] != "cpu" else False

        # Generate masks for training set, validation set, test set and secondary dataset, while counting molecules and samples in each dataset, and handling overlap between main and secondary datasets
        train_mask, val_mask, test_mask = self.get_split_masks(val_frac, test_frac, split_key, split_seed)
        
        # Print quantities for each split
        print(f"Training set size: {np.sum(train_mask)}")
        print(f"Validation set size: {np.sum(val_mask)}")
        print(f"Test set size: {np.sum(test_mask)}")

        all_idx = np.arange(len(self))
        # th_data.RandomSampler()
        train_ss = TrainSubset(self, all_idx[train_mask])
        # th_data.RandomSampler(th_data.Subset(self,all_idx[val_mask]))
        val_ss = th_data.Subset(self, all_idx[val_mask])
        # th_data.RandomSampler(th_data.Subset(self,all_idx[test_mask]))
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
                drop_last=True  # this is to prevent single data batches that mess with batchnorm
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
        sec_dls = []

        # set up dl_dict Build dataloader dictionary dl_dict
        dl_dict = {}
        dl_dict["train"] = train_dl
        dl_dict["primary"] = {
            "train": train_dl_2,
            "val": val_dl,
            "test": test_dl
        }

        # set up split_id_dict Build split ID dictionary split_id_dict, generate corresponding sample spectrum ID list for each split
        split_id_dict = {}
        split_id_dict["primary"] = {}
        split_id_dict["primary"]["train"] = self.spec_df.iloc[all_idx[train_mask]]["spec_id"].to_numpy()
        split_id_dict["primary"]["val"] = self.spec_df.iloc[all_idx[val_mask]]["spec_id"].to_numpy()
        split_id_dict["primary"]["test"] = self.spec_df.iloc[all_idx[test_mask]]["spec_id"].to_numpy()
        return dl_dict, split_id_dict

    def get_track_dl(
            self,
            idx, # Input sample index list, used to build tracking dataloader
            num_rand_idx=0, # Number of samples
            topk_idx=None, # Indices of samples with highest and lowest similarity in validation set
            bottomk_idx=None,
            other_idx=None, # Other sample indices to track
            spec_ids=None):

        track_seed = 5585
        track_dl_dict = {} # Initialize empty dictionary to store tracking dataloaders of different categories
        collate_fn = self.get_collate_fn()
        if num_rand_idx > 0: # Build random sample dataloader
            with np_temp_seed(track_seed):
                rand_idx = np.random.choice(idx, size=num_rand_idx, replace=False) # Randomly select num_rand_idx samples from input indices idx
            rand_dl = th_data.DataLoader(
                th_data.Subset(self, rand_idx),
                batch_size=1,
                collate_fn=collate_fn,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
                drop_last=False
            )
            track_dl_dict["rand"] = rand_dl # Store random sample dataloader in tracking dictionary
        if not (topk_idx is None): # Build Top-K sample loader
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
        if not (bottomk_idx is None): # Build Bottom-K sample loader
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
        if not (other_idx is None): # Build other index sample loader
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
        if not (spec_ids is None): # Build loader based on Spec IDs
            # preserves order
            spec_idx = []
            for spec_id in spec_ids:
                spec_idx.append(
                    int(self.spec_df[self.spec_df["spec_id"] == spec_id].index[0])) # Find corresponding sample indices from dataset based on input spec_ids
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

        data = self.__getitem__(0)
        dim_d = {}
        fp_dim = -1
        n_dim = e_dim = -1
        c_dim = l_dim = -1
        g_dim = 0  # -1
        o_dim = data["spec"].shape[1] # Get spectrum feature dimension o_dim from data, i.e., original spectrum data dimension

        dim_d = {
            "fp_dim": fp_dim,
            "n_dim": n_dim,
            "e_dim": e_dim,
            "c_dim": c_dim,
            "l_dim": l_dim,
            "g_dim": g_dim,
            "o_dim": o_dim
        }
        return dim_d

    def get_collate_fn(self):
        # Custom data alignment function, responsible for integrating batch data into model input format. This alignment function will be used as collate_fn parameter of DataLoader for data integration of each batch
        def _collate(data_ds):
            # check for rebatching
            if isinstance(data_ds[0], list): # Check if first item of current batch is a list. If yes, call flatten_lol to flatten nested list into single-layer flat list
                data_ds = flatten_lol(data_ds)
            assert isinstance(data_ds[0], dict)
            batch_data_d = {k: [] for k in data_ds[0].keys()} # Initialize batch dictionary batch_data_d with sample keys as keys and empty lists as values
            for data_d in data_ds: # Traverse each sample, collect data item by item by key, store each key's value in corresponding list of batch dictionary
                for k, v in data_d.items():
                    batch_data_d[k].append(v)
            for k, v in batch_data_d.items(): # Perform merge operations on each key's data based on different data types
                if isinstance(data_ds[0][k], th.Tensor):
                    batch_data_d[k] = th.cat(v, dim=0)
                elif isinstance(data_ds[0][k], list):
                    batch_data_d[k] = flatten_lol(v)
                elif k == "hdse" and isinstance(data_ds[0][k], torch_geometric.data.Data):
                    batch_data_d[k] = torch_geometric.data.Batch.from_data_list(v)
                else:
                    raise ValueError(f"{type(data_ds[0][k])} is not supported")
            return batch_data_d

        return _collate

    def process_entry(self, spec_entry, mol):
        """Returns a dictionary including spectrum vector spec, metadata spec_meta (already one-hot processed), data in pyg format and other information"""
        # initialize data with shared attributes
        spec_feats = self.get_spec_feats(spec_entry) # Returns a dictionary including spectrum vector, metadata, etc.
        data = {**spec_feats}
        smile = data_utils.mol_to_smiles(mol)
        data["smiles"] = [smile] # Continue adding data to dictionary
        data["formula"] = [data_utils.mol_to_formula(mol)]
        # add dset_type specific attributes Add attributes based on data type
        if self.is_hdse_data_dst:
            hdse_data = molspectra_data_utils.Mol_preprocess(mol,spec_entry["spec_id"])
            data["hdse"] = hdse_data
        if self.lm_fp:
            fp = data_utils.make_maccs_fingerprint(mol)
            fp = th.as_tensor(fp, dtype=th.float32).unsqueeze(0)
            data["lm_fp"] = fp
        return data

    def batch_from_smiles(self, smiles_list, ref_spec_entry):

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


def get_dset_types(embed_types):
    # Determine which types of datasets should be loaded based on model embedding types (embed_types)
    dset_types = set()
    for embed_type in embed_types:
        if embed_type in ["hdse"]:
            dset_types.add("hdse")
        else:
            raise ValueError(f"invalid embed_type {embed_type}")
    dset_types = list(dset_types)
    return dset_types


def get_default_ds(data_d_ow=dict(),model_d_ow=dict(),run_d_ow=dict()):
    """
    get_default_ds function is used to load default configurations (data configuration data_d, model configuration model_d, run configuration run_d),
    and allows users to modify these default settings through override dictionaries data_d_ow, model_d_ow and run_d_ow
    """
    from massformer.runner import load_config
    template_fp = "config/template.yml"
    custom_fp = None
    device_id = None
    checkpoint_name = None
    _, _, _, data_d, model_d, run_d = load_config( # Call load_config function to load default configurations from template file template_fp
        template_fp, 
        custom_fp, 
        device_id,
        checkpoint_name
    )
    # Override default configurations
    for k, v in data_d_ow.items():
        if k in data_d:
            data_d[k] = v
    for k,v in model_d_ow.items():
        if k in model_d:
            model_d[k] = v
    for k,v in run_d_ow.items():
        if k in run_d:
            run_d[k] = v
    return data_d, model_d, run_d


def get_dataloader(data_d_ow=dict(),model_d_ow=dict(),run_d_ow=dict()):

    data_d, model_d, run_d = get_default_ds(
        data_d_ow=data_d_ow,
        model_d_ow=model_d_ow,
        run_d_ow=run_d_ow
    )
    dset_types = get_dset_types(model_d["embed_types"])
    ds = BaseDataset(*dset_types, **data_d)
    ds.compute_lda(run_d)
    dl_dict, split_id_dict = ds.get_dataloaders(run_d)
    return ds, dl_dict, data_d, model_d, run_d


class LMDataset(BaseDataset):

    def __init__(self, ds, *dset_types, **kwargs):

        self.is_hdse_data_dst = "hdse" in dset_types
        assert self.is_hdse_data_dst
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.spec_df = pd.read_pickle(os.path.join(self.proc_dp,"spec_df.pkl"))
        self.mol_df = pd.read_pickle(os.path.join(self.proc_dp,"mol_df.pkl"))
        self.cand_df = pd.read_pickle(os.path.join(self.proc_dp,"cand_df.pkl"))
        # select the spectra
        self.spec_df = self.spec_df[self.spec_df["ion_mode"] == "EI"].reset_index(drop=True)
        self.spec_df = self.spec_df[self.spec_df["prec_type"] == "EI"].reset_index(drop=True)
        # extract the query and candidate mol_ids
        query_mol_id = set(self.spec_df["mol_id"]) & set(self.mol_df["mol_id"]) & set(self.cand_df["query_mol_id"])
        cand_mol_id = set(self.cand_df[self.cand_df["query_mol_id"].isin(query_mol_id)]["candidate_mol_id"])
        # drop candidates with duplicate inchikey_s (i.e. stereoisomers)
        # in principle, this might drop the matching mol_id for a challenge spectrum!
        # so, we need to do it this way
        # Filter records in mol_df related to query molecules (query_mol_id) or candidate molecules (cand_mol_id)
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(query_mol_id | cand_mol_id)]
        # Extract records from mol_df that match query molecules (query_mol_id)
        match_mol = self.mol_df[self.mol_df["mol_id"].isin(query_mol_id)]
        # Remove records from mol_df that have duplicate inchikey_s with match_mol
        self.mol_df = self.mol_df[~self.mol_df["inchikey_s"].isin(match_mol["inchikey_s"])]
        # Further deduplicate mol_df, keeping only unique records for each inchikey_s
        self.mol_df = self.mol_df.drop_duplicates(subset=["inchikey_s"])
        self.mol_df = pd.concat([self.mol_df, match_mol])
        # Update query molecule IDs, ensuring only molecules existing in mol_df are kept. Update candidate molecule IDs, ensuring only molecules existing in mol_df are kept
        query_mol_id = query_mol_id & set(self.mol_df["mol_id"])
        cand_mol_id = cand_mol_id & set(self.mol_df["mol_id"])
        # remove dropped candidates
        # Remove records from spec_df whose mol_id is not in updated query_mol_id
        self.spec_df = self.spec_df[self.spec_df["mol_id"].isin(query_mol_id)].reset_index(drop=True)
        # Remove records from mol_df whose mol_id is not in updated query_mol_id or cand_mol_id
        self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(query_mol_id | cand_mol_id)].reset_index(drop=True)
        # Remove records from cand_df whose query_mol_id or candidate_mol_id is not in updated query_mol_id and cand_mol_id
        self.cand_df = self.cand_df[self.cand_df["query_mol_id"].isin(query_mol_id) & self.cand_df["candidate_mol_id"].isin(cand_mol_id)].reset_index(drop=True)
        # sanity checks
        # Ensure mol column in molecular data (mol_df) has no missing values
        assert not self.mol_df["mol"].isna().any()
        df = self.spec_df[["mol_id", "spec_id"]].rename(columns={"mol_id": "query_mol_id", "spec_id": "query_spec_id"})
        self.mol_spec_df = self.cand_df.merge(df, on="query_mol_id", how="inner")
        del df
        # remove spec with no candidates (this is possible due to filtering) Remove spectra with no candidate molecules
        self.spec_df = self.spec_df[self.spec_df["spec_id"].isin(self.mol_spec_df["query_spec_id"])]
        # set indices for quicker access Set index of spectral data table self.spec_df to spec_id column, set index of molecular data table self.mol_df to mol_id column
        self.spec_df = self.spec_df.set_index("spec_id", drop=False).sort_index().rename_axis(None)
        self.mol_df = self.mol_df.set_index("mol_id", drop=False).sort_index().rename_axis(None)

    def _copy_from_ds(self, ds):

        assert isinstance(ds, BaseDataset)
        attrs = [
            "inst_type_c2i",
            "inst_type_i2c",
            "prec_type_c2i",
            "prec_type_i2c",
            "frag_mode_c2i",
            "frag_mode_i2c",
            "num_inst_type",
            "num_prec_type",
            "num_frag_mode",
            "can_seeds",
            "max_ce", "mean_ce", "std_ce"
        ]
        for attr in attrs:
            if hasattr(ds, attr):
                setattr(self, attr, getattr(ds, attr))

    def get_dataloader(self, run_d, mode, group_id=None):

        if mode == "spec":
            batch_size = self.spec_df.shape[0] # Batch size equals total number of rows in spectral data
            num_workers = 0
            pin_memory = False
            ds = LMSpecDataset(self)
        else:
            assert mode == "group"
            batch_size = run_d["fp_batch_size"]
            if batch_size == -1:
                assert run_d["batch_size"] % run_d["grad_acc_interval"] == 0
                batch_size = run_d["batch_size"] // run_d["grad_acc_interval"]
            num_workers = run_d["fp_num_workers"]
            if num_workers == -1:
                num_workers = run_d["num_workers"]
            pin_memory = run_d["pin_memory"] if run_d["device"] != "cpu" else False
            ds = LMGroupDataset(self, group_id=group_id)
        dl = th_data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
            drop_last=False,
            collate_fn=self.get_collate_fn()
        )
        return dl

    def __getitem__(self, idx):

        raise NotImplementedError

    def __len__(self):

        raise NotImplementedError


class LMSpecDataset(th_data.Dataset):

    def __init__(self, lm_d):

        self.spec_df = lm_d.spec_df
        self.mol_df = lm_d.mol_df
        self.process_entry = lm_d.process_entry

    def __getitem__(self, idx):

        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        if mol_id in self.mol_df["mol_id"]:
            mol_entry = self.mol_df.loc[mol_id]
        else:
            raise ValueError(f"mol_id {mol_id} not found in mol_df")
            # just choose an artbitrary mol_entry
            mol_entry = self.mol_df.iloc[0]
        data = self.process_entry(spec_entry, mol_entry["mol"])
        return data

    def __len__(self):

        return self.spec_df.shape[0]


class LMGroupDataset(th_data.Dataset):

    def __init__(self, lm_d, group_id=None):

        self.spec_df = lm_d.spec_df
        self.mol_df = lm_d.mol_df
        self.mol_spec_df = lm_d.mol_spec_df
        self.process_entry = lm_d.process_entry # Can be understood as self.cand_df with added spec_id
        if not (group_id is None):
            self.spec_df = self.spec_df[self.spec_df["group_id"] == group_id]
            self.mol_spec_df = self.mol_spec_df[self.mol_spec_df["query_spec_id"].isin(self.spec_df["spec_id"])]
            self.mol_df = self.mol_df[self.mol_df["mol_id"].isin(self.mol_spec_df["candidate_mol_id"])]

    def __getitem__(self, idx):

        mol_spec_entry = self.mol_spec_df.iloc[idx]
        mol_id = mol_spec_entry["candidate_mol_id"]
        spec_id = mol_spec_entry["query_spec_id"]
        # mol_id = mol_spec_entry["candidate_mol_id"].item()
        # spec_id = mol_spec_entry["query_spec_id"].item()
        mol_entry = self.mol_df.loc[mol_id]
        spec_entry = self.spec_df.loc[spec_id].copy()
        # don't use .loc[:,"mol_id"] since it's a single row
        spec_entry.loc["mol_id"] = mol_id
        data = self.process_entry(spec_entry, mol_entry["mol"])
        return data

    def __len__(self):

        return self.mol_spec_df.shape[0]
