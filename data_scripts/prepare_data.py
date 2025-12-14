import numpy as np
import os
import pandas as pd
import ast
import json
from pprint import pprint, pformat
import joblib
from tqdm import tqdm
import argparse

import data_utils as data_utils
from data_utils import par_apply_series, par_apply_df_rows, seq_apply_series, seq_apply_df_rows, check_mol_props
from misc_utils import list_str2float, booltype, tqdm_joblib


def load_df(df_dp, dset_names, num_entries):
    """
    The function of this code is to load multiple dataset files from a specified directory, merge them into a DataFrame and return it
    df_dp: String, directory path where dataset files are located.
    dset_names: List or string list, representing dataset names to load.
    num_entries: Integer, controls the number of entries to load from each dataset (-1 means load all)
    """
    dset_names = ["NIST_EI_23"]
    dfs = []
    for dset_name in dset_names:
        dset_fp = os.path.join(df_dp, f"{dset_name}_df.json") # Assuming df_dp="data/df", dset_name="nist", generates path as data/df/nist_df.json
        with open(dset_fp, "r") as f:
            dset_df = json.load(f)
        dset_df = pd.DataFrame(dset_df)
        # dset_df = pd.read_json(open(dset_fp, "r", encoding="utf8"))
        dset_df.loc[:, "dset"] = dset_name # Add a column dset to identify the current dataset name for subsequent analysis
        print("Column names:", list(dset_df.columns))
        dfs.append(dset_df)
    if num_entries > 0: # If num_entries > 0, randomly sample num_entries data from each dataset
        dfs = [
            df.sample(
                n=num_entries,
                replace=False, # Sampling without replacement, set random seed to ensure reproducible sampling results
                random_state=420) for df in dfs]
    if len(dfs) > 1: # If multiple datasets are loaded, merge them into one DataFrame
        all_df = pd.concat(dfs, ignore_index=True)
    else:
        all_df = dfs[0]
    all_df = all_df.reset_index(drop=True) # Reset the DataFrame index to start from 0 without keeping the old index

    return all_df



def preprocess_spec(spec_df):

    # convert smiles to mol and back (for standardization/stereochemistry) Parse and validate SMILES, generate unique mol_id and clean invalid molecular data
    spec_df.loc[:, "mol"] = par_apply_series(spec_df["smiles"], data_utils.mol_from_smiles)  # Convert each SMILES string in spec_df["smiles"] to a "standardized" molecule object
    spec_df.loc[:, "smiles"] = par_apply_series(spec_df["mol"], data_utils.mol_to_smiles) # Convert molecule object to standardized SMILES representation
    spec_df = spec_df.dropna(subset=["mol", "smiles"]) # Delete rows with NaN values in mol or smiles columns, if any row has NaN in either column, that row will be deleted
    # check atom types, number of bonds, neutral charge, call check_mol_props function to perform a series of validations on molecular properties
    spec_df = check_mol_props(spec_df)  
    # enumerate smiles to create molecule ids, use set to deduplicate spec_df["smiles"], generate unique SMILES set
    smiles_set = set(spec_df["smiles"])  
    print("> num_smiles", len(smiles_set))
    smiles_to_mid = {smiles: i for i, smiles in enumerate(sorted(smiles_set))} # Use enumerate to number the sorted smiles_set one by one, assigning a unique mol_id to each unique SMILES
    spec_df.loc[:, "mol_id"] = spec_df["smiles"].replace(smiles_to_mid) # Replace each SMILES in spec_df["smiles"] with its corresponding mol_id

    # extract peak info (still represented as str) Each peaks is a list containing tuples (mz, ints)
    spec_df.loc[:, "peaks"] = par_apply_series(spec_df["peaks"], data_utils.parse_peaks_str)

    # get mz resolution
    spec_df.loc[:, "res"] = par_apply_series(spec_df["peaks"], data_utils.get_res)
    # standardize the instrument type and frag_mode
    inst_type, frag_mode = seq_apply_df_rows(spec_df, data_utils.parse_inst_info)
    spec_df.loc[:, "inst_type"] = inst_type
    spec_df.loc[:, "frag_mode"] = frag_mode
    # standardise prec_type
    spec_df.loc[:, "prec_type"] = par_apply_series(spec_df["spectra_type"], data_utils.parse_prec_type_str)
    # convert prec_mz Convert data in the prec_mz column of spec_df to numeric type, errors="coerce" means: if a value cannot be converted to numeric, replace it with NaN (missing value)
    spec_df.loc[:, "prec_mz"] = pd.to_numeric(spec_df["exact_mass"], errors="coerce")
    spec_df = spec_df.astype({"prec_mz": float})
    # convert ion_mode
    spec_df.loc[:, "ion_mode"] = par_apply_series(spec_df["spectra_type"], data_utils.parse_ion_mode_str)
    # convert peaks to float
    spec_df.loc[:, "peaks"] = par_apply_series(spec_df["peaks"], data_utils.convert_peaks_to_float)
     # ---- New: Convert spec_id to int type ----
    spec_df.loc[:, "spec_id"] = pd.to_numeric(spec_df["spec_id"], errors="coerce").astype('Int64')

    # remove columns from spec_df Keep specified columns in spec_df, other columns will be discarded
    spec_df = spec_df[["spec_id", "mol_id", "prec_type", "inst_type", "frag_mode", "spectra_type", "ion_mode", "dset", "res","prec_mz", "peaks"]]

   

    # get mol df The function of this code is to build the molecular data table mol_df, by parsing molecules represented by SMILES and extracting relevant chemical and molecular property information
    # Build a DataFrame containing: smiles: unique SMILES representation. mol_id: unique ID assigned to each SMILES (usually an integer starting from 0)
    mol_df = pd.DataFrame(zip(sorted(smiles_set), list(range(len(smiles_set)))), columns=["smiles", "mol_id"])
    mol_df.loc[:, "mol"] = par_apply_series(mol_df["smiles"], data_utils.mol_from_smiles)
    mol_df.loc[:, "inchikey_s"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchikey_s)
    mol_df.loc[:, "scaffold"] = par_apply_series(mol_df["mol"], data_utils.get_murcko_scaffold)
    mol_df.loc[:, "formula"] = par_apply_series(mol_df["mol"], data_utils.mol_to_formula)
    mol_df.loc[:, "inchi"] = par_apply_series(mol_df["mol"], data_utils.mol_to_inchi)
    mol_df.loc[:, "mw"] = par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=False))
    mol_df.loc[:, "exact_mw"] = par_apply_series(mol_df["mol"], lambda mol: data_utils.mol_to_mol_weight(mol, exact=True))

    # remove invalid mols and corresponding spectra 
    # The main purpose of this code is to remove invalid molecules and their corresponding spectral data from the molecular data table (mol_df) and spectral data table (spec_df), while rearranging the data indices
    all_mol_id = set(mol_df["mol_id"])
    mol_df = mol_df.dropna(subset=["mol"], axis=0) # Remove rows with NaN values in the mol column, these rows represent invalid molecule objects (e.g., parsing failed)
    bad_mol_id = all_mol_id - set(mol_df["mol_id"]) # Determine mol_id of invalid molecules
    print("> bad_mol_id", len(bad_mol_id))
    spec_df = spec_df[~spec_df["mol_id"].isin(bad_mol_id)] # Remove invalid molecules from spectral data

    # reset indices
    spec_df = spec_df.reset_index(drop=True) # Call reset_index method to reset the index, ensuring the index starts from 0 with consecutive numbering (the original data order will not be changed)
    mol_df = mol_df.reset_index(drop=True)

    return spec_df, mol_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--df_dp", type=str, default="data/processed")
    parser.add_argument("--dset_names", type=str, default="NIST_EI_23")
    parser.add_argument("--proc_dp", type=str, default="data/proc")
    parser.add_argument("--num_entries", type=int, default=-1)
    flags = parser.parse_args()

    os.makedirs(flags.proc_dp, exist_ok=True)
    data_dp = flags.proc_dp
    # Define output file paths
    spec_df_fp = os.path.join(data_dp, "spec_23_EI_df.pkl")
    mol_df_fp = os.path.join(data_dp, "mol_23_EI_df.pkl")

    print("> creating new spec_df, mol_df")
    assert os.path.isdir(flags.df_dp), flags.df_dp # Verify that the input directory exists
    # Call load_df function to load datasets from the specified directory flags.df_dp, return merged DataFrame containing all loaded datasets. And add a column dset to identify the current dataset name
    all_df = load_df(flags.df_dp, flags.dset_names, flags.num_entries) 

    spec_df, mol_df = preprocess_spec(all_df)

    # save everything to file
    spec_df.to_pickle(spec_df_fp)
    mol_df.to_pickle(mol_df_fp)

    print(spec_df.shape)
    print(spec_df.isna().sum())
    print(mol_df.shape)
    print(mol_df.isna().sum())

    # export smiles (.txt, cfm) and inchi (.tsv, classyfire)
    smiles_df = mol_df[["mol_id", "smiles"]]
    smiles_df.to_csv(os.path.join(data_dp,"all_smiles.txt"),sep=" ",header=False,index=False) # Export SMILES
    inchi_df = mol_df[["mol_id", "inchi"]] # Export InChI
    inchi_df.to_csv(os.path.join(data_dp,"all_inchi.tsv"),sep="\t",header=False,index=False)
