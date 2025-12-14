from collections import Counter
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm
import pickle
import argparse
import ast
import glob
import importlib
import re
from rdkit import Chem
import sys
import requests
import joblib
import json

from tqdm import tqdm
# This is a dictionary used to map field names from the original data (e.g., "Precursor_type") to shorter or more standardized field names (e.g., "prec_type")
key_dict = {
    "SpectraType": "spectra_type",   # Spectra type
    "DB#": "spec_id",
    "Num Peaks": "num_peaks",
    "MW": "mw",
    "ExactMass": "exact_mass",
    "CAS#": "cas_num",
    "NIST#": "nist_num",
    "Name": "name",
    "MS": "peaks",
    "SMILES": "smiles",
    "InChIKey": "inchi_key"
}


# Define a function to query SMILES via InChIKey
def inchi_key_to_smiles(inchi_key, local_mapping_file="/home/lwh/projects/lzq_3/NIST23/inchikey_to_smiles_2.json", delay=1, missing_keys=None):
    """
    Query SMILES corresponding to InChIKey, prioritize querying from local file, if not found then call PubChem API.
    :param inchi_key: str, InChIKey
    :param local_mapping_file: str, local JSON file path storing InChIKey to SMILES mapping
    :param delay: float, delay time (seconds) between each API request, default value is 0.2 seconds (i.e., maximum 5 requests per second)
    :param missing_keys: set, used to collect InChIKeys not found locally
    :return: str or None, returns SMILES or None (if query fails)
    """
    # Prioritize querying from local file
    if os.path.exists(local_mapping_file):
        try:
            with open(local_mapping_file, "r") as f:
                local_mapping = json.load(f)
            if inchi_key in local_mapping:
                return local_mapping[inchi_key]  # If corresponding SMILES is found, return directly
        except Exception as e:
            print(f"Error reading local mapping file: {e}")

    # Call PubChem API for query
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi_key}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=60)  # Set timeout to 60 seconds
        time.sleep(delay)  # Control query frequency, force delay between each request
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
        else:
            # Print detailed information for non-200 status codes
            print(f"HTTP Error {response.status_code} for InChIKey {inchi_key}: {response.text}")
            return None  # Query failed, return None
    except requests.exceptions.RequestException as e:
        # Print detailed information for network request exceptions
        print(f"RequestException for InChIKey {inchi_key}: {e}")
        return None



def inchi_to_smiles(inchi_keys):
    """
    Sequentially process InChIKey to SMILES conversion and count failures
    :param inchi_keys: list, list containing InChIKeys
    :return: tuple, (smiles_list, failed_count)
    """
    failed_count = 0  # Initialize failure counter
    smiles_list = []  # Initialize SMILES list

    # Iterate through each InChIKey
    for inchi_key in tqdm(inchi_keys, desc="Processing InChIKey to SMILES"):
        try:
            smiles = inchi_key_to_smiles(inchi_key)
            if smiles is None:
                failed_count += 1  # Count when query fails
            # if failed_count % 10000  == 0:
            #     print(f"Failed to convert {failed_count} InChIKeys to SMILES")
            smiles_list.append(smiles)
        except Exception as e:
            failed_count += 1  # Count on exception
            smiles_list.append(None)  # Ensure list length matches input

    return smiles_list, failed_count




"""
Convert data from MSMS database format to pandas dataframe with JSON
No type conversions or filtering: all of that is done downstream
"""

def validate_peaks(peaks_str, line_number):
    """
    Check if peaks data conforms to 'm/z intensity;' pair format.
    :param peaks_str: String containing mass spectrum peak data, each peak ending with semicolon.
    :param line_number: Line number of current entry in source file (for debugging).
    :return: Returns True if all peak data is valid; otherwise returns False and prints error line.
    """
    if not peaks_str.strip():
        print(f"Empty peaks data at line {line_number}")
        return False

    # Concatenate all lines into one string, then split by semicolon
    peaks = peaks_str.replace('\n', ' ').split(';')
    for peak in peaks:
        peak = peak.strip()
        if not peak:
            continue
        parts = peak.split()
        if len(parts) != 2:
            print(f"Invalid peak format at line {line_number}: {peak}")
            return False
        try:
            m_z, intensity = float(parts[0]), float(parts[1])
            if m_z < 0 or intensity < 0:
                print(f"Negative value in peaks at line {line_number}: {peak}")
                return False
        except ValueError:
            print(f"Non-numeric peak data at line {line_number}: {peak}")
            return False
    return True





def preproc_msp(msp_dirs, keys, num_entries):
    """ 
    Modified function to read MSP file content from multiple folders and parse into a DataFrame.

    Parameters:
    - msp_dirs: List of folder paths containing MSP files.
    - keys: List of field names to extract, usually pass in key_dict.keys().
    - num_entries: Maximum number of entries to read. If value is -1, read all entries.
    """
    # Get all -smiles.MSP file paths
    msp_files = []
    for msp_dir in msp_dirs:
        for root, _, files in os.walk(msp_dir):
            for file in files:
                # Only process files ending with -smiles.MSP
                if file.endswith("-smiles.MSP"):
                    msp_files.append(os.path.join(root, file))
    
    print(f"Found {len(msp_files)} MSP files to process.")

    # Initialize
    raw_data_list = []  # Store entries from all files
    sum_invalid = 0  # Count invalid data
    total_entries = 0  # Count total entries

    # Iterate through each file
    for msp_fp in tqdm(msp_files, desc="Processing MSP files"):
        with open(msp_fp) as f:
            raw_data_lines = f.readlines()
        # Subsequent processing code...
        raw_data_item = {key: None for key in keys}  # Single entry
        read_ms = False  # Flag indicating whether currently reading "MS" (mass spectrum data) field
        line_number = 0  # Track line number in file

        for raw_l in raw_data_lines:
            # Check if maximum entry limit is reached
            if num_entries > -1 and total_entries >= num_entries:
                break

            raw_l = raw_l.replace('\n', '')  # Remove newline at end of line
            line_number += 1  # Increment line number

            if raw_l == '':
                # Check if MS field is valid
                if raw_data_item['MS'] and not validate_peaks(raw_data_item['MS'], line_number):
                    sum_invalid += 1
                    print(f"Invalid peaks data at line {line_number} in file {msp_fp}")
                    raw_data_item['MS'] = None  # Mark invalid data
                    raw_data_item = {key: None for key in keys} # Directly delete this spectrum data

                # Add entry to list and reset
                # if raw_data_item["InChIKey"] is None:
                #     print(f"Missing InChIKey at line {line_number} in file {msp_fp}")
                raw_data_list.append(raw_data_item.copy())
                raw_data_item = {key: None for key in keys}
                read_ms = False
                total_entries += 1  # Increment total entry count
            elif read_ms:
                raw_data_item['MS'] = raw_data_item['MS'] + raw_l + '\n'
            else:

                raw_l_split = raw_l.split(': ')
                assert len(raw_l_split) >= 2
                key = raw_l_split[0]
                if key == "Num peaks" or key == "Num Peaks":
                    assert len(raw_l_split) == 2, raw_l_split
                    value = raw_l_split[1]
                    raw_data_item['Num Peaks'] = int(value)
                    raw_data_item['MS'] = ''  # Initialize "MS" field as empty string
                    read_ms = True
                elif key == "CAS#":
                    # Process CAS and NIST fields
                    cas_nist_data = raw_l_split[1].split(";")
                    cas_value = cas_nist_data[0].strip()
                    nist_value = raw_l_split[2].strip()  
                    raw_data_item["CAS#"] = cas_value
                    raw_data_item["NIST#"] = nist_value
                elif key in keys:
                    value = raw_l_split[1]
                    raw_data_item[key] = value

    # Convert to DataFrame
    msp_df = pd.DataFrame(raw_data_list)
    msp_df = msp_df.dropna(axis=0, how="all")  # Delete rows with all null values
    # Set spectra type to EI
    msp_df["SpectraType"] = "EI"
    print(f"A total of {sum_invalid} invalid spectrum data were filtered")
    print(f"Total of {total_entries} valid entries processed")
    return msp_df



def merge_and_check(msp_df, mol_df, rename_dict):
    """
    Clean and merge msp_df data, match with mol_df, and verify data integrity.
    
    :param msp_df: DataFrame, containing original msp data.
    :param mol_df: DataFrame, optional, molecular data. If empty, process msp_df separately.
    :param rename_dict: dict, column name mapping dictionary for renaming msp_df columns.
    :return: DataFrame, processed spec_df.
    """
    # Delete unnecessary columns, only keep fields in rename_dict
    msp_bad_cols = set(msp_df.columns) - set(rename_dict.keys())
    msp_df = msp_df.drop(columns=msp_bad_cols)
    
    # Rename columns to standardized names
    msp_df = msp_df.rename(columns=rename_dict)
    
    if mol_df is None:
        # Verify integrity of smiles and spec_id columns
        assert not msp_df["smiles"].isna().all(), "All values in smiles column are empty, data is incomplete!"
        assert not msp_df["spec_id"].isna().all(), "All values in spec_id column are empty, data does not meet expectations!"
        
        # Generate auto-increment for spec_id
        # msp_df.loc[:, "spec_id"] = np.arange(msp_df.shape[0])
        spec_df = msp_df
    else:
        # Generate spec_id from nist_num column
        msp_df["spec_id"] = msp_df["nist_num"]
        # Print missing value statistics
        print("Missing values per column:")
        print(msp_df.isna().sum())
        
        # Delete rows where inchi_key column value is None
        msp_df = msp_df[msp_df["inchi_key"].notna()]
        # Print missing value statistics
        print("Missing values per column:")
        print(msp_df.isna().sum())

        # Calculate SMILES based on inchi_key
        smiles_list, failed_count = inchi_to_smiles(msp_df["inchi_key"])
         # Assert that inchi_key and smiles_list have consistent length
        assert len(smiles_list) == len(msp_df["inchi_key"]), (
            f"Length mismatch: inchi_key ({len(msp_df['inchi_key'])}) and smiles_list ({len(smiles_list)})"
        )
        print(f"Failed to convert {failed_count} InChIKeys to SMILES")
        
        # Add SMILES to msp_df and delete original inchi_key column
        msp_df["smiles"] = smiles_list
        msp_df = msp_df.drop(columns=["inchi_key"])
        spec_df = msp_df

    # Print missing value statistics
    print("Missing values per column:")
    print(spec_df.isna().sum())
    
    # Reset index
    spec_df = spec_df.reset_index(drop=True)
    
    return spec_df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--msp_file', type=str, required=False)
    parser.add_argument('--mol_dir', type=str, required=False)
    parser.add_argument('--output_name', type=str, default='NIST_EI_23_df')
    parser.add_argument('--raw_data_dp', type=str, default='data/raw')
    parser.add_argument('--output_dp', type=str, default='data/processed')
    parser.add_argument('--num_entries', type=int, default=-1)
    parser.add_argument(
        '--output_type',
        type=str,
        default="json",
        choices=[
            "json",
            "csv"])
    args = parser.parse_args()
    msp_fp = ["/home/lwh/projects/lzq_3/EI-MS-2023"] 
    
    mol_df = None
    os.makedirs(args.output_dp, exist_ok=True)

    msp_df = preproc_msp(msp_fp, key_dict.keys(), args.num_entries)

    spec_df = merge_and_check(msp_df, mol_df, key_dict)
    # save files
    spec_df_fp = os.path.join(args.output_dp,
                              f"{args.output_name}.{args.output_type}")
    if args.output_type == "json":
        spec_df.to_json(spec_df_fp)
    else:
        assert args.output_type == "csv"
        spec_df.to_csv(spec_df_fp, index=False)

