import importlib
import re
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from pprint import pformat, pprint

from misc_utils import np_temp_seed, none_or_nan


# ELEMENT_LIST = ['H', 'C', 'O', 'N', 'P', 'S', 'Cl', 'F']
# ELEMENT_LIST = ['F', 'Hg', 'Cl', 'Pt', 'As', 'I', 'Co', 'C', 'Se', 'Gd', 'Au', 'Si', 'H', 'P', 'V', 'O', 'T', 'Sb', 'Cu', 'Sn', 'Ag', 'N', 'Cr', 'S', 'B', 'Fe', 'Br']
ELEMENT_LIST = ['Ag', 'Al', 'As', 'B', 'Be', 'Bi', 'Br', 'C', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'F', 'Fe', 'Gd', 'Ge', 'H', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nb', 'Ni', 'O', 'Os', 'P', 'Pb', 'Pd', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se', 'Si', 'Sn', 'Te', 'Th', 'Ti', 'Tl', 'U', 'V', 'W', 'Y', 'Zn', 'Zr'] # 60 elements
# Metals: Ag(Silver), Al(Aluminum), Be(Beryllium), Bi(Bismuth), Cd(Cadmium), Ce(Cerium), Co(Cobalt), Cr(Chromium), Cs(Cesium), Cu(Copper), Fe(Iron), Gd(Gadolinium), Ge(Germanium), Hf(Hafnium), Hg(Mercury), In(Indium), Ir(Iridium), K(Potassium), Li(Lithium), Mg(Magnesium), Mn(Manganese), Mo(Molybdenum), Na(Sodium), Nb(Niobium), Ni(Nickel), Os(Osmium), Pb(Lead), Pd(Palladium), Pt(Platinum), Rb(Rubidium), Re(Rhenium), Rh(Rhodium), Ru(Ruthenium), Sc(Scandium), Sn(Tin), Te(Tellurium), Th(Thorium), Ti(Titanium), Tl(Thallium), U(Uranium), V(Vanadium), W(Tungsten), Y(Yttrium), Zn(Zinc), Zr(Zirconium)
# Non-metals: As(Arsenic), B(Boron), Br(Bromine), C(Carbon), Cl(Chlorine), F(Fluorine), H(Hydrogen), I(Iodine), N(Nitrogen), O(Oxygen), P(Phosphorus), S(Sulfur), Se(Selenium), Si(Silicon), Sb(Antimony)
TWO_LETTER_TOKEN_NAMES = [
    'Al',
    'Ce',
    'Co',
    'Ge',
    'Gd',
    'Cs',
    'Th',
    'Cd',
    'As',
    'Na',
    'Nb',
    'Li',
    'Ni',
    'Se',
    'Sc',
    'Sb',
    'Sn',
    'Hf',
    'Hg',
    'Si',
    'Be',
    'Cl',
    'Rb',
    'Fe',
    'Bi',
    'Br',
    'Ag',
    'Ru',
    'Zn',
    'Te',
    'Mo',
    'Pt',
    'Mn',
    'Os',
    'Tl',
    'In',
    'Cu',
    'Mg',
    'Ti',
    'Pb',
    'Re',
    'Pd',
    'Ir',
    'Rh',
    'Zr',
    'Cr',
    '@@',
    'se',
    'si',
    'te']

LC_TWO_LETTER_MAP = {
    "se": "Se", "si": "Si", "te": "Te"
}

# these are all (exact) atomic masses
H_MASS = 1.007825  # 1.008
O_MASS = 15.994915  # 15.999
NA_MASS = 22.989771  # 22.990
N_MASS = 14.003074  # 14.007
C_MASS = 12.  # 12.011

JOBLIB_BACKEND = "loky" # Set joblib.Parallel backend to "loky". loky is the default backend designed to handle Python's GIL (Global Interpreter Lock), suitable for multi-process computation tasks
JOBLIB_N_JOBS = joblib.cpu_count() # Get the number of CPU cores on the current machine and use this value as the number of parallel tasks
JOBLIB_TIMEOUT = None  # Set task timeout limit to None, meaning no time limit on task execution. Default is 300 seconds for "loky"


def rdkit_import(*module_strs): # Function accepts one or more module names as parameters (module_strs), which are part of the RDKit library
    # rdkit_import function is a utility function for dynamically importing RDKit-related modules
    RDLogger = importlib.import_module("rdkit.RDLogger") # rdkit.RDLogger, which is the module RDKit uses to control logging
    RDLogger.DisableLog('rdApp.*') # Disable logging in RDKit
    modules = []
    for module_str in module_strs:
        modules.append(importlib.import_module(module_str))
    return tuple(modules) # Return tuple of modules


def normalize_ints(ints):

    total_ints = sum(ints)
    ints = [ints[i] / total_ints for i in range(len(ints))]
    return ints


def randomize_smiles(smiles, rseed, isomeric=False, kekule=False):
    """Perform a randomization of a SMILES string must be RDKit sanitizable"""
    if rseed == -1:
        return smiles
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    m = Chem.MolFromSmiles(smiles)
    assert not (m is None)
    ans = list(range(m.GetNumAtoms()))
    with np_temp_seed(rseed):
        np.random.shuffle(ans)
    nm = Chem.RenumberAtoms(m, ans)
    smiles = Chem.MolToSmiles(
        nm,
        canonical=False,
        isomericSmiles=isomeric,
        kekuleSmiles=kekule)
    assert not (smiles is None)
    return smiles


def split_smiles(smiles_str):

    token_list = []
    ptr = 0

    while ptr < len(smiles_str):
        if smiles_str[ptr:ptr + 2] in TWO_LETTER_TOKEN_NAMES:
            smiles_char = smiles_str[ptr:ptr + 2]
            if smiles_char in LC_TWO_LETTER_MAP:
                smiles_char = LC_TWO_LETTER_MAP[smiles_char]
            token_list.append(smiles_char)
            ptr += 2
        else:
            smiles_char = smiles_str[ptr]
            token_list.append(smiles_char)
            ptr += 1

    return token_list


def list_replace(l, d):
    return [d[data] for data in l]


def mol_from_inchi(inchi):
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        mol = Chem.MolFromInchi(inchi)
    except BaseException:
        mol = np.nan
    if none_or_nan(mol):
        mol = np.nan
    return mol


def rdkit_standardize(mol):
    # adapted from
    # https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
    modules = rdkit_import(
        "rdkit.Chem",
        "rdkit.Chem.MolStandardize",
        "rdkit.Chem.MolStandardize.rdMolStandardize")
    rdMolStandardize = modules[-1]
    mol = rdMolStandardize.Cleanup(mol)
    te = rdMolStandardize.TautomerEnumerator()
    mol = te.Canonicalize(mol)
    return mol


def mol_from_smiles(smiles, standardize=True):
    """Parse the SMILES representation of a molecule into an RDKit molecule object (mol), and standardize the molecule if needed"""
    # this is incompatible with rdkit version 2021.09 and greater
    modules = rdkit_import("rdkit.Chem")
    Chem = modules[0]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if standardize:
            mol = rdkit_standardize(mol)
    except BaseException:
        mol = np.nan
    if none_or_nan(mol):
        mol = np.nan
    return mol


def mol_to_smiles(
        mol,
        canonical=True, # Specify whether to generate canonical SMILES, which is a unique molecular representation that is always the same for the same molecular structure
        isomericSmiles=False, # Whether to include stereochemical information in isomeric SMILES. Enabling this option includes stereochemical information such as chiral center configurations in SMILES
        kekuleSmiles=False): # Whether to use Kekule form SMILES, i.e., not using aromaticity symbols (like 'b' or 'c') but using actual double bonds to represent aromatic rings
    modules = rdkit_import("rdkit.Chem") # rdkit_import function is a utility function for dynamically importing RDKit-related modules
    Chem = modules[0]
    try:
        smiles = Chem.MolToSmiles(
            mol,
            canonical=canonical,
            isomericSmiles=isomericSmiles,
            kekuleSmiles=kekuleSmiles)
    except BaseException:
        smiles = np.nan
    return smiles


def mol_to_formula(mol):
    """The main function of this function is to calculate the molecular formula based on the given molecule object (mol). The molecular formula is the chemical composition of the molecule, represented by chemical symbols and subscripts indicating the number of atoms of each element (e.g., C2H6O for ethanol)"""
    modules = rdkit_import("rdkit.Chem.AllChem") # rdkit_import function is a utility function for dynamically importing RDKit-related modules
    AllChem = modules[0]
    try:
        formula = AllChem.CalcMolFormula(mol)
    except BaseException:
        formula = np.nan
    return formula


def mol_to_inchikey(mol):
    modules = rdkit_import("rdkit.Chem.inchi")
    inchi = modules[0]
    try:
        inchikey = inchi.MolToInchiKey(mol)
    except BaseException:
        inchikey = np.nan
    return inchikey


def mol_to_inchikey_s(mol):
    """The main function of this code is to extract the first 14 characters (InChIKey prefix) from the standardized InChIKey string of a molecule object (mol) for unique molecular identification or quick lookup"""
    modules = rdkit_import("rdkit.Chem.inchi")
    inchi = modules[0]
    try:
        inchikey = inchi.MolToInchiKey(mol)
        inchikey_s = inchikey[:14]
    except BaseException:
        inchikey_s = np.nan
    return inchikey_s


def mol_to_inchi(mol):
    """The function of this code is to convert a given molecule object (mol) into its International Chemical Identifier (InChI). InChI is a standardized chemical molecule representation widely used for unique molecular identification and database storage."""
    modules = rdkit_import("rdkit.Chem.rdinchi")
    rdinchi = modules[0]
    try:
        # -SNon: Ignore stereochemical information (e.g., chirality). Return value: a tuple, first element is InChI string, second element is status information (error codes, etc.)
        inchi = rdinchi.MolToInchi(mol, options='-SNon')[0] 
    except BaseException:
        inchi = np.nan
    return inchi


def mol_to_mol_weight(mol, exact=True):
    """The function of this function is to calculate the molecular weight (MolWt) or exact molecular weight (ExactMolWt) based on the given molecule object (mol)"""
    modules = rdkit_import("rdkit.Chem.Descriptors")
    Desc = modules[0]
    try:
        if exact:
            mol_weight = Desc.ExactMolWt(mol)
        else:
            mol_weight = Desc.MolWt(mol)
    except BaseException:
        mol_weight = np.nan
    return mol_weight


def mol_to_charge(mol):
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        charge = rdmolops.GetFormalCharge(mol)
    except BaseException:
        charge = np.nan
    return charge


def check_neutral_charge(mol):
    """Function check_neutral_charge checks if a molecule object has neutral charge. Returns True if the total charge of the molecule is 0 (i.e., neutral molecule); otherwise returns False"""
    valid = mol_to_charge(mol) == 0
    return valid


def check_single_mol(mol):
    """
    Function check_single_mol checks if a molecule object is a single molecule (i.e., whether the molecule consists of a single fragment).
    Returns True if the molecule is a single molecule; returns False if the molecule consists of multiple fragments or is invalid
    """
    modules = rdkit_import("rdkit.Chem.rdmolops")
    rdmolops = modules[0]
    try:
        num_frags = len(rdmolops.GetMolFrags(mol))
    except BaseException:
        num_frags = np.nan
    valid = num_frags == 1
    return valid


def inchi_to_smiles(inchi):
    try:
        mol = mol_from_inchi(inchi)
        smiles = mol_to_smiles(mol)
    except BaseException:
        smiles = np.nan
    return smiles


def smiles_to_selfies(smiles):
    sf, Chem = rdkit_import("selfies", "rdkit.Chem")
    try:
        # canonicalize, strip isomeric information, kekulize
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(
            mol,
            canonical=False,
            isomericSmiles=False,
            kekuleSmiles=True)
        selfies = sf.encoder(smiles)
    except BaseException:
        selfies = np.nan
    return selfies


def make_morgan_fingerprint(mol, radius=3):
    """The function of this code is to generate the Morgan fingerprint of a molecule and convert it to NumPy array format. Morgan fingerprint is a substructure-based fingerprint used to represent molecular structural information"""
    modules = rdkit_import("rdkit.Chem.rdMolDescriptors", "rdkit.DataStructs")
    rmd = modules[0]
    ds = modules[1]
    # Generate sparse hash fingerprint, where each bit of the fingerprint corresponds to a specific molecular substructure. The fingerprint length is fixed (default 2048 bits), but can be adjusted via the radius parameter
    fp = rmd.GetHashedMorganFingerprint(mol, radius) 
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr) # Use ConvertToNumpyArray function to convert Morgan fingerprint object fp to NumPy array format
    return fp_arr


def make_rdkit_fingerprint(mol):
    """
    Generate the RDKit fingerprint of a molecule and convert it to NumPy array format.
    RDKit fingerprint is a path-based molecular fingerprint, similar to Daylight fingerprint. It generates fingerprints by traversing all paths in the molecule (usually limited to a certain maximum length, such as 7 bonds).
    The output is a sparse bit vector (ExplicitBitVect type), where each bit indicates whether a certain path exists. Default 2048 bits
    """
    chem, ds = rdkit_import("rdkit.Chem", "rdkit.DataStructs")
    fp = chem.RDKFingerprint(mol)
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def make_maccs_fingerprint(mol):
    """Generate the MACCS fingerprint of a molecule and convert it to NumPy array format. MACCS fingerprint is a rule-based molecular fingerprint consisting of 166 bits. Each bit represents whether a specific chemical structure or functional group exists"""
    maccs, ds = rdkit_import("rdkit.Chem.MACCSkeys", "rdkit.DataStructs")
    fp = maccs.GenMACCSKeys(mol)
    fp_arr = np.zeros(1)
    ds.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def split_selfies(selfies_str):
    selfies = importlib.import_module("selfies")
    selfies_tokens = list(selfies.split_selfies(selfies_str))
    return selfies_tokens


def seq_apply(iterator, func):
    """This code is a simple function that sequentially applies the specified function func to each element of the input iterator, and stores all results as a list to return"""
    result = []
    for i in iterator:
        result.append(func(i))
    return result


def par_apply(iterator, func):
    """This function uses the joblib library to apply a given function to each element in an iterator in parallel"""
    n_jobs = joblib.cpu_count() # Get the number of CPU cores on the current machine. This value is used to determine how many tasks can run simultaneously during parallel execution
    par_func = joblib.delayed(func) # joblib.delayed is a decorator used to wrap the func function in delayed execution form. This means func will not execute immediately, but will execute on demand when joblib.Parallel is called
    parallel = joblib.Parallel( # Create a joblib.Parallel instance, setting the backend, number of parallel tasks, and timeout, which come from previously defined global variables
        backend=JOBLIB_BACKEND,
        n_jobs=JOBLIB_N_JOBS,
        timeout=JOBLIB_TIMEOUT
    )
    result = parallel(par_func(i) for i in iterator) # Use list comprehension to create a sequence of delayed execution functions par_func, and execute these functions in parallel through the parallel object
    return result # Return the list of parallel processing results, where each element is the result after applying function func


def par_apply_series(series, func):
    """This code implements parallel processing of pandas.Series data type, applying the specified function func to each element of the series, and returning a new Series"""
    series_iter = tqdm(
        series.items(),
        desc=pformat(func),
        total=series.shape[0]) # Use the formatted string of func as the progress bar description

    def series_func(tup): return func(tup[1]) # Define an internal function series_func to extract tup[1] (the value part of the series) and pass it to func for processing. Ignore the index part tup[0]
    result_list = par_apply(series_iter, series_func)
    result_series = pd.Series(result_list, index=series.index) # Convert result_list to a new pandas.Series, maintaining the same index as the original series
    return result_series


def seq_apply_series(series, func):

    series_iter = tqdm(
        series.iteritems(),
        desc=pformat(func),
        total=series.shape[0])

    def series_func(tup): return func(tup[1])
    result_list = seq_apply(series_iter, series_func)
    result_series = pd.Series(result_list, index=series.index)
    return result_series


def par_apply_df_rows(df, func):
    
    df_iter = tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0])  
    def df_func(tup): return func(tup[1]) 
    result_list = par_apply(df_iter, df_func)
    if isinstance(result_list[0], tuple):
        result_series = tuple([pd.Series(rl, index=df.index)
                              for rl in zip(*result_list)])
    else:
        result_series = pd.Series(result_list, index=df.index)
    return result_series


def seq_apply_df_rows(df, func):
    """Apply the specified function func to each row of a pandas.DataFrame row by row, and return the result as one or more Series"""
    # df.iterrows(): Returns tuples for each row [(index_0, row_0),(index_1,row_1)], where index_i is the corresponding column index and row_i is the corresponding column value
    df_iter = tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0])
    def df_func(tup): return func(tup[1]) # Define an internal function df_func to extract row from the tuple and pass it to the user-defined function func
    result_list = seq_apply(df_iter, df_func) # Use seq_apply to apply df_func to each row of df_iter row by row. result_list is a list storing the processing results of func for each row
    if isinstance(result_list[0], tuple):
        result_series = tuple([pd.Series(rl, index=df.index) # Use zip(*result_list) to unpack and split the results into multiple columns. Create a pandas.Series for each column with the same index as the original DataFrame
                              for rl in zip(*result_list)]) # Return a tuple containing multiple Series
    else:
        result_series = pd.Series(result_list, index=df.index) # Convert result_list to a Series with the same index as the original DataFrame
    return result_series


def parse_ace_str(ce_str):
    """The function of this code is to parse collision energy strings (ce_str), extract the absolute collision energy value, and convert it to float"""
    # ce_str: String representing collision energy, may contain various formats (e.g., "30 eV", "20HCD", "NCE=10% 30eV", etc.)
    if none_or_nan(ce_str):
        return np.nan
    matches = {
        # nist ones
        # this case is ambiguous (float(x) >= 2. or float(x) == 0.)
        r"^[\d]+[.]?[\d]*$": lambda x: float(x), # Rule 1: Pure number (e.g., "30")
        r"^[\d]+[.]?[\d]*[ ]?eV$": lambda x: float(x.rstrip(" eV")), # Rule 2: Number plus unit "eV" (e.g., "30 eV")
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[1].rstrip("eV")), # Rule 3: Compound format with "NCE" prefix (e.g., "NCE=10% 30eV")
        # other ones
        r"^[\d]+[.]?[\d]*HCD$": lambda x: float(x.rstrip("HCD")), # Number plus "HCD" suffix (e.g., "20HCD")
        r"^CE [\d]+[.]?[\d]*$": lambda x: float(x.lstrip("CE ")), # Format with "CE" prefix (e.g., "CE 30")
        r"^[\d]+HCD$": lambda x: float(x.rstrip("HCD")),  # New rule: Match 65HCD format
        # New rules
        r"^[\d]+[.]?[\d]*V$": lambda x: float(x.rstrip("V")),  # Number plus "V" suffix (e.g., "6V")
        r"^[\d]+[.]?[\d]* [Vv]$": lambda x: float(x.rstrip(" Vv")),  # Number plus " V" suffix (e.g., "30 V")
    }
    for k, v in matches.items(): # Iterate through each regex and corresponding processing function in matches
        if re.match(k, ce_str): # If the current regex k matches the input string ce_str, call the processing function v(ce_str) to parse and return the result
            return v(ce_str)
    return np.nan


def parse_nce_str(ce_str):
    """The function of this code is to parse collision energy strings (ce_str), extract the Normalized Collision Energy (NCE), and normalize it to float"""

    if none_or_nan(ce_str):
        return np.nan
    matches = {
        # nist ones
        r"^NCE=[\d]+[.]?[\d]*% [\d]+[.]?[\d]*eV$": lambda x: float(x.split()[0].lstrip("NCE=").rstrip("%")), # Rule 1: NCE=10% 30eV format. Extract the numerical value from the NCE=10% part at the beginning of the string
        r"^NCE=[\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("NCE=").rstrip("%")), # Rule 2: NCE=10% format. Remove NCE= and %, keep the numerical value
        # other ones
        # this case is ambiguous
        r"^[\d]+[.]?[\d]*$": lambda x: 100. * float(x) if float(x) < 2. else np.nan, # Rule 3: Single number (e.g., 1.5). If the number is less than 2, return 100 * value, otherwise return np.nan
        r"^[\d]+[.]?[\d]*[ ]?[%]? \(nominal\)$": lambda x: float(x.rstrip(" %(nominal)")), # Rule 4: 50% (nominal) format. Extract: remove %(nominal)
        r"^HCD [\d]+[.]?[\d]*%$": lambda x: float(x.lstrip("HCD ").rstrip("%")), # Rule 5: HCD 10% format. Extract: remove HCD and %
        r"^[\d]+[.]?[\d]* NCE$": lambda x: float(x.rstrip("NCE")), # Match 50 NCE format. Remove NCE at the end of the string, keep the preceding number. Convert the extracted number to float
        r"^[\d]+[.]?[\d]*\(NCE\)$": lambda x: float(x.rstrip("(NCE)")), # Match 50(NCE) format. Remove (NCE) at the end of the string, keep the preceding number. Convert the extracted number to float
        r"^[\d]+[.]?[\d]*[ ]?%$": lambda x: float(x.rstrip(" %")), # Rule 8: Match 50% or 50 % format. Remove % at the end of the string, keep the preceding number and convert to float
        r"^HCD \(NCE [\d]+[.]?[\d]*%\)$": lambda x: float(x.lstrip("HCD (NCE").rstrip("%)")), # Match HCD (NCE 50%) format. Remove HCD (NCE at the beginning and %) at the end of the string, keep the number and convert to float
        r"^[\d]+[.]?[\d]* \(nominal\)$": lambda x: float(x.rstrip(" (nominal)")),  # New rule: Match 180 (nominal) format
    }
    for k, v in matches.items(): # Iterate through each regex and corresponding processing function in matches
        if re.match(k, ce_str): # If the current regex k matches the input string ce_str, call the processing function v(ce_str) to parse and return the result
            return v(ce_str)
    return np.nan


def parse_inst_info(df):
    """The function of this function is to parse mass spectrometer instrument information and fragmentation mode, and standardize this information into a unified format"""
    # instrument type 
    return "EI", "EI"



def parse_ion_mode_str(ion_mode_str):
    """The function of this code is to parse the ion mode string (ion_mode_str) in mass spectrometry data and standardize it into a unified abbreviation format (e.g., P for positive ion mode, N for negative ion mode)"""
    return "EI"


def parse_ri_str(ri_str):
    """The function of this code is to parse the Retention Index (ri_str) and convert it from string form to float"""
    if none_or_nan(ri_str):
        return np.nan
    else:
        return float(ri_str)


def parse_prec_type_str(prec_type_str):
    """The function of this function is to parse the precursor type string (prec_type_str) in mass spectrometry data and standardize it into a unified format"""
    if none_or_nan(prec_type_str):
        return np.nan
    if prec_type_str == "EI":
        return "EI"
    elif prec_type_str.endswith("1+"):
        return prec_type_str.replace("1+", "+")
    elif prec_type_str.endswith("1-"):
        return prec_type_str.replace("1-", "-")
    else:
        return prec_type_str


def parse_peaks_str(peaks_str):
    """
    Parse spectral peak data from string to structured data (list form).
    Supports each peak as 'm/z intensity;', multi-line data, automatically handles whitespace.
    """
    if peaks_str is None or str(peaks_str).strip().lower() in ['nan', 'none', '']:
        return None  # Or np.nan, if you prefer using np.nan as missing value

    peaks = []
    # Concatenate all lines into one string, then split by semicolon
    for peak in peaks_str.replace('\n', ' ').split(';'):
        peak = peak.strip()
        if not peak:
            continue
        parts = peak.split()
        if len(parts) != 2:
            # Incorrect format, can choose to report error or skip
            continue
        try:
            mz = float(parts[0])
            intensity = float(parts[1])
            peaks.append((mz, intensity))
        except ValueError:
            # Non-numeric, skip or report error
            continue
    return peaks



def convert_peaks_to_float(peaks):
    """The function of this code is to convert peak values (peaks) in mass spectrometry data from string format to float format"""
    # assumes no nan
    float_peaks = []
    for peak in peaks:
        float_peaks.append((float(peak[0]), float(peak[1])))
    return float_peaks


def get_res(peaks):
    """Fixed return 1"""
    return 1



def get_murcko_scaffold(mol, output_type="smiles", include_chirality=False):
    """The function of this code is to calculate the Murcko scaffold (core structure) of a given molecule, commonly used for chemical structure classification and feature extraction. Murcko scaffold is the core ring system and connectors retained after removing side chains from the molecule"""
    if none_or_nan(mol):
        return np.nan
    MurckoScaffold = importlib.import_module(
        "rdkit.Chem.Scaffolds.MurckoScaffold")
    if output_type == "smiles":
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=include_chirality)
    else:
        raise NotImplementedError
    return scaffold


def atom_type_one_hot(atom):

    chemutils = importlib.import_module("dgllife.utils")
    return chemutils.atom_type_one_hot(
        atom, allowable_set=ELEMENT_LIST, encode_unknown=True
    )


def atom_bond_type_one_hot(atom):
    """
    Function:
    Perform one-hot encoding on the bond types of the input atom, and return a boolean list indicating whether the atom has specific types of bonds.
    Input parameters:
    atom: An RDKit Atom object representing a certain atom in the molecule.
    Output:
    A boolean list of length 4, each element indicates whether the atom has the corresponding type of bond (e.g., single bond, double bond, etc.).
    """
    chemutils = importlib.import_module("dgllife.utils")
    bs = atom.GetBonds() # Call atom.GetBonds() to get all bonds connected to this atom. Returns a list containing Bond objects
    if not bs:
        return [False, False, False, False] # If the atom has no bonds (i.e., bs is empty), directly return a boolean list of length 4 [False, False, False, False]
    # Result is a 2D NumPy array bt with shape [n,4], where: n is the number of bonds around the atom. Each row is the one-hot encoding of a certain bond, with length 4 (assuming there are 4 types of bonds in total)
    bt = np.array([chemutils.bond_type_one_hot(b) for b in bs])
    # Iterate through each column of the one-hot encoding array bt (corresponding to each bond type), check if at least one bond belongs to that type. Return a boolean list of length 4, each element indicates whether the atom has the corresponding type of bond
    return [any(bt[:, i]) for i in range(bt.shape[1])]


def analyze_mol(mol):

    import rdkit
    from rdkit.Chem.Descriptors import MolWt
    import rdkit.Chem as Chem
    mol_dict = {}
    mol_dict["num_atoms"] = mol.GetNumHeavyAtoms()
    mol_dict["num_bonds"] = mol.GetNumBonds(onlyHeavy=True)
    mol_dict["mol_weight"] = MolWt(mol)
    mol_dict["num_rings"] = len(list(Chem.GetSymmSSSR(mol)))
    mol_dict["max_ring_size"] = max(
        [-1] + [len(list(atom_iter)) for atom_iter in Chem.GetSymmSSSR(mol)])
    cnops_counts = {
        "C": 0,
        "N": 0,
        "O": 0,
        "P": 0,
        "S": 0,
        "Cl": 0,
        "other": 0}
    bond_counts = {"single": 0, "double": 0, "triple": 0, "aromatic": 0}
    cnops_bond_counts = {"C": [-1], "N": [-1],
                         "O": [-1], "P": [-1], "S": [-1], "Cl": [-1]}
    h_counts = 0
    p_num_bonds = [-1]
    s_num_bonds = [-1]
    other_atoms = set()
    for atom in mol.GetAtoms():
        atom_symbol = atom.GetSymbol()
        if atom_symbol in cnops_counts:
            cnops_counts[atom_symbol] += 1
            cnops_bond_counts[atom_symbol].append(len(atom.GetBonds()))
        else:
            cnops_counts["other"] += 1
            other_atoms.add(atom_symbol)
        h_counts += atom.GetNumImplicitHs()
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        if bond_type == rdkit.Chem.rdchem.BondType.SINGLE:
            bond_counts["single"] += 1
        elif bond_type == rdkit.Chem.rdchem.BondType.DOUBLE:
            bond_counts["double"] += 1
        elif bond_type == rdkit.Chem.rdchem.BondType.TRIPLE:
            bond_counts["triple"] += 1
        else:
            assert bond_type == rdkit.Chem.rdchem.BondType.AROMATIC
            bond_counts["aromatic"] += 1
    mol_dict["other_atoms"] = ",".join(sorted(list(other_atoms)))
    mol_dict["H_counts"] = h_counts
    for k, v in cnops_counts.items():
        mol_dict[f"{k}_counts"] = v
    for k, v in bond_counts.items():
        mol_dict[f"{k}_counts"] = v
    for k, v in cnops_bond_counts.items():
        mol_dict[f"{k}_max_bond_counts"] = max(v)
    return mol_dict


def check_atoms(mol, element_list=ELEMENT_LIST):
    """If all atom symbols in the molecule are in element_list, return True, otherwise return False"""
    rdkit = importlib.import_module("rdkit")
    valid = all(a.GetSymbol() in element_list for a in mol.GetAtoms())
    return valid


def check_num_bonds(mol):
    """This code defines a function check_num_bonds to check if a molecule object has at least one chemical bond. Returns True if there are chemical bonds in the molecule, otherwise returns False"""
    rdkit = importlib.import_module("rdkit")
    valid = mol.GetNumBonds() > 0
    return valid


CHARGE_FACTOR_MAP = {
    1: 1.00,
    2: 0.90,
    3: 0.85,
    4: 0.80,
    5: 0.75,
    "large": 0.75
}


def get_charge(prec_type_str):

    if prec_type_str == "EI":
        return 1
    end_brac_idx = prec_type_str.index("]")  # Find the position of the bracket in the precursor type string
    charge_str = prec_type_str[end_brac_idx + 1:]  # Extract the charge string
    # Handle cases with only sign but no number
    if charge_str == "-":
        charge_str = "1-"
    elif charge_str == "+":
        charge_str = "1+"
    assert len(charge_str) >= 2
    sign = charge_str[-1] # Extract the sign
    assert sign in ["+", "-"]
    magnitude = int(charge_str[:-1]) # Extract the charge magnitude
    # Determine the sign of the charge based on the sign
    if sign == "+":
        charge = magnitude
    else:
        charge = -magnitude
    return charge


def nce_to_ace_helper(nce, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    ace = (nce * prec_mz * charge_factor) / 500.
    return ace


def ace_to_nce_helper(ace, charge, prec_mz):

    if charge in CHARGE_FACTOR_MAP:
        charge_factor = CHARGE_FACTOR_MAP[charge]
    else:
        charge_factor = CHARGE_FACTOR_MAP["large"]
    nce = (ace * 500.) / (prec_mz * charge_factor)
    return nce


def nce_to_ace(row):

    prec_mz = row["prec_mz"] # Precursor mass-to-charge ratio
    nce = row["nce"]  # Normalized collision energy
    prec_type = row["prec_type"]  # Precursor type
    charge = np.abs(get_charge(prec_type)) # Calculate charge number, use custom get_charge function to parse charge number from precursor type
    ace = nce_to_ace_helper(nce, charge, prec_mz)
    return ace


def ace_to_nce(row):

    prec_mz = row["prec_mz"]
    ace = row["ace"]
    prec_type = row["prec_type"]
    charge = np.abs(get_charge(prec_type))
    nce = ace_to_nce_helper(ace, charge, prec_mz)
    return nce


def parse_formula(formula):

    element_counts = {element: 0 for element in ELEMENT_LIST}
    cur_element = None
    cur_count = 1
    for token in re.findall('[A-Z][a-z]?|\\d+|.', formula):
        if token.isalpha():
            if cur_element is not None:
                assert element_counts[cur_element] == 0
                element_counts[cur_element] += cur_count
            cur_element = token
            cur_count = 1
        elif token.isdigit():
            cur_count = int(token)
        else:
            raise ValueError(f"Invalid token {token}")
    assert element_counts[cur_element] == 0
    element_counts[cur_element] += cur_count
    return element_counts


def check_mol_props(df):
    """This code implements validation of molecular properties in a DataFrame. It filters out molecular data that meets the requirements through a series of check functions and returns the validated DataFrame"""
    # The following four check atom types, number of bonds, neutral charge, single mol, return boolean Series
    valid_atoms = par_apply_series(df["mol"], check_atoms)
    valid_num_bonds = par_apply_series(df["mol"], check_num_bonds)
    valid_charge = par_apply_series(df["mol"], check_neutral_charge)
    valid_single_mol = par_apply_series(df["mol"], check_single_mol)
    print(
        f"mol filters: atoms={valid_atoms.sum()}, num_bonds={valid_num_bonds.sum()}, charge={valid_charge.sum()}, single_mol={valid_single_mol.sum()}")
    df = df[valid_atoms & valid_num_bonds & valid_charge & valid_single_mol] # Filter molecules that meet all conditions
    return df
