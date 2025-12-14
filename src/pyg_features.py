import numpy as np
from typing import Union, List
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdPartialCharges
import rdkit.Chem as Chem

atom_features = [
    'chiral_center',
    'cip_code',
    'crippen_log_p_contrib',
    'crippen_molar_refractivity_contrib',
    'degree',
    'element',
    'formal_charge',
    'gasteiger_charge',
    'hybridization',
    'is_aromatic',
    'is_h_acceptor',
    'is_h_donor',
    'is_hetero',
    'labute_asa_contrib',
    'mass',
    'num_hs',
    'num_valence',
    'tpsa_contrib',
    'atom_in_ring',
]

bond_features = [
    'bondstereo',
    'bondtype',
    'is_conjugated',
    'is_rotatable',
    'bond_dir',
    'bond_is_in_ring',
]

# Returns a list of floats with length matching allowable_set. Each item indicates whether x equals the corresponding value in allowable_set, with 1.0 if equal, otherwise 0.0
def onehot_encode(x: Union[float, int, str],
                  allowable_set: List[Union[float, int, str]]) -> List[float]:
    result = list(map(lambda s: float(x == s), allowable_set))
    return result

# This code checks if x is None or NaN, and if so, replaces x with 0.0. Finally returns a list containing only the value of x
def encode(x: Union[float, int, str]) -> List[float]:
    if x is None or np.isnan(x):
        x = 0.0
    return [float(x)]

# This function is used to extract all features for a single chemical bond in a molecule and combine them into a feature vector.
# bond_featurizer function dynamically calls each feature extraction function (e.g., bondtype(bond)) via globals()[bond_feature](bond)
def bond_featurizer(bond: Chem.Bond) -> np.ndarray:
    return np.concatenate([
        globals()[bond_feature](bond) for bond_feature in bond_features
    ], axis=0)

# This function is similar to bond_featurizer, but it is used to extract features for each atom in a molecule.
def atom_featurizer(atom: Chem.Atom) -> np.ndarray:
    return np.concatenate([
        globals()[atom_feature](atom) for atom_feature in atom_features
    ], axis=0)

# This function is used to determine whether a chemical bond belongs to a ring structure, returns a list containing 0 or 1. 1 indicates the bond belongs to a ring, 0 indicates it does not
def is_in_ring(bond: Chem.Bond) -> List[float]:
    return encode(
        x=bond.IsInRing()
    )

# This function is used to extract the type of chemical bond (such as single bond, double bond, triple bond, aromatic bond, etc.) and perform one-hot encoding
def bondtype(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetBondType(),
        allowable_set=[
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
    )

# This function is used to determine whether a chemical bond is a conjugated bond
def is_conjugated(bond):
    return encode(
        x=bond.GetIsConjugated()
    )

# Returns a list generated through one-hot encoding, indicating the direction type of the chemical bond
def bond_dir(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetBondDir(),
        allowable_set=[
            Chem.rdchem.BondDir.NONE,
            Chem.rdchem.BondDir.ENDUPRIGHT,
            Chem.rdchem.BondDir.ENDDOWNRIGHT,
        ]
    )

# Returns a list indicating whether the bond is rotatable. Uses encode function to convert boolean value (whether rotatable) to numeric 0 or 1
def is_rotatable(bond: Chem.Bond) -> List[float]:
    mol = bond.GetOwningMol()
    atom_indices = tuple(
        sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]))
    return encode(
        x=atom_indices in Lipinski._RotatableBonds(mol)
    )

# Returns a list generated through one-hot encoding, indicating the stereochemical type of the chemical bond
def bondstereo(bond: Chem.Bond) -> List[float]:
    return onehot_encode(
        x=bond.GetStereo(),
        allowable_set=[
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
    )

# Returns a one-hot encoded list indicating the size of the ring the bond is in. Possible ring sizes include: 0 (not in ring), 3, 4, 5, 6, 7, 8, 9, 10, representing different ring sizes
def bond_is_in_ring(bond) -> List[float]:
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
        if bond.IsInRingSize(ring_size): break
    return onehot_encode(
        x=ring_size,
        allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10]
    )

# Returns a one-hot encoded list indicating the explicit valence of the atom. Possible values are [1, 2, 3, 4, 5, 6]
def ExplicitValence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetExplicitValence(),
        allowable_set=[1, 2, 3, 4, 5, 6]
    )

# Returns a one-hot encoded list indicating the implicit valence of the atom. Possible values are [0, 1, 2, 3]
def ImplicitValence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetImplicitValence(),
        allowable_set=[0, 1, 2, 3]
    )

# This function is used to invert chirality, returns a numeric list indicating the result after inversion. Uses encode function to convert result (boolean) to numeric 0 or 1
def invert_Chirality(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.InvertChirality()
    )

# Returns a one-hot encoded list indicating the total degree of the atom. Possible values are [1, 2, 3, 4]
def Total_degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetTotalDegree(),
        allowable_set=[1, 2, 3, 4]
    )

# Returns a one-hot encoded list indicating the number of explicit hydrogen atoms on the atom. Possible values are [0, 1]
def Num_ExplicitHs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetNumExplicitHs(),
        allowable_set=[0, 1]
    )

# Returns a numeric list indicating whether the atom belongs to a ring. Uses encode function to convert boolean (whether in ring) to numeric 0 or 1
def atom_in_ring(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.IsInRing()
    )

# Returns a numeric list indicating whether the atom is a chiral center. Uses encode function to convert boolean (whether chiral) to numeric 0 or 1
def chiral_center(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.HasProp("_ChiralityPossible")
    )

# This function is used to get the CIP stereochemical symbol of an atom, i.e., the R or S configuration symbol (Cahn-Ingold-Prelog rules)
# Returns a one-hot encoded list indicating the CIP code of the atom. Possible values are ["R", "S"], if atom has no CIP code, returns [0.0, 0.0]
def cip_code(atom: Chem.Atom) -> List[float]:
    if atom.HasProp("_CIPCode"):
        return onehot_encode(
            x=atom.GetProp("_CIPCode"),
            allowable_set=[
                "R", "S"
            ]
        )
    return [0.0, 0.0]

# This function is used to get the chiral tag (ChiralTag) of an atom, indicating the stereochemical type of the atom, returns a one-hot encoded list indicating the chirality type of the atom
def ChiralTag(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetChiralTag(),
        allowable_set=[
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,  # Unspecified chirality type
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,  # Tetrahedral clockwise rotation
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,  # Tetrahedral counterclockwise rotation
        ]
    )

# Returns a one-hot encoded list indicating the element symbol of the atom. Supported element symbols include: ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
def element(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetSymbol(),
        allowable_set=['F', 'Hg', 'Cl', 'Pt', 'As', 'I', 'Co', 'C', 'Se', 'Gd', 'Au', 'Si', 'H', 'P', 'V', 'O', 'T', 'Sb', 'Cu', 'Sn', 'Ag', 'N', 'Cr', 'S', 'B', 'Fe', 'Br']
    )

# Returns a one-hot encoded list indicating the hybridization type of the atom
def hybridization(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetHybridization(),
        allowable_set=[
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
        ]
    )

# Returns a one-hot encoded list indicating the formal charge of the atom. Possible values are [-1, 0, 1]
def formal_charge(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetFormalCharge(),
        allowable_set=[-1, 0, 1]
    )

# Returns a numeric list indicating the mass of the atom divided by 100
def mass(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetMass() / 100
    )

# Returns a numeric list indicating whether the atom is aromatic. Boolean True converts to 1.0, False converts to 0.0
def is_aromatic(atom: Chem.Atom) -> List[float]:
    return encode(
        x=atom.GetIsAromatic()
    )

# Returns a one-hot encoded list indicating the number of hydrogen atoms on the atom. Possible values are [0, 1, 2, 3]
def num_hs(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetTotalNumHs(),
        allowable_set=[0, 1, 2, 3]
    )

# Returns a one-hot encoded list indicating the total valence of the atom. Possible values are [1, 2, 3, 4, 5, 6]
def num_valence(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetTotalValence(),
        allowable_set=[1, 2, 3, 4, 5, 6])

# Returns a one-hot encoded list indicating the degree of the atom. Possible values are [1, 2, 3, 4]
def degree(atom: Chem.Atom) -> List[float]:
    return onehot_encode(
        x=atom.GetDegree(),
        allowable_set=[1, 2, 3, 4]
    )

# Returns a one-hot encoded list indicating the size of the ring the atom is in. Possible values are [0, 3, 4, 5, 6, 7, 8, 9, 10]
def is_in_ring_size_n(atom: Chem.Atom) -> List[float]:
    for ring_size in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
        if atom.IsInRingSize(ring_size): break
    return onehot_encode(
        x=ring_size,
        allowable_set=[0, 3, 4, 5, 6, 7, 8, 9, 10]
    )

# Returns a numeric list indicating whether the atom is a heteroatom. Boolean True converts to 1.0, False converts to 0.0
def is_hetero(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._Heteroatoms(mol)]
    )

# Returns a numeric list indicating whether the atom is a hydrogen bond donor. Boolean True converts to 1.0, False converts to 0.0
# Lipinski._HDonors(mol) gets indices of all hydrogen bond donors in the molecule. atom.GetIdx() gets the index of the current atom and checks if it is a hydrogen bond donor. encode(x) converts boolean to 1.0 or 0.0
def is_h_donor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._HDonors(mol)]
    )

# Returns a numeric list indicating whether the atom is a hydrogen bond acceptor. Boolean True converts to 1.0, False converts to 0.0
def is_h_acceptor(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=atom.GetIdx() in [i[0] for i in Lipinski._HAcceptors(mol)]
    )

# Returns a numeric list indicating the LogP contribution value of the atom.
def crippen_log_p_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=Crippen._GetAtomContribs(mol)[atom.GetIdx()][0]
    )

# This function is used to calculate Crippen molar refractivity contribution, i.e., the atom's molar refractivity contribution in the molecule.
def crippen_molar_refractivity_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=Crippen._GetAtomContribs(mol)[atom.GetIdx()][1]
    )

# This function is used to calculate topological polar surface area contribution (TPSA, Topological Polar Surface Area), i.e., the atom's polarity contribution in the molecule.
def tpsa_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=rdMolDescriptors._CalcTPSAContribs(mol)[atom.GetIdx()]
    )

# This function is used to calculate Labute surface area contribution (Labute ASA), i.e., the atom's surface area contribution in the molecule, returns a numeric list indicating the Labute ASA contribution value of the atom
def labute_asa_contrib(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    return encode(
        x=rdMolDescriptors._CalcLabuteASAContribs(mol)[0][atom.GetIdx()]
    )

# Returns a numeric list indicating the Gasteiger charge of the atom
def gasteiger_charge(atom: Chem.Atom) -> List[float]:
    mol = atom.GetOwningMol()
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return encode(
        x=atom.GetDoubleProp('_GasteigerCharge')
    )
