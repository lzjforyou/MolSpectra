# MolSpectra

A deep learning framework for molecular spectrum prediction supporting multiple spectroscopy types including Infrared (IR), Nuclear Magnetic Resonance (NMR), Ultraviolet-Visible (UV-Vis), and Electron Ionization Mass Spectrometry (EI-MS).

## Features

- **Multi-modal Spectrum Prediction**: Support for IR, NMR, UV-Vis, and EI-MS spectroscopy
- **Multiple Molecular Representations**: Fingerprints, molecular graphs, SMILES, and various GNN architectures
- **Flexible Architecture**: Support for HDSE, PyG, and other molecular encoding methods
- **Library Matching**: Built-in functionality for spectral library matching
- **Configurable Training**: YAML-based configuration system for easy experimentation

## Environment Setup

### Conda Installation

1. Create a new conda environment with Python 3.9.20:

conda create -n HDSE-MS python=3.9.20

2. Activate the environment and install dependencies:

conda activate HDSE-MS
pip install -r requirements.txt

## Data Preparation

### Infrared Spectroscopy (IR)

**Experimental Data:**

- NIST Chemical WebBook: Available at https://webbook.nist.gov/chemistry/
- Chemotion Database: Available at https://zenodo.org/records/13318653
- Format: JCAMP-DX format
- Representation: Vectors of length 1801 with normalized absorbance values sampled every 2 cm⁻¹ over the 400-4000 cm⁻¹ range

**Simulated Data:**

- Source: Benchmark dataset by Marvin Alberts
- Access: https://rxn4chemistry.github.io/multimodal-spectroscopic-dataset/
- Representation: Vectors of length 1800 (400-4000 cm⁻¹, step size 2 cm⁻¹)
- Coverage: 794,403 deduplicated molecules containing C, H, O, N, S, P, Si, B, and halogens
- Additional Modalities: Includes NMR and tandem mass spectrometry data

### Nuclear Magnetic Resonance (NMR)

**Data Sources:**

- nmrshiftdb2:
  - Download from https://sourceforge.net/p/nmrshiftdb2/code/1624/
  - Or via GT-NMR repository: https://github.com/Anan-Wu-XMU/GT-NMR/tree/main/datasets
- HMDB: Available at https://www.hmdb.ca/downloads

### Ultraviolet-Visible Spectroscopy (UV-Vis)

**Experimental Data:**

- Source: NIST Chemical WebBook
- Access: https://webbook.nist.gov/chemistry/

### Electron Ionization Mass Spectrometry (EI-MS)

**Data Source:**

- NIST 23 Spectral Library: Commercial database available for purchase
- Information: https://www.nist.gov/
- Note: Employs logarithmic normalization during training (unlike other spectroscopy types)

## Usage

### Training Models

To train a model for a specific spectroscopy type, use the following command:

# For Infrared Spectroscopy
python src/train.py -c config/IR.yml
