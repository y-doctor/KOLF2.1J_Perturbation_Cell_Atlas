# KOLF Perturbation Atlas

Computational framework for analyzing single-cell CRISPR perturbation sequencing data from the KOLF2.1J Perturbation Cell Atlas. This project contains the custom scripts used for quality control, preprocessing, differential expression analysis, and downstream analysis of perturbation effects. See methods section and SI fig 2 for full explanation of methods used. 

## Overview

The pipeline contains several submodules from raw data processing to advanced downstream analysis, including:

- **Quality Control**: Cell and gene filtering, batch effect handling
- **Preprocessing**: Normalization, cell filtering, perturbation validation
- **Differential Expression**: Pseudo-bulk analysis with DESeq2
- **Downstream Analysis**: Energy-based tests, perturbation correlation, complex mapping
- **Visualization**: Comprehensive plotting utilities

These are also extensible to analyzing other single cell CRISPRi Perturb-Seq experiments. 

## Project Structure

```
├── psp/                          # Main analysis package
│   ├── qc/                       # Quality control modules
│   ├── pp/                       # Preprocessing modules
│   ├── de/                       # Differential expression analysis
│   ├── da/                       # Data analysis and downstream 
│   ├── pl/                       # Plotting utilities
│   ├── utils/                    # Utility functions
│   └── notebooks/                # Example notebooks
```

## Installation

### Prerequisites

- Python 3.11+
- Conda or Miniconda

### Setup

1. **Clone the repository**:
   ```bash
   git clone <https://github.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas.git>
   cd KOLF_Perturbation_Atlas
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda env create -f environment.yml
   conda activate perturb_seq_env
   ```
   Typical installation time on a normal desktop computer: ~30 min. This software has been tested on a linux machine in a high performance computing (HPC) cluster. No non-standard hardware is required. 

## Usage

See the processed data notebooks in the notebooks directory. Input files required to perform this analysis are present within the input_files subdirectory other than the raw .h5mu files and protospacer_calls_per_cell files which can be found at: https://figshare.com/s/ee85bb1880921326249b. A demo file can be found in the notebooks subdirectory as well for the smaller scale chromatin modifiers dataset. This notebook can typically be run in ~30 min on a linux machine using 32 CPU cores and 256 GB of RAM, or ~2 hours on a "normal" dektop computer. For the full dataset, we reccomend, if available, 64 CPU cores and 1TB of RAM, although 32 CPU cores and 256 GB of RAM will suffice. 

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
https://doi.org/10.1101/2024.11.03.621734
```
