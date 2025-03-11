import os
import glob
import subprocess

def submit_bam_creation_jobs(dry_run=True):
    """
    Submits sbatch jobs for each CSV file in cells_per_channel directory.
    
    Parameters:
    - dry_run: If True, only print the sbatch commands without submitting
    """
    # Configuration
    base_dir = "/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing"
    csv_dir = os.path.join(base_dir, "cells_per_channel")
    output_dir = os.path.join(base_dir, "Chip_GAMMA_sgRNA_output_BAMs")
    script_path = "/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/psp/aei_utils/create_bams_per_sgrna.py"
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "slurm_logs"), exist_ok=True)

    # Find all CSV files
    csv_files = glob.glob(os.path.join(csv_dir, "*_Chip-*-Channel-*.csv"))
    
    for csv_path in csv_files:
        # Extract base name without extension (e.g., ALPHA_Chip-1-Channel-1)
        prefix = os.path.basename(csv_path).replace(".csv", "")
        if "ALPHA" in prefix:
            continue
        if "BETA" in prefix:
            continue

        batch = prefix.split("_")[0]
        channel = prefix.split("_")[1]
        input_bam = f"/tscc/projects/ps-malilab/ydoctor/iPSC_Pan_Genome/Pan_Genome/cellranger_files/ALU_Editing/{batch}/{channel}/cellranger_outputs/outs/possorted_genome_bam.bam"
        
        # Create sbatch script content
        sbatch_content = f"""#!/bin/bash
#SBATCH --job-name={prefix}_bam
#SBATCH --partition=condo
#SBATCH --qos=condo
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem=256G
#SBATCH --account=csd852
#SBATCH -o /tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/logs/{prefix}_individual_bams.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ydoctor@ucsd.edu

eval "$(conda shell.bash hook)" 
conda activate subset_bam

python {script_path} \\
    --csv {csv_path} \\
    --input_bam {input_bam} \\
    --output_dir {output_dir} \\
    --prefix {prefix}_ \\
"""

        # Write sbatch file
        sbatch_path = os.path.join(output_dir, f"{prefix}.sbatch")
        with open(sbatch_path, "w") as f:
            f.write(sbatch_content)
        
        # Submit job or print command
        if dry_run:
            print(f"Would submit: {sbatch_path}")
            print(sbatch_content)
            print("-" * 80 + "\n")
        else:
            result = subprocess.run(["sbatch", sbatch_path], capture_output=True, text=True)
            print(f"Submitted {sbatch_path}: {result.stdout.strip()}")

if __name__ == "__main__":
    # Run in dry-run mode by default - set dry_run=False to actually submit jobs
    submit_bam_creation_jobs(dry_run=False)