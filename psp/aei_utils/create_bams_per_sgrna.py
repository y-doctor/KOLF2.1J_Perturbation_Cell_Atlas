#!/usr/bin/env python3
import argparse
import os
import time
import pysam
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import psutil

def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a full BAM file in chunks and write separate BAM files per gRNA with progress and memory usage reports."
    )
    parser.add_argument("--csv", required=True,
                        help="CSV file mapping cell_barcode to gRNA (columns: cell_barcode,gRNA)")
    parser.add_argument("--input_bam", required=True, help="Input BAM file")
    parser.add_argument("--output_dir", required=True, help="Output directory for the resulting BAM files")
    parser.add_argument("--prefix", required=True,
                        help="Prefix for output files. The output files will be named as <prefix><gRNA>.bam")
    parser.add_argument("--chunk_size", type=int, default=10000000,
                        help="Number of reads per chunk (default: 10,000,000)")
    return parser.parse_args()

def print_memory_usage():
    proc = psutil.Process(os.getpid())
    mem_used = proc.memory_info().rss / (1024**3)  # in GB
    total_mem = psutil.virtual_memory().total / (1024**3)  # in GB
    print(f"Memory usage: {mem_used:.2f} GB / {total_mem:.2f} GB", flush=True)

def main():
    args = parse_args()

    # Ensure output directory exists.
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSV mapping.
    mapping_df = pd.read_csv(args.csv)
    barcode_to_gRNA = dict(zip(mapping_df['cell_barcode'], mapping_df['gRNA']))
    print(f"Loaded mapping for {len(barcode_to_gRNA)} cell barcodes.", flush=True)

    # Use idxstats to get the total number of reads.
    idxstats_output = pysam.idxstats(args.input_bam)
    total_reads = sum(int(line.split()[2]) for line in idxstats_output.splitlines())
    print(f"Total reads in BAM (from idxstats): {total_reads}", flush=True)

    # Retrieve header for writing.
    header = pysam.AlignmentFile(args.input_bam, "rb").header

    # Open input BAM.
    in_bam = pysam.AlignmentFile(args.input_bam, "rb")
    read_iterator = in_bam.fetch(until_eof=True)

    total_processed = 0
    overall_start = time.time()
    last_memory_print = time.time()

    # Create a global progress bar.
    progress = tqdm(total=total_reads, desc="Processing BAM", mininterval=20)

    # Dictionary for persistent output file handles.
    out_handles = {}

    while True:
        # Process one chunk.
        groups = defaultdict(list)
        chunk_count = 0

        while chunk_count < args.chunk_size:
            try:
                read = next(read_iterator)
            except StopIteration:
                break  # End of BAM file.
            try:
                cell_barcode = read.get_tag("CB")
            except KeyError:
                continue  # Skip reads without 'CB'.
            if cell_barcode in barcode_to_gRNA:
                gRNA = barcode_to_gRNA[cell_barcode]
                groups[gRNA].append(read)
            chunk_count += 1
            total_processed += 1
            progress.update(1)

            # Every 20 seconds, print memory usage.
            current_time = time.time()
            if current_time - last_memory_print >= 20:
                print("\n[Processing Chunk] ", end="", flush=True)
                print_memory_usage()
                last_memory_print = current_time

        if chunk_count == 0:
            break  # No more reads in the current chunk; exit loop.

        # Write each group in this chunk to persistent file handles.
        for gRNA, reads in groups.items():
            if gRNA not in out_handles:
                out_filename = os.path.join(args.output_dir, f"{args.prefix}{gRNA}.bam")
                # Open new file for writing; keep it open persistently.
                out_handles[gRNA] = pysam.AlignmentFile(out_filename, "wb", header=header, threads=1)
            for read in reads:
                out_handles[gRNA].write(read)
        groups.clear()  # Free memory for this chunk.

    progress.close()
    in_bam.close()

    # Close all persistent output file handles.
    for out_bam in out_handles.values():
        out_bam.close()

    overall_time = time.time() - overall_start
    print(f"\nTotal processed reads: {total_processed}", flush=True)
    print(f"Total processing time: {overall_time/60:.2f} minutes.", flush=True)
    print("Final memory usage:", end=" ", flush=True)
    print_memory_usage()

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
    output_dir = os.path.join(base_dir, "bam_outputs")
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
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --account=csd852
#SBATCH -o {output_dir}/slurm_logs/{prefix}_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ydoctor@ucsd.edu

eval "$(conda shell.bash hook)" 
conda activate subset_bam

python {script_path} \\
    --csv {csv_path} \\
    --input_bam {input_bam} \\
    --output_dir {output_dir}/{prefix} \\
    --prefix {prefix}_ \\
    --batch_size 1000 \\
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
    submit_bam_creation_jobs(dry_run=True)

if __name__ == "__main__":
    main()
