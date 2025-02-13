#!/usr/bin/env python3
import os
import subprocess
import sys
import argparse
import shutil

def find_bam_files_in_batch(batch_dir: str):
    """
    Find all 'alu_editing.bam' files within each channel directory inside a batch directory.
    
    Parameters:
    -----------
    batch_dir : str
        Path to the batch directory.
    
    Returns:
    --------
    List[str]
        Full paths to each found 'alu_editing.bam' file.
    """
    bam_files = []
    # Loop over each channel directory inside the batch directory.
    for channel in sorted(os.listdir(batch_dir)):
        channel_dir = os.path.join(batch_dir, channel)
        if os.path.isdir(channel_dir):
            bam_path = os.path.join(channel_dir, "alu_editing.bam")
            if os.path.exists(bam_path):
                bam_files.append(bam_path)
                print(f"Found: {bam_path}")
            else:
                print(f"Warning: {bam_path} does not exist.")
    return bam_files

def merge_bams(bam_files, output_file):
    """
    Merge BAM files using 'samtools merge' and output the merged file.
    
    Parameters:
    -----------
    bam_files : List[str]
        List of full paths to BAM files.
    output_file : str
        Path to the output merged BAM file.
    """
    # Optionally update your PATH with the samtools directory.
    samtools_dir = "/tscc/local/apps/samtools/1.13-fd7mbdu/bin"
    os.environ["PATH"] += os.pathsep + samtools_dir

    samtools_executable = "samtools"
    if not shutil.which(samtools_executable):
        print("samtools not found in PATH even after update, using full path instead.")
        samtools_executable = os.path.join(samtools_dir, "samtools")

    # Construct and run the merge command.
    cmd = [samtools_executable, "merge", "-f", output_file] + bam_files
    print("Running command:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print(f"Merge completed successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print("Error merging BAM files:")
        print(e)
        sys.exit(1)

def merge_bams_per_batch(base_dir: str, perturbation: str):
    """
    For each batch directory under base_dir, merge the individual channel BAM files.
    
    Parameters:
    -----------
    base_dir : str
        The path to the perturbation directory containing batch directories.
    perturbation : str
        The perturbation name.
    """
    for batch in sorted(os.listdir(base_dir)):
        batch_dir = os.path.join(base_dir, batch)
        if os.path.isdir(batch_dir):
            print(f"\nProcessing batch directory: {batch_dir}")
            bam_files = find_bam_files_in_batch(batch_dir)
            if bam_files:
                # Create an output file in the batch directory.
                output_file = os.path.join(batch_dir, f"merged_{batch}_{perturbation}_alu_editing.bam")
                print(f"Merging BAM files into: {output_file}")
                merge_bams(bam_files, output_file)
            else:
                print(f"No BAM files found in {batch_dir}, skipping.")

def main():
    parser = argparse.ArgumentParser(
        description="Merge alu_editing.bam files per batch for a given perturbation."
    )
    parser.add_argument(
        "perturbation",
        help="Name of the perturbation (e.g., ADAR) to process."
    )
    args = parser.parse_args()
    perturbation = args.perturbation
    
    # The base directory now includes the perturbation.
    base_dir = f"/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/{perturbation}"
    if not os.path.isdir(base_dir):
        print(f"Base directory {base_dir} does not exist.")
        sys.exit(1)
    
    merge_bams_per_batch(base_dir, perturbation)

if __name__ == "__main__":
    main()