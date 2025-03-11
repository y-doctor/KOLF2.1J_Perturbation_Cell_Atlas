#!/usr/bin/env python3
"""
This module renames BAM files (and their index files) into batches and then writes
an sbatch script for each batch to run RNAEditingIndexer on those files.

For a single sample, RNAEditingIndexer is called as:
    /tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/RNAEditingIndexer/RNAEditingIndex \
      -d /tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/merged_bam_files \
      -f Aligned.sortedByCoord.NUMBER.bam. \
      -l /tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/AEI_logs \
      -os /tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/AEI_outputs/NUMBER/ \
      --genome hg38
"""

import os
import re
import subprocess
import argparse
import glob
import pandas as pd
def rename_bam_with_batch(base_dir: str, batch_size: int):
    """
    Rename all BAM and BAM index files in 'base_dir' into batches.
    Each BAM (and corresponding .bam.bai) file is renamed to include a suffix:
       Aligned.sortedByCoord.<batch>.bam
       Aligned.sortedByCoord.<batch>.bam.bai
    """
    # files = sorted(os.listdir(base_dir))
    # for i, file in enumerate(files):
    #     batch = i // batch_size
    #     # Avoid re-renaming if file already contains our batch suffix.
    #     if "BETA" in file:
    #         if file.endswith(".bam"):
    #             file_name = file.split(".")[0]
    #             suffix = f".Aligned.sortedByCoord.{batch}.bam"
    #             new_file_name = file_name + suffix
    #             os.rename(os.path.join(base_dir, file), os.path.join(base_dir, new_file_name))
    #             os.rename(os.path.join(base_dir, file + ".bai"), os.path.join(base_dir, new_file_name + ".bai"))
    #         if i % 100 == 0:
    #             print(f"Renamed {i}/{len(files)} files")

def get_unique_batches(base_dir: str):
    """
    Return sorted list of unique batch numbers parsed from renamed BAM filenames.
    It looks for files that end with 'Aligned.sortedByCoord.NUMBER.bam' or 'Aligned.sortedByCoord.NUMBER.bam.'.
    """
    pattern = re.compile(r'\.Aligned\.sortedByCoord\.(\d+)\.bam')
    batches = set()
    for file in os.listdir(base_dir):
        if "GAMMA" in file:
            match = pattern.search(file)
            if match:
                batch_num = int(match.group(1))
                batches.add(batch_num)
    return sorted(batches)

def write_sbatch_scripts(merged_bam_dir: str, rna_indexer_path: str, log_dir: str,
                         out_base: str, genome: str, batches: list,
                         sbatch_time: str, sbatch_mem: str, sbatch_cpus: str,
                         submit: bool):
    """
    For each batch, build an sbatch script file that calls RNAEditingIndexer with:
      - -d : merged_bam_dir
      - -f : Aligned.sortedByCoord.NUMBER.bam.
      - -l : log_dir
      - -os: out_base/NUMBER/
      --genome: genome

    The script file includes a header with SBATCH settings, plus:
      eval "$(conda shell.bash hook)"
      conda activate RNAEditingIndexer
    """
    scripts = []
    for batch in batches:
        # Construct the suffix that identifies this batch.
        suffix = f"Aligned.sortedByCoord.{batch}.bam"
        # Create the output directory for this batch if not exists.
        batch_out_dir = os.path.join(out_base, str(batch))
        if not os.path.exists(batch_out_dir):
            os.makedirs(batch_out_dir)
        # Build the RNAEditingIndexer command.
        run_command = (f"{rna_indexer_path} -d {merged_bam_dir} -f {suffix} "
                       f"-l {log_dir} -o {batch_out_dir} -os {batch_out_dir} --genome {genome}")
        # Build the sbatch header.
        header = f"""#!/bin/bash
#SBATCH --job-name=RNAEI_batch{batch}
#SBATCH --output={log_dir}/RNAEI_batch{batch}.out
#SBATCH --time={sbatch_time}
#SBATCH --mem={sbatch_mem}
#SBATCH --tasks-per-node={sbatch_cpus}
#SBATCH --nodes=1
#SBATCH --account=csd852
#SBATCH --partition=condo
#SBATCH --qos=condo
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ydoctor@ucsd.edu

eval "$(conda shell.bash hook)"
conda activate RNAEditingIndexer
"""
        # Combine header and command.
        script_content = header + "\n" + run_command + "\n"
        # Determine a path to write the script.
        script_filename = os.path.join(log_dir, f"RNAEI_batch{batch}.sh")
        with open(script_filename, "w") as script_file:
            script_file.write(script_content)
        os.chmod(script_filename, 0o755)
        scripts.append(script_filename)
        print(f"Created sbatch script: {script_filename}")
    
    if submit:
        for script in scripts:
            print(f"Submitting {script} ...")
            subprocess.run(["sbatch", script])
    else:
        print("All sbatch scripts created. They have not been submitted automatically.")
    return scripts

def main():
    parser = argparse.ArgumentParser(
        description="Rename BAM files into batches and write sbatch scripts for RNAEditingIndexer jobs for each batch."
    )
    parser.add_argument("-m", "--merged_dir", type=str,
                        default="/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/merged_bam_files",
                        help="Directory with merged BAM files")
    parser.add_argument("-r", "--rna_indexer", type=str,
                        default="/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/RNAEditingIndexer/RNAEditingIndex",
                        help="Path to RNAEditingIndexer executable")
    parser.add_argument("-l", "--log_dir", type=str,
                        default="/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/AEI_logs",
                        help="Directory for RNAEditingIndexer logs and sbatch scripts")
    parser.add_argument("-os", "--out_base", type=str,
                        default="/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/AEI_outputs_BETA",
                        help="Base directory for RNAEditingIndexer outputs; each batch creates a subdirectory")
    parser.add_argument("-b", "--batch_size", type=int,
                        default=200,
                        help="Batch size (number of samples per batch)")
    parser.add_argument("-g", "--genome", type=str,
                        default="hg38",
                        help="Reference genome (e.g., hg38)")
    parser.add_argument("--sbatch_time", type=str, default="24:00:00",
                        help="Time limit for each sbatch job")
    parser.add_argument("--sbatch_mem", type=str, default="64G",
                        help="Memory allocation for each sbatch job")
    parser.add_argument("--sbatch_cpus", type=str, default="10",
                        help="Number of CPUs per task for each sbatch job")
    parser.add_argument("--submit", action="store_true",
                        help="Actually submit the sbatch scripts after writing them")
    
    args = parser.parse_args()

    merged_dir = args.merged_dir
    rna_indexer = args.rna_indexer
    log_dir = args.log_dir
    out_base = args.out_base
    batch_size = args.batch_size
    genome = args.genome
    sbatch_time = args.sbatch_time
    sbatch_mem = args.sbatch_mem
    sbatch_cpus = args.sbatch_cpus
    submit_flag = args.submit

    # Create log and output directories if they don't exist.
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(out_base):
        os.makedirs(out_base)
    
    print(f"Renaming BAM files in {merged_dir} with batch size {batch_size} ...")
    rename_bam_with_batch(merged_dir, batch_size)

    print("Identifying unique batches from renamed files ...")
    batches = get_unique_batches(merged_dir)
    print(f"Found batches: {batches}")
     # Create a subdirectory for each batch
    for batch in batches:
        batch_dir = os.path.join(out_base, str(batch))
        if not os.path.exists(batch_dir):
            os.makedirs(batch_dir)

    print("Writing sbatch scripts for each batch ...")
    write_sbatch_scripts(merged_dir, rna_indexer, log_dir, out_base, genome, batches,
                         sbatch_time, sbatch_mem, sbatch_cpus, submit_flag)

    print("Done.")

def merge_csv_files(base_directory, output_file):
    """
    Search through the subdirectories of 'base_directory' for CSV files,
    merge them into a single CSV file with the same header, and write the result to 'output_file'.

    Parameters:
        base_directory (str): The directory containing multiple subdirectories, each with one CSV file.
        output_file (str): The output CSV file path.
    """
    # Build search pattern: look for *.csv in all first-level subdirectories
    pattern = os.path.join(base_directory, "*", "*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print("No CSV files found in any subdirectory of", base_directory)
        return

    # For logging purposes, list found files
    print("Found CSV files:")
    for file in csv_files:
        print(f" - {file}")

    # Read and store each CSV into a list of DataFrames
    df_list = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not df_list:
        print("No CSV files could be read successfully.")
        return

    # Concatenate all DataFrames into a single DataFrame
    merged_df = pd.concat(df_list, ignore_index=True)

    # Write the merged DataFrame to the output file
    try:
        merged_df.to_csv(output_file, index=False)
        print(f"Merged CSV saved to: {output_file}")
    except Exception as e:
        print(f"Error writing merged CSV to {output_file}: {e}")

if __name__ == '__main__':
    main()