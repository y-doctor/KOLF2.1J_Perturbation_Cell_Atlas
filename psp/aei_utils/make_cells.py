import pandas as pd
import os
import textwrap
import subprocess
def prepare_cells(gene_target: str, knockdown_filtered_cells_dataframe: pd.DataFrame, output_dir: str = None, batch: str = None):
    """
    Create CSV files containing cells filtered by gene target (and optional batch),
    grouped by batch, gRNA, and channel.

    Parameters:
    - gene_target (str): The gene target to filter cells by.
    - knockdown_filtered_cells_dataframe (pd.DataFrame): DataFrame containing cell data
      with pre-parsed 'batch', 'gRNA', and 'channel' columns.
    - output_dir (str, optional): The directory to save the output CSV files.
      If None, the current working directory is used.
    - batch (str, optional): The specific batch to filter cells by. If None, all batches are included.

    The function filters the data based on the provided gene target (and optional batch),
    groups the cells by batch, gRNA, and channel, and saves the grouped data to CSV files at:
        {output_dir}/{gene_target}/{batch}/{gRNA}/{channel}/cells.csv
    """
    cells_and_batches = knockdown_filtered_cells_dataframe

    # Filter by batch if specified
    if batch is not None:
        cells_and_batches = cells_and_batches[cells_and_batches['batch'].str.contains(batch, na=False)]
        
    # Filter by gene target
    cells_and_batches = cells_and_batches[cells_and_batches['gene_target'] == gene_target]

    # Ensure the output directory exists
    base_path = f"{output_dir}/{gene_target}" if output_dir is not None else gene_target
    os.makedirs(base_path, exist_ok=True)

    write_cell_per_channel(cells_and_batches, base_path)
    make_sbatch_files(base_path)

def write_cell_per_channel(cells_and_batches: pd.DataFrame, write_path: str):
    """
    Write the cells to a new CSV file for each combination of batch, gRNA, and channel.
    The file is saved at: {write_path}/{batch}/{gRNA}/{channel}/cells.csv.
    """
    for (batch, gRNA, channel), data in cells_and_batches.groupby(['batch', 'gRNA', 'channel']):
        # Get the list of cell barcodes
        cells = [cell.rsplit('-', 1)[0] + '-1' for cell in data['cell_barcode'].tolist()]
        
        # Create the output directory for this batch, gRNA, and channel
        dir_path = os.path.join(write_path, batch, gRNA, channel)
        os.makedirs(dir_path, exist_ok=True)
        
        # Write the cells to a CSV file in this directory.
        with open(os.path.join(dir_path, "cells.csv"), "w") as f:
            for cell in cells:
                f.write(f"{cell}\n")

def make_sbatch_files(path: str):
    """
    Create sbatch files for each channel in the given path.
    The directory structure is expected to be: path/batch/gRNA/channel
    """
    for batch in os.listdir(path):
        batch_dir = os.path.join(path, batch)
        if not os.path.isdir(batch_dir):
            continue
        for gRNA in os.listdir(batch_dir):
            gRNA_dir = os.path.join(batch_dir, gRNA)
            if not os.path.isdir(gRNA_dir):
                continue
            for channel in os.listdir(gRNA_dir):
                channel_dir = os.path.join(gRNA_dir, channel)
                if not os.path.isdir(channel_dir):
                    continue
                sbatch_file_path = os.path.join(channel_dir, "alu_editing_sbatch.sh")
                script = textwrap.dedent(f'''\
                    #!/bin/bash
                    #SBATCH --job-name=ALU
                    #SBATCH --partition=condo
                    #SBATCH --qos=condo
                    #SBATCH --nodes=1
                    #SBATCH --tasks-per-node=1
                    #SBATCH --time=6:00:00
                    #SBATCH --mem=8
                    #SBATCH --account=csd852
                    #SBATCH -o {os.path.join(channel_dir, "slurm_outputs.txt")}
                    #SBATCH --mail-type=FAIL
                    #SBATCH --mail-user=ydoctor@ucsd.edu
                    # Load any necessary modules
                    module load shared
                    module load cpu/0.17.3
                    module load gcc/10.2.0-2ml3m2l
                    module load samtools/1.13-fd7mbdu
                    export PATH="/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/subset-bam_linux:${{PATH}}"

                    # Define parameters (remove spaces around =)
                    BAM_PATH=/tscc/projects/ps-malilab/ydoctor/iPSC_Pan_Genome/Pan_Genome/cellranger_files/ALU_Editing/{batch}/{channel}/cellranger_outputs/outs/possorted_genome_bam.bam
                    CELLS_PATH={os.path.join(channel_dir, "cells.csv")}
                    OUTPUT_BAM={os.path.join(channel_dir, "alu_editing.bam")}
                    # Your commands to run on the node go here
                    /tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/subset-bam_linux --bam $BAM_PATH --cell-barcodes $CELLS_PATH --out-bam $OUTPUT_BAM
                ''')
                with open(sbatch_file_path, "w") as f:
                    f.write(script)

def submit_bam_subset_jobs(base_dir: str):
    """
    Traverse the base directory and its subdirectories. For each directory that contains an
    'alu_editing_sbatch.sh' file and follows the expected structure:
         {output_dir}/{gene_target}/{batch}/{gRNA}/{channel},
    check if an 'alu_editing.bam' file exists in the same directory. If the BAM file does not exist,
    change into that directory and submit the sbatch job.
    
    Parameters:
    - base_dir: The top-level directory to search (typically {output_dir}/{gene_target}).
    """
    for dirpath, dirnames, filenames in os.walk(base_dir):
        if "alu_editing_sbatch.sh" in filenames:
            # Skip this directory if alu_editing.bam already exists.
            if "alu_editing.bam" in filenames:
                print(f"Skipping {dirpath}: alu_editing.bam already exists.")
            else:
                print(f"Submitting job in {dirpath}...")
                try:
                    result = subprocess.run(
                        ["sbatch", "alu_editing_sbatch.sh"],
                        cwd=dirpath,   # Run command in the directory containing the sbatch script.
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    print(f"Submitted job in {dirpath}: {result.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    error_details = e.stderr.strip() if e.stderr else str(e)
                    print(f"Error submitting job in {dirpath}: {error_details}")


def merge_bam_files(gene_target: str, output_dir: str = None):
    """
    For the given gene target folder, traverse each batch and gRNA subdirectory,
    and create an sbatch script to merge, sort, and index all the channel-level
    'alu_editing.bam' files. The merged BAM file is produced at the gRNA level.

    Directory structure:
        {output_dir}/{gene_target}/{batch}/{gRNA}/{channel}

    For each gRNA directory, the merged BAM files will be:
        merged_{gRNA}_{gene_target}_alu_editing.bam
    And the sorted (and indexed) BAM file will be:
        merged_{gRNA}_{gene_target}_alu_editing.sorted.bam

    Parameters:
    - gene_target (str): The gene target folder name (e.g., "ALU").
    - output_dir (str, optional): The parent directory where the gene_target folder is located.
      If None, the current working directory is used.
    """
    import os
    import textwrap
    import subprocess

    # Determine the base path for the gene target folder.
    base_path = os.path.join(output_dir, gene_target) if output_dir is not None else gene_target

    if not os.path.exists(base_path):
        print(f"Base path {base_path} does not exist.")
        return

    # Iterate over each batch directory within the gene target folder.
    for batch in os.listdir(base_path):
        batch_dir = os.path.join(base_path, batch)
        if not os.path.isdir(batch_dir):
            continue

        # Iterate over each gRNA directory within the batch.
        for gRNA in os.listdir(batch_dir):
            gRNA_dir = os.path.join(batch_dir, gRNA)
            if not os.path.isdir(gRNA_dir):
                continue

            sbatch_file = os.path.join(gRNA_dir, "merge_index_bams.sbatch")

            # Create an sbatch script that will:
            # 1. Merge all 'alu_editing.bam' files found under the gRNA directory (from channel subdirectories),
            # 2. Sort the merged BAM file,
            # 3. Index the sorted BAM file.
            script = textwrap.dedent(f'''\
                #!/bin/bash
                #SBATCH --job-name=MERGE_{gRNA}
                #SBATCH --partition=condo
                #SBATCH --qos=condo
                #SBATCH --nodes=1
                #SBATCH --tasks-per-node=8
                #SBATCH --time=12:00:00
                #SBATCH --mem=64
                #SBATCH --account=csd852
                #SBATCH -o {os.path.join(gRNA_dir, "merge_index_slurm_outputs.txt")}
                #SBATCH --mail-type=FAIL
                #SBATCH --mail-user=ydoctor@ucsd.edu

                module load shared
                module load cpu/0.17.3
                module load gcc/10.2.0-2ml3m2l
                module load samtools/1.13-fd7mbdu

                MERGED_BAM=merged_{gRNA}_{gene_target}_alu_editing.bam
                SORTED_BAM=merged_{gRNA}_{gene_target}_alu_editing.sorted.bam

                echo "Merging BAM files for gRNA {gRNA} in batch {batch}..."
                samtools merge $MERGED_BAM $(find . -type f -name "alu_editing.bam")

                echo "Sorting merged BAM file..."
                samtools sort -o $SORTED_BAM $MERGED_BAM

                echo "Indexing sorted BAM file..."
                samtools index $SORTED_BAM

                echo "Merge, sort, and index job completed for gRNA {gRNA} in batch {batch}."
            ''')

            # Write the script to file.
            with open(sbatch_file, "w") as f:
                f.write(script)

            # Submit the sbatch job from the gRNA directory.
            try:
                result = subprocess.run(
                    ["sbatch", "merge_index_bams.sbatch"],
                    cwd=gRNA_dir,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"Submitted merge job in {gRNA_dir}: {result.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                error_details = e.stderr.strip() if e.stderr else str(e)
                print(f"Error submitting merge job in {gRNA_dir}: {error_details}")

