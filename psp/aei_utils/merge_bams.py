#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
import textwrap

def group_bam_files(input_dir: str):
    """
    Group BAM files by (BATCH, GENE_NUMBER) based on filename format: 
    BATCH_CHANNEL_GENE_NUMBER.bam
    """
    groups = {}
    for entry in os.listdir(input_dir):
        if entry.endswith(".bam"):
            base = entry[:-4]  # remove ".bam"
            parts = base.split("_")
            if len(parts) != 4:
                print(f"Warning: Filename '{entry}' does not match expected format (BATCH_CHANNEL_GENE_NUMBER.bam). Skipping.")
                continue
            batch, channel, gene, guide = parts
            key = (batch, f"{gene}_{guide}")
            file_path = os.path.join(input_dir, entry)
            groups.setdefault(key, []).append(file_path)
    return groups

def generate_merge_command(batch: str, gene: str, file_list, output_dir: str) -> str:
    """
    Generate the bash commands (as a multiline string) to merge a group of BAM files using samtools merge,
    then sort the merged BAM file, index it, and finally remove the unsorted file.

    Parameters:
    -----------
    batch: str
        The batch identifier.
    gene: str
        The gene (or gene_guide) identifier.
    file_list: list
        List of BAM file paths to merge.
    output_dir: str
        Directory where output files will be written.

    Returns:
    --------
    str
        Multiline bash command string.
    """
    merged_unsorted = os.path.join(output_dir, f"{batch}_{gene}_unsorted.bam")
    sorted_file = os.path.join(output_dir, f"{batch}_{gene}.bam")
    
    # Merge the BAM files into an unsorted file.
    merge_cmd = f"samtools merge -f -o {merged_unsorted} " + " ".join(file_list)
    
    # Sort the merged file.
    sort_cmd = f"samtools sort -o {sorted_file} {merged_unsorted}"
    
    # Index the sorted file.
    index_cmd = f"samtools index {sorted_file}"
    
    # Remove the intermediate unsorted merge file.
    remove_cmd = f"rm {merged_unsorted}"
    
    # Echo command to log which group is being processed.
    echo_cmd = f'echo "$(date) Processing batch {batch}, gene {gene}"'
    
    full_cmd = "\n".join([echo_cmd, merge_cmd, sort_cmd, index_cmd, remove_cmd])
    return full_cmd


def create_sbatch_script(commands: list, job_index: int, output_dir: str, sbatch_time: str, sbatch_mem: str, sbatch_cpus: str) -> str:
    """
    Create an sbatch script file to run a chunk of merge commands.
    
    Modifications:
      - Adds module loads for cpu, bamtools2, and samtools.
      - Prints out all the BATCH_GENE groups in that job's chunk to the output file before running the merge commands.
    
    Parameters:
      commands (list): List of multi-line merge command strings.
      job_index (int): Index number for this sbatch chunk.
      output_dir (str): Directory where the sbatch script (and logs) will be written.
      sbatch_time (str): Time limit for this sbatch job.
      sbatch_mem (str): Memory allocation for this sbatch job.
      sbatch_cpus (str): Number of CPUs per task.
    
    Returns:
      str: Path to the created sbatch script file.
    """
    script_filename = os.path.join(output_dir, f"merge_bams_chunk_{job_index}.sh")
    sbatch_header = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name=merge_bams_chunk_{job_index}
        #SBATCH --output={output_dir}/merge_bams_chunk_{job_index}.out
        #SBATCH --time={sbatch_time}
        #SBATCH --mem={sbatch_mem}
        #SBATCH --tasks-per-node={sbatch_cpus}
        #SBATCH --nodes=1
        #SBATCH --account=csd852
        #SBATCH --partition=condo
        #SBATCH --qos=condo
        #SBATCH --mail-type=FAIL
        #SBATCH --mail-user=ydoctor@ucsd.edu
        """)
    
    # Add module load commands.
    module_commands = "\nmodule load cpu\nmodule load samtools\n"
    
    # Extract group names (BATCH_GENE) from the first line of each merge command.
    group_names = []
    for cmd in commands:
        first_line = cmd.splitlines()[0].strip()
        # Expected format: echo "Processing batch {batch}, gene {gene}"
        if first_line.startswith('echo "'):
            # Remove leading 'echo "' and trailing '"' to get the inner message.
            line_content = first_line[len('echo "'):].rstrip('"')
            prefix = "Processing batch "
            if line_content.startswith(prefix):
                remainder = line_content[len(prefix):].strip()  # e.g. "ALPHA, gene CBX8"
                parts = remainder.split(", gene ")
                if len(parts) == 2:
                    group_names.append(f"{parts[0].strip()}_{parts[1].strip()}")
                else:
                    group_names.append(line_content)
            else:
                group_names.append(line_content)
        else:
            group_names.append(first_line)
    
    # Build the block that echoes out the group names.
    group_echo_commands = 'echo "This job will process the following groups:"'
    for name in group_names:
        batch_gene = name.split(", gene ")
        if len(batch_gene) == 2:
            group_name = f"{batch_gene[0].replace('Processing batch ', '').strip()}_{batch_gene[1].strip()}"
            group_echo_commands += f'\necho "{group_name}"'
    
    # Combine header, module loads, set -e, group echo block, and merge commands.
    script_content = (
        sbatch_header + "\n" +
        module_commands + "\n" +
        "set -e\n\n" +
        group_echo_commands + "\n\n" +
        "\n\n".join(commands) + "\n"
    )
    
    with open(script_filename, "w") as f:
        f.write(script_content)
    
    os.chmod(script_filename, 0o755)
    return script_filename



def split_into_chunks(lst: list, chunk_size: int):
    """
    Yield successive chunk-sized sublists from list lst.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

def main():
    parser = argparse.ArgumentParser(
        description="Prepare and submit sbatch jobs to merge BAM files grouped by BATCH and GENE_NUMBER."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing input .bam files (named as BATCH_CHANNEL_GENE_NUMBER.bam).")
    parser.add_argument("--output_dir", default="/tscc/projects/ps-malilab/ydoctor/KOLF_Perturbation_Atlas/alu_editing/merged_bam_files",
                        help="Directory to store merged and sorted BAM files along with sbatch logs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Number of merge tasks per sbatch submission.")
    parser.add_argument("--sbatch_time", default="12:00:00", help="Time limit for each sbatch job.")
    parser.add_argument("--sbatch_mem", default="8G", help="Memory allocation for each sbatch job.")
    parser.add_argument("--sbatch_cpus", default="1", help="Number of CPUs per task for each sbatch job.")
    parser.add_argument("--submit", action="store_true", help="If provided, the script will submit the sbatch jobs after creation.")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group the BAM files based on their BATCH and GENE_NUMBER fields.
    groups = group_bam_files(input_dir)
    if not groups:
        print("No valid BAM files found in the input directory. Exiting.")
        sys.exit(1)
    
    # For each group create the corresponding commands.
    merge_commands = []
    for (batch, gene), file_list in groups.items():
        cmd = generate_merge_command(batch, gene, file_list, output_dir)
        merge_commands.append(cmd)
    
    # Split all merge commands into chunks for separate sbatch jobs.
    chunks = list(split_into_chunks(merge_commands, batch_size))
    total_tasks = len(merge_commands)
    total_jobs = len(chunks)
    print(f"Total merge tasks (groups): {total_tasks}")
    print(f"Creating {total_jobs} sbatch script(s), each handling up to {batch_size} tasks.")
    
    # Now write an sbatch script for each chunk and (optionally) submit it.
    for i, chunk in enumerate(chunks, start=1):
        script_file = create_sbatch_script(chunk, i, output_dir, args.sbatch_time, args.sbatch_mem, args.sbatch_cpus)
        print(f"Created sbatch script: {script_file}")
        if args.submit:
            print(f"Submitting job chunk {i}...")
            subprocess.run(["sbatch", script_file])
            
if __name__ == "__main__":
    main()