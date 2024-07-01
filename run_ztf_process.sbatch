#!/bin/bash
#SBATCH --job-name=ztf_process
#SBATCH --output=ztf_process_%A_%a.out
#SBATCH --error=ztf_process_%A_%a.err
#SBATCH -p RM-shared
#SBATCH --array=1-3  # Replace N with the total number of batch files
#SBATCH --nodes=1
##SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --mem=8G
#SBATCH --time=12:00:00

# Load necessary modules or activate environment
# module load python
# source activate myenv

# Set the paths
BATCH_DIR="/path/to/batches"  # Replace with the path where the batch files are saved
SCRIPT_PATH="/path/to/0.SIMPLE_ALL_DOWNLOAD.py"  # Replace with the path to your main Python script
BASE_EXTRACT_TO="/path/to/extracted"  # Base directory where tar.gz contents will be extracted
OUTPUT_FOLDER="/path/to/output"  # Base directory where processed data will be saved

# Get the current batch file
BATCH_FILE=$(ls ${BATCH_DIR}/batch_*.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Unique identifier for the batch
UNIQUE_ID=$(basename ${BATCH_FILE} .txt)

# Append unique identifier to paths
EXTRACT_TO="${BASE_EXTRACT_TO}_${UNIQUE_ID}"

# Run the Python script with the batch file
python ${SCRIPT_PATH} --urls_file "${BATCH_FILE}" --extract_to "${EXTRACT_TO}" --output_folder "${OUTPUT_FOLDER}" --min_file_size 512 --batch_size 2048 --unique_id ${UNIQUE_ID}"
