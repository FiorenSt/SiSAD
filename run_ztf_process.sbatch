#!/bin/bash
#SBATCH --job-name=ztf_process
#SBATCH --output=ztf_process_%A_%a.out
#SBATCH --error=ztf_process_%A_%a.err
#SBATCH --array=1-N  # Replace N with the total number of batch files
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Load necessary modules or activate environment
# module load python
# source activate myenv

# Set the paths
BATCH_DIR="/path/to/batches"  # Replace with the path where the batch files are saved
SCRIPT_PATH="/path/to/0.ALL_DOWNLOAD_TFRecords.py"  # Replace with the path to your main Python script
EXTRACT_TO="/path/to/extracted"  # Directory where tar.gz contents will be extracted
OUTPUT_FOLDER="/path/to/output"  # Directory where processed data will be saved

# Get the current batch file
BATCH_FILE=$(ls ${BATCH_DIR}/batch_*.txt | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Unique identifier for the batch
UNIQUE_ID=$(basename ${BATCH_FILE} .txt)

# Run the Python script with the batch file
python ${SCRIPT_PATH} --urls_file "${BATCH_FILE}" --extract_to "${EXTRACT_TO}" --output_folder "${OUTPUT_FOLDER}" --min_file_size 512 --batch_size 2048 --min_batch_size 10240 --unique_id ${UNIQUE_ID}
