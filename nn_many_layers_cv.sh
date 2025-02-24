#!/usr/bin/env bash

# Define log directory and ensure it exists
LOG_DIR="logs"  # Replace with your desired log directory
mkdir -p "$LOG_DIR"

# Set up log file with timestamp
LOG_FILE="$LOG_DIR/log_$(date +'%Y%m%d_%H%M%S').log"

# Function to log messages with a timestamp
log_with_timestamp() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Start logging
log_with_timestamp "Starting script execution."

# Set environment variables
log_with_timestamp "Setting MAMBA_ROOT_PREFIX."
export MAMBA_ROOT_PREFIX="/hpc-home/her24bip/.local/share/mamba"

# Define the path to the mamba executable for convenience
MAMBA_EXEC="/hpc-home/her24bip/miniconda3/condabin/mamba"

# 1. Run coptr index
log_with_timestamp "Running many layers with cv"
$MAMBA_EXEC run -n pytorch python 4b_nn_many_layers_cross_small.py | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
  log_with_timestamp "Error: many layers with cv command failed."
  exit 1
fi

# Completion message
log_with_timestamp "Script execution completed successfully."

exit 0
