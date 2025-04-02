#!/usr/bin/env bash
#SBATCH -p ei-gpu
#SBATCH --job-name=all_1
#SBATCH --output=all_1_%j.out
#SBATCH --error=all_1_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # notifications for job done & fail
#SBATCH --mail-user=her24bip@nbi.ac.uk # send-to address

# We'll pass it as an environment variable or argument.
export THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Now launch your script with srun.
srun --gres=gpu:1 --cpus-per-task=20 --mem=128G \
  ./nn_many_layers_cv.sh

