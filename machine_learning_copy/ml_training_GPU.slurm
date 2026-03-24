#!/bin/bash
#SBATCH --job-name=ml_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1                   
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=botingl@princeton.edu
#SBATCH -A pppl

module purge
export PATH=/home/botingl/miniconda3/bin:$PATH
unset PYTHONPATH
unset PYTHONHOME

# Run the script using conda run
conda run -n GPU python /home/botingl/machine\ learning/ML_testing.py
