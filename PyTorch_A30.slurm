#!/bin/bash
##NECESSARY JOB SPECIFICATIONS

#SBATCH --job-name=tf_MNIST
#SBATCH --time=01:00:00                  
#SBATCH --nodes=1                  
#SBATCH --output=PyTorch_MNIST.slurm_run.%j
#SBATCH --mem=480GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1 
##SBATCH --nodelist=lg01
##SBATCH --exclusive
##SBATCH --reservation=training



# Load required modules 
module load WebProxy 
module load GCCcore/12.2.0  Python/3.10.8 
module load GCC/12.3.0  OpenMPI/4.1.5
module load PyTorch/2.1.2-CUDA-12.1.1


# Activate virtual environment
python -m venv PyTorchenv # Create virtual env
source cvenv2/bin/activate


# Upgrade pip and install dependencies
pip install --upgrade pip
pip install torch==2.1.2 
pip install torchvision==0.15.1

# Verify module loading
module list


cd $SCRATCH/A30
python PyTorch_MNIST.py
