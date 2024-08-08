#!/bin/bash
##NECESSARY JOB SPECIFICATIONS

#SBATCH --job-name=TF_MNIST
#SBATCH --time=01:00:00                  
#SBATCH --nodes=1                  
#SBATCH --output=TF_MNIST.slurm_run.%j
#SBATCH --mem=480GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a30:1 
##SBATCH --nodelist=lg01
##SBATCH --exclusive
##SBATCH --reservation=training


# Load required modules in the correct order
module load WebProxy
#module load GCCcore/11.3.0  Python/3.10.8 
module load GCCcore/12.2.0  Python/3.10.8 
module load GCC/12.2.0 OpenMPI/4.1.4 CUDA/11.7.0 TensorFlow/2.13.0   



# Verify module loading
module list

# Check Python version
#python_version=$(python --version 2>&1)
#echo "Python Version: $python_version"


# Activate the Python virtual environment
python -m venv cvenv # Create virtual env
source cvenv/bin/activate



# Upgrade pip and resolve TensorFlow dependencies
pip install --upgrade pip
pip install wheel==0.36.2 six==1.16.0 requests==2.25.1 packaging==20.9 typing-extensions==3.7.4.3 urllib3==1.26.5


cd $SCRATCH/A30
python TF_MNIST.py
