#!/bin/sh
### -- Set the job name --
#BSUB -J sweep
### -- Specify the queue --
#BSUB -q gpua10
### -- Ask for number of cores (at least 4 for GPU) --
#BSUB -n 4
### -- Request GPU (exclusive use of 1 GPU) --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- Specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- Request memory per core --
#BSUB -R "rusage[mem=4GB]"
### -- Set per-core memory limit (max 1GB above requested) --
#BSUB -M 4GB
### -- Set walltime limit: hh:mm --
#BSUB -W 20:00
### -- Specify the output and error files --
#BSUB -o Output_%J.out
#BSUB -e Error_%J.err

# Activate your virtual environment (update with your own vvirtual env path)
cd 02456-Group39 # relative to where you are running the script from
python3 testrun.py