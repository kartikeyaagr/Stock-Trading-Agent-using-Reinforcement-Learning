#! /bin/bash
#PBS -N DQN_Train
#PBS -o output/out.log
#PBS -e output/err.log
#PBS -l select=1:ncpus=24
#PBS -l select=1::ngpus=2
#PBS -l place=scatter
#PBS -q gpu
#PBS -P Rainbow_DQN

module purge
module load compiler/anaconda3
module load compiler/cuda-10.2
cd /home/kartikeya.agrawal_ug25/RL_Final
conda run -n RL python RL.py