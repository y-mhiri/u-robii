#!/bin/sh
#SBATCH --job-name=learned-imager
#SBATCH --output=logs/log.out
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_long
#SBATCH --ntasks=7
#SBATCH --ntasks-per-node=1

module purge
module load anaconda3/2021.05/gcc-9.2.0

conda deactivate && conda activate myenv

cd /gpfs/users/mhiriy/Documents/EUSIPCO-23/unrolled/data/models/config

ROOT_PATH=/gpfs/users/mhiriy/Documents/EUSIPCO-23/
arr=(*.yaml)

for filename in "${arr[@]}" 
do
    name="${filename%.*}"
    echo $filename
    srun -N1 -n1 --output $ROOT_PATH/logs/$name.out train_model $filename &
done

wait