#!/bin/sh
#SBATCH --job-name=benchmark
#SBATCH --output=log.out
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=cpu_long
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1


module purge
module load anaconda3/2021.05/gcc-9.2.0

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/gpfs/softs/spack/opt/spack/linux-centos7-cascadelake/gcc-9.2.0/anaconda3-2021.05-iqwuixltaz4o4tspbuo2fgpqpsdsj74q/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/gpfs/softs/spack/opt/spack/linux-centos7-cascadelake/gcc-9.2.0/anaconda3-2021.05-iqwuixltaz4o4tspbuo2fgpqpsdsj74q/etc/profile.d/conda.sh" ]; then
        . "/gpfs/softs/spack/opt/spack/linux-centos7-cascadelake/gcc-9.2.0/anaconda3-2021.05-iqwuixltaz4o4tspbuo2fgpqpsdsj74q/etc/profile.d/conda.sh"
    else
        export PATH="/gpfs/softs/spack/opt/spack/linux-centos7-cascadelake/gcc-9.2.0/anaconda3-2021.05-iqwuixltaz4o4tspbuo2fgpqpsdsj74q/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda deactivate && conda activate myenv


DSETPATH="/gpfs/users/mhiriy/Documents/u-robii/data/datasets/student_test.zip"
MODELPATH="/gpfs/users/mhiriy/Documents/u-robii/data/models/student/student-0.0001.pth"
OUT="/gpfs/users/mhiriy/Documents/u-robii/benchmark/student-benchmark"

srun -N1 -n1 python plot_results.py $DSETPATH $MODELPATH $OUT &

wait
