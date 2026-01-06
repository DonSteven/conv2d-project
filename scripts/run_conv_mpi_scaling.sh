#!/bin/bash
#
#SBATCH -J conv_mpi_scaling          # Job name
#SBATCH -p batch                     # Oscar CPU partition
#SBATCH -N 1                         # 1 node
#SBATCH -n 8                         # at most 8 MPI ranks
#SBATCH -t 00:10:00                  # at most longest 10 minutes
#SBATCH --mem=8G                     # Memory per node
#SBATCH -o logs/conv_mpi_%j.out      # standard output
#SBATCH -e logs/conv_mpi_%j.err      # standard error

module purge
module load hpcx-mpi

cd "$SLURM_SUBMIT_DIR"

echo "JobID: $SLURM_JOB_ID"
echo "Workdir: $(pwd)"
echo

mkdir -p logs

# fix the fixed problem size, test scaling
N=4096
K=7

for P in 1 2 4 8; do
    echo "============================"
    echo "  Running with $P MPI ranks"
    echo "============================"
    echo

    srun -n $P ./conv_mpi $N $K

    echo
done
