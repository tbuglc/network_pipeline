#!/bin/bash
#SBATCH --job-name=net-metrics
#SBATCH --account=def-mlafond
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=3:00:00
#SBATCH --mail-user=bugl2301@Usherbrooke.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE


echo "Start running at: \$(date)"

srun tar cf - data/chunk_1 | pigz -p 32 > data/chunk_1.tar.gzx

echo "Finishing run at: \$(date)"
