#!/bin/bash
#SBATCH --job-name=net-metrics
#SBATCH --account=def-mlafond
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00           	# time limit (HH:MM:SS)
#SBATCH --mail-user=manueul.lafond@usherbrooke.ca,bugl2301@Usherbrooke.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
# ---------------------------------------------------------------------

module load python/3.10

source venv/bin/activate

python metric_calculator.py -i /home/bugl2301/projects/def-mlafond/bugl2301/data

# ---------------------------------------------------------------------
echo "Finishing run at: $(date)"
# ---------------------------------------------------------------------


