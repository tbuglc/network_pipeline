#!/bin/bash
#SBATCH --job-name=phat700
#SBATCH --output=report/%x-%j.out
#SBATCH --account=def-mlafond
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=20
#SBATCH --time=0-03:00:00           	# time limit (DD-HH:MM)
#SBATCH --mail-user=manuel.lafond@USherbrooke.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
# ---------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

module --force purge
module load CCEnv
module load StdEnv/2020 gcc/9.3.0
module load openmpi/4.0.3
module load boost/1.80.0
module list

#gcc --version
#mpirun --version
echo "Current working directory: $(pwd)"
echo "Starting run at: $(date)"

srun ./a.out -job_id $SLURM_JOBID -nodes $SLURM_NNODES -ntasks_per_node $SLURM_NTASKS_PER_NODE -ntasks_per_socket $SLURM_NTASKS_PER_SOCKET -cpus_per_task $SLURM_CPUS_PER_TASK -I input/p_hat700_1
# ---------------------------------------------------------------------
echo "Finishing run at: $(date)"
# ---------------------------------------------------------------------
