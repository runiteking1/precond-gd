#!/bin/bash
#SBATCH -J jaxjob      # Name the job
#SBATCH -p glinda
#SBATCH -N 2           # Take a single node
#SBATCH -t 24:00:00

module load python/3.11.6
source /home/abvoron/precond-gd/env/bin/activate


NUM_THREADS=4

export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_force_host_platform_device_count=${NUM_THREADS}"
export OPENBLAS_NUM_THREADS=${NUM_THREADS}
export MKL_NUM_THREADS=${NUM_THREADS}
export OMP_NUM_THREAD=${NUM_THREADS}

export JAX_TRACEBACK_FILTERING=off
export JAX_PLATFORMS=cpu


srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner smw

#srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner gn_exact --threshold 1e-5 
#srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner gn_exact --threshold 1e-4 
#srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner gn_exact --threshold 1e-3 
#srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner gn_exact --threshold 1e-2 
#srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner gn_exact --threshold 1e-1 
#srun -n 9 --cpus-per-task=${NUM_THREADS} --cpu-bind=cores python grokking.py --preconditioner gn_exact --threshold 1e-0 
