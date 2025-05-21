#!/bin/bash
#SBATCH -J sgd      # Name the job
#SBATCH -p glinda
#SBATCH -N 1          # Take a single node
#SBATCH -t 24:00:00
#SBATCH --exclude=gn[1-62]

N_PROCESSES=1

module load python/3.13.0 cuda/12.6.2 cudnn/9.2.0.82-12 openmpi/5.0.5
source /home/abvoron/precond-gd/env/bin/activate

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDNN_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH

echo "Check GPU availability"
python -c "import jax;print(jax.devices())"
echo "======================"


export XLA_FLAGS="--xla_gpu_cuda_data_dir=${CUDA_HOME}"
export JAX_TRACEBACK_FILTERING=off

#python point_test.py --training sgd
#python point_test.py --training smw
python point_test.py --training gn
