#!/bin/bash
export WORKING_DIR="/home/files0/warf949/caffe5/caffe-intel"
export RUN_DIR="$WORKING_DIR/runs_puma/Alexnet_Quicksolver_Batchsize_256/Layerwise_current_different/16_models"
cd ${WORKING_DIR}
source ./env_caffe.sh
#SBATCH --nodes=4     
#SBATCH --ntasks-per-node=4
#SBATCH --time=4:00:00
#SBATCH --job-name "alexnet-quicksolver-layerwise_current_rotate-16models-5cores-batch256"
#SBATCH -o stdout.alexnet.quicksolver.layerwise_current_rotate.16models.5cores.batch256.txt
#SBATCH -e stderr.alexnet.quicksolver.layerwise_current_rotate.16models.5cores.batch256.txt
#SBATCH -p all
mpiexec \
	-genv MV2_VBUF_TOTAL_SIZE=65536 \
        -genv MV2_IBA_EAGER_THRESHOLD=8192 \
        -genv OMP_NUM_THREADS=5 \
        -genv MKL_NUM_THREADS=5 \
        --wdir ${WORKING_DIR}  \
        ./build/tools/caffe train --solver=${RUN_DIR}/quicksolver_batch256.prototxt \
	--par MPI_Layerwise_Rotate_CPU 

