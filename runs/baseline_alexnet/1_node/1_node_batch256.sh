#!/bin/bash
export WORKING_DIR="/home/files0/warf949/caffe-cascade1/caffe-intel"
cd ${WORKING_DIR}
source ./env_caffe.sh
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1     
#SBATCH --time=12:00:00
#SBATCH --job-name "alexnet-1node-batch256"
#SBATCH -o stdout.alexnet.1node.batch256.txt
#SBATCH -e stderr.alexnet.1node.batch256.txt
#SBATCH -p all
mpiexec \
	-genv MV2_VBUF_TOTAL_SIZE=6144 \
        -genv MV2_IBA_EAGER_THRESHOLD=6144 \
        -genv OMP_NUM_THREADS=16 \
        -genv MKL_NUM_THREADS=16 \
        --wdir ${WORKING_DIR}  \
        ./build/tools/caffe train --solver=${WORKING_DIR}/runs/baseline_alexnet/1_node/solver_batch256.prototxt --par MPISyncCPU 
