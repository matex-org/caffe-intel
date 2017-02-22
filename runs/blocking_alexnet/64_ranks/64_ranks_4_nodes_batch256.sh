#!/bin/bash
export WORKING_DIR="/home/files0/warf949/caffe5/caffe-intel"
cd ${WORKING_DIR}
source ./env_caffe.sh
#SBATCH --ntasks-per-node=16
#SBATCH --nodes=4     
#SBATCH --time=12:00:00
#SBATCH --job-name "blocking-64ranks-4nodes-batch256"
#SBATCH -o stdout.blocking.64ranks.4nodes.batch256.txt
#SBATCH -e stderr.blocking.64ranks.4nodes.batch256.txt
#SBATCH -p all
mpiexec \
	-genv MV2_VBUF_TOTAL_SIZE=6144 \
        -genv MV2_IBA_EAGER_THRESHOLD=6144 \
        -genv OMP_NUM_THREADS=1 \
        -genv MKL_NUM_THREADS=1 \
        --wdir ${WORKING_DIR}  \
        ./build/tools/caffe train --solver=${WORKING_DIR}/runs/blocking_alexnet/64_ranks/solver_batch256.prototxt --par MPISyncCPU --rgroup_bits 5
