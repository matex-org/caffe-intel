#!/bin/bash
export WORKING_DIR="/home/files0/warf949/caffe5/caffe-intel"
cd ${WORKING_DIR}
source ./env_caffe.sh
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4 
#SBATCH --time=12:00:00
#SBATCH --job-name "alexnet-quicksolver-16ranks-4nodes-allreduce-batch256"
#SBATCH -o stdout.alexnet.quicksolver.16ranks.4nodes.allreduce.batch256.txt
#SBATCH -e stderr.alexnet.quicksolver.16ranks.4nodes.allreduce.batch256.txt
#SBATCH -p all
mpiexec \
	-genv MV2_VBUF_TOTAL_SIZE=6144 \
        -genv MV2_IBA_EAGER_THRESHOLD=6144 \
        -genv OMP_NUM_THREADS=5 \
        -genv MKL_NUM_THREADS=5 \
        --wdir ${WORKING_DIR}  \
        ./build/tools/caffe train \
             --solver=${WORKING_DIR}/runs/alexnet_quicksolver/16_ranks/alexnet_solver_exp.prototxt \
             --par MPISyncCPU 
