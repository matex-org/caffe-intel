#!/bin/bash
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=4     
#SBATCH --time=12:30:00
#SBATCH --job-name "calexnet-16-3a"
#SBATCH -o stdout.alexnet.puma.16on4.rgbits3a.txt
#SBATCH -e stderr.alexnet.puma.16on4.rgbits3a.txt
#SBATCH -p all
export WORKING_DIR="/home/files0/warf949/caffe5/caffe-intel"
cd ${WORKING_DIR}
source ./env_caffe.sh
module load gcc/4.8.2
module load mvapich2/2.1a
module load binutils/2.24
/share/apps/mvapich2/2.1a/gcc/4.8.2/bin/mpiexec \
	-genv MV2_VBUF_TOTAL_SIZE=6144 \
        -genv MV2_IBA_EAGER_THRESHOLD=6144 \
        -genv OMP_NUM_THREADS=5 \
        -genv MKL_NUM_THREADS=3 \
        --wdir ${WORKING_DIR}  \
        ./build/tools/caffe train --solver=${WORKING_DIR}/runs/test3_alexnet/16nodes/solver3a.prototxt \
                                  --par MPISyncCPU \
                                  --rgroup_bits 3

