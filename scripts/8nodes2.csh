#!/bin/csh
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=8     
#SBATCH --time=00:15:00
#SBATCH --job-name "c-intel-parallel"
#SBATCH -o stdout.caffe.8.txt
#SBATCH -e stderr.caffe.8.txt
#SBATCH -p datavortex
module unload python/2.7.8
module unload gcc/4.9.2
module load gcc/4.8.2
module load mvapich2/2.1a
module load binutils/2.24
setenv WORKING_DIR "/home/files0/warf949/caffe2/caffe-intel"
cd ${WORKING_DIR}
/share/apps/mvapich2/2.1a/gcc/4.8.2/bin/mpiexec \
	-genv MV2_VBUF_TOTAL_SIZE=6144 \
        -genv MV2_IBA_EAGER_THRESHOLD=6144 \
        --wdir ${WORKING_DIR}  \
        ./build/tools/caffe train --solver=${WORKING_DIR}/examples/mnist/lenet_solver_pnetcdf.prototxt --par MPISyncCPU 
