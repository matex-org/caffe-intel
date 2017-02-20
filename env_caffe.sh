# this script should be "source"d from within the caffe-intel directory
# the following paths may need to be adjusted for the platform you are running on
# Additional paths are in the solver.prototxt and train*.prototxt files

# This version configured for  
#  gcc4.8.2, Jeff-compiled mvapich2.2, leveldb, and pnetcdf 

MPI_HOME=/home/d3n000/local_caffe_mvapich22
export JEFF_CAFFE_HOME=/home/d3n000/local_caffe

# device queue must still be specified in batch script

# make sure we're in the caffe-intel directory when running the script
export REAL_CURRENT_DIR=`pwd -P`
cd "${REAL_CURRENT_DIR}"

oldIFS=$IFS
IFS='/' 
current_dirpath=($REAL_CURRENT_DIR) && export foo=${current_dirpath[0]}
IFS=$oldIFS
if test "${current_dirpath[${#current_dirpath[@]}-1]}" != "caffe-intel" 
then 
  echo "Error - You must be in the local caffe-intel root directory for this to work properly"
  echo "Environment values not set"
  exit
else
  echo "Setting environment variables"
fi


# set default paths
module purge
export LD_LIBRARY_PATH=""
module load gcc/4.8.2
module load binutils/2.24
module load cmake/3.3.0
module load autotools/2014.10


export MPI_INCLUDE=$MPI_HOME/include
export MPI_LIB=$MPI_HOME/lib
export MPICC=$MPI_HOME/bin/mpicc
export MPICXX=$MPI_HOME/bin/mpicxx



if test "x$PATH" = x
then
  export PATH=${JEFF_CAFFE_HOME}/bin:${MPI_HOME}/bin
else
  export PATH=${JEFF_CAFFE_HOME}/bin:${MPI_HOME}/bin:${PATH}
fi

if test "x$LD_LIBRARY_PATH" = x
then
  export LD_LIBRARY_PATH=${JEFF_CAFFE_HOME}/lib:${MPI_HOME}/lib
else
  export LD_LIBRARY_PATH=${JEFF_CAFFE_HOME}/lib:${MPI_HOME}/lib:${LD_LIBRARY_PATH}
fi

# for cuda-7.5
# export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:${LD_LIBRARY_PATH}

# for cudnn
#export LD_LIBRARY_PATH=${JEFF_CAFFE_HOME}/../cuda/lib64:${LD_LIBRARY_PATH}

# for MKL
. ${JEFF_CAFFE_HOME}/../intel/mkl/bin/mklvars.sh intel64

# for leveldb
export LD_LIBRARY_PATH=${JEFF_CAFFE_HOME}/../leveldb-1.18:${LD_LIBRARY_PATH}

# for pycaffe
if test "x$PYTHONPATH" = x
then
  export PYTHONPATH=${REAL_CURRENT_DIR}/python
else
  export PYTHONPATH=${REAL_CURRENT_DIR}/python:${PYTHONPATH}
fi

export MV2_ENABLE_AFFINITY=0

