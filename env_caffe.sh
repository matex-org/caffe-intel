module purge
module load gcc/4.8.2
module load mvapich2/2.1a

module load binutils/2.24
module load cmake/3.3.0
module load autotools/2014.10

if test "x$PATH" = x
then
  export PATH=/home/d3n000/local_caffe/bin
else
  export PATH=/home/d3n000/local_caffe/bin:${PATH}
fi

if test "x$LD_LIBRARY_PATH" = x
then
  export LD_LIBRARY_PATH=/home/d3n000/local_caffe/lib
else
  export LD_LIBRARY_PATH=/home/d3n000/local_caffe/lib:${LD_LIBRARY_PATH}
fi

# for cuda-7.5
# export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:${LD_LIBRARY_PATH}

# for cudnn
# export LD_LIBRARY_PATH=/home/d3n000/cuda/lib64:${LD_LIBRARY_PATH}

# for MKL
. /home/d3n000/intel/mkl/bin/mklvars.sh intel64

# for leveldb
export LD_LIBRARY_PATH=/home/d3n000/leveldb-1.18:${LD_LIBRARY_PATH}

# for pycaffe
if test "x$PYTHONPATH" = x
then
  export PYTHONPATH=/home/files0/warf949/caffe5/caffe-intel/python
else
  export PYTHONPATH=/home/files0/warf949/caffe5/caffe-intel/python:${PYTHONPATH}
fi

export MV2_ENABLE_AFFINITY=0

