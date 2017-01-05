#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/mpi_sync_cpu.hpp"

DEFINE_string(par, "",
    "Optional; select parallelization strategy, e.g., MPISyncCPU");
DEFINE_int32(buffer_depth, 2,
    "Optional; parallel mode, the number of buffers used by "
    "communication code.");
DEFINE_bool(scale_on_apply, true,
    "Optional; parallel mode, whether scaling gradients occurs during "
    "the ApplyUpdate phase as part of regular operations.");

namespace caffe {

template<typename Dtype>
MPISyncCPU<Dtype>::MPISyncCPU(shared_ptr<Solver<Dtype> > root_solver)
    : CPUParams<Dtype>(root_solver),
#ifdef USE_MPI
      comm_(),
#endif
      comm_size_(),
      solver_() {
#ifdef USE_MPI
  comm_ = caffe::mpi::comm_dup();
  comm_size_ = caffe::mpi::comm_size(comm_);
  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);
  caffe::mpi::bcast(data_, size_, 0, comm_);
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncCPU<Dtype>::~MPISyncCPU() {
}

template<typename Dtype>
#ifdef ADAPTIVE_BATCH
void MPISyncCPU<Dtype>::on_start(int iter) {
#else
void MPISyncCPU<Dtype>::on_start() {
#endif
  DLOG(INFO) << "on_start()";
#ifdef ADAPTIVE_BATCH
 #ifdef USE_MPI
  // Sum data_
  // std::cout << "Here--------on_start()size_:" << size_ << std::endl;
  DLOG(INFO) << "on_start(), Data";
  //std::cout << "Here--------on_start() comm_size_:" << comm_size_ << std::endl;

  if(iter == 0) {
    caffe::mpi::bcast(data_, size_, 0, comm_);
  } else {
    caffe::mpi::allreduce(data_, size_, MPI_SUM, comm_);
    // std::cout << "Here ---- caffe_scal(data)called" << std::endl;
    caffe_scal(size_, Dtype(1.0 / comm_size_), data_);
  }
 #else
  NO_MPI;
 #endif
#endif
}

template<typename Dtype>
void MPISyncCPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
#ifdef USE_MPI
  // Sum gradients
  caffe::mpi::allreduce(diff_, size_, MPI_SUM, comm_);
  if (!FLAGS_scale_on_apply) {
    caffe_scal(size_, Dtype(1.0 / comm_size_), diff_);
  }
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPISyncCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPISyncCPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncCPU);

}  // namespace caffe
