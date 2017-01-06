#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/mpi_sync_cpu.hpp"

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
  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
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
  //std::cout << "Here--------on_start()size_:" << size_ << std::endl;
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

