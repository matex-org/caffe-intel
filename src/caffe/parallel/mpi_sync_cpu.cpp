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
  #if CAFFE_FT
  comm_ = caffe::mpi::get_working_comm();
  std::cout << "Working Comm MPISYNCCPU.\n";
  #else
  comm_ = caffe::mpi::comm_dup();
  #endif
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
void MPISyncCPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
#ifdef CAFFE_FT
std::tuple<int, bool> MPISyncCPU<Dtype>::on_gradients_ready() {
#else
void MPISyncCPU<Dtype>::on_gradients_ready() {
#endif
  DLOG(INFO) << "on_gradients_ready()";
#ifdef USE_MPI
  // Sum gradients
  #ifdef CAFFE_FT
  comm_ = caffe::mpi::get_working_comm();
  std::tuple<int,bool> ret_val
      = caffe::mpi::allreduce(diff_, size_, MPI_SUM, this->comm_);
  if(std::get<1>(ret_val)) {
    this->comm_ = caffe::mpi::get_working_comm();
    DLOG(INFO) << "RETVAL<1> true, MPISYNCCPU --------------" ;
  }
  if(std::get<0>(ret_val) != MPI_SUCCESS) { // This should not be triggered
    comm_ = caffe::mpi::get_working_comm();
    int temp_sz = caffe::mpi::comm_size(comm_);
    DLOG(INFO) << "Corrected Communicator Size {mpi_sync_cpu}!!!!!: " << temp_sz;
  }
  return ret_val;
  #else
  caffe::mpi::allreduce(diff_, size_, MPI_SUM, comm_);
  #endif
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
