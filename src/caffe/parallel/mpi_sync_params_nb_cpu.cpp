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
#include "caffe/parallel/mpi_sync_params_nb_cpu.hpp"

namespace caffe {

template<typename Dtype>
MPISyncParamsNBCPU<Dtype>::MPISyncParamsNBCPU(shared_ptr<Solver<Dtype> > root_solver)
    : CPUParams<Dtype>(root_solver),
#ifdef USE_MPI
      comm_(),
      requests_(),
#endif
      comm_size_(),
      solver_(),
      params_(root_solver->net()->learnable_params()),
      timer_(),
      time_(0.0)
{
#ifdef USE_MPI
  comm_ = caffe::mpi::comm_dup();
  comm_size_ = caffe::mpi::comm_size(comm_);
  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);
  caffe::mpi::bcast(data_, size_, 0, comm_);
  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
  requests_.resize(params_.size());
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncParamsNBCPU<Dtype>::~MPISyncParamsNBCPU() {
}

template<typename Dtype>
void MPISyncParamsNBCPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
  LOG(INFO) << "time comm " << time_;
  time_ = 0.0;
}

template<typename Dtype>
void MPISyncParamsNBCPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
#ifdef USE_MPI
  caffe::mpi::waitall(requests_);
#endif
}

template<typename Dtype>
void MPISyncParamsNBCPU<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(param_id)";
#ifdef USE_MPI
  Blob<Dtype> *blob = params_[param_id];
  Dtype *param_diff = blob->mutable_cpu_diff();
  caffe::mpi::iallreduce(requests_[param_id], param_diff, blob->count(), MPI_SUM, comm_);
  caffe::mpi::test(requests_[param_id]);
#endif
}

template<typename Dtype>
int MPISyncParamsNBCPU<Dtype>::on_apply(int param_id) {
  return param_id;
}

template<typename Dtype>
void MPISyncParamsNBCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPISyncParamsNBCPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncParamsNBCPU);

}  // namespace caffe

