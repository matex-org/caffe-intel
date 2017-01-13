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
#include "caffe/parallel/mpi_sync_params_cpu.hpp"

namespace caffe {

template<typename Dtype>
MPISyncParamsCPU<Dtype>::MPISyncParamsCPU(shared_ptr<Solver<Dtype> > root_solver)
    : CPUParams<Dtype>(root_solver),
#ifdef USE_MPI
      comm_(),
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
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPISyncParamsCPU<Dtype>::~MPISyncParamsCPU() {
}

template<typename Dtype>
void MPISyncParamsCPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
  LOG(INFO) << "time comm " << time_;
  time_ = 0.0;
}

template<typename Dtype>
void MPISyncParamsCPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
}

template<typename Dtype>
void MPISyncParamsCPU<Dtype>::on_apply(int param_id) {
#ifdef USE_MPI
  Blob<Dtype> *blob = params_[param_id];
  Dtype *param_diff = blob->mutable_cpu_diff();
  // Sum gradients
  timer_.Start();
  caffe::mpi::allreduce(param_diff, blob->count(), MPI_SUM, comm_);
  time_ += timer_.MilliSeconds();
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPISyncParamsCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPISyncParamsCPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncParamsCPU);

}  // namespace caffe

