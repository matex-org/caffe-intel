#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/mpi_server_cpu.hpp"

#define MYTAG 22333

DECLARE_bool(scale_on_apply);

DEFINE_bool(random_worker, true,
    "if par == MPIServerCPU, select worker at random");

namespace caffe {

template<typename Dtype>
MPIServerCPU<Dtype>::MPIServerCPU(shared_ptr<Solver<Dtype> > root_solver)
    : CPUParams<Dtype>(root_solver),
#ifdef USE_MPI
      comm_(),
#endif
      comm_rank_(),
      comm_size_(),
      current_worker_(0),
      last_worker_(0),
      solver_() {
#ifdef USE_MPI
  comm_ = caffe::mpi::comm_dup();
  comm_rank_ = caffe::mpi::comm_rank(comm_);
  comm_size_ = caffe::mpi::comm_size(comm_);
  if (0 == comm_rank_) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(root_solver->param(), root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);
  caffe::mpi::bcast(data_, size_, 0, comm_);
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIServerCPU<Dtype>::~MPIServerCPU() {
}

template<typename Dtype>
void MPIServerCPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
#ifdef USE_MPI
  if (0 == comm_rank_) {
    caffe::mpi::send(data_, size_, last_worker_, MYTAG, comm_);
  } else {
    MPI_Status status;
    status = caffe::mpi::recv(data_, size_, 0, MYTAG, comm_);
  }
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPIServerCPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
#ifdef USE_MPI
  if (0 == comm_rank_) {
    MPI_Status status;
    int worker = 0;
    if (FLAGS_random_worker) {
      worker = MPI_ANY_SOURCE;
    } else {
      worker = (current_worker_ >= (comm_size_-1)) ? 1 : current_worker_+1;
    }
    status = caffe::mpi::recv(diff_, size_, worker, MYTAG, comm_);
    last_worker_ = status.MPI_SOURCE;
  } else {
    caffe::mpi::send(diff_, size_, 0, MYTAG, comm_);
  }
#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPIServerCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIServerCPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIServerCPU);

}  // namespace caffe

