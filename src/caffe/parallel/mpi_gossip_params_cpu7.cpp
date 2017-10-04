#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/parallel/mpi_gossip_params_cpu7.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
static void apply_buffers(const vector<shared_ptr<Blob<Dtype> > >& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::next() {
  if (cube_) {
    if (rotate_) {
      next_cube_rotate();
    }
    else {
      next_cube();
    }
  }
  else {
    if (rotate_) {
      next_diffuse_rotate();
    }
    else {
      next_diffuse();
    }
  }
  //LOG(INFO) << "rank " << comm_rank_orig_ << " rot rank " << comm_rank_ << " send " << send_pair_ << " recv " << recv_pair_;
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::next_cube() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::next_cube_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    mci_ = (mci_+1)%comm_size_;
    comm_rank_ = caffe::mpi::comm_rank(comms_[mci_]);
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::next_diffuse() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  recv_pair_ = comm_rank_ + int(pow(2,hci_));
  send_pair_ = comm_rank_ - int(pow(2,hci_));
  if (recv_pair_ >= comm_size_) {
    recv_pair_ = recv_pair_ - comm_size_;
  }
  if (send_pair_ < 0) {
    send_pair_ = send_pair_ + comm_size_;
  }
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::next_diffuse_rotate() {
  if (hci_ > logp_) {
    hci_ = 0;
    mci_ = (mci_+1)%comm_size_;
    comm_rank_ = caffe::mpi::comm_rank(comms_[mci_]);
  }
  recv_pair_ = comm_rank_ + int(pow(2,hci_));
  send_pair_ = comm_rank_ - int(pow(2,hci_));
  if (recv_pair_ >= comm_size_) {
    recv_pair_ = recv_pair_ - comm_size_;
  }
  if (send_pair_ < 0) {
    send_pair_ = send_pair_ + comm_size_;
  }
  ++hci_;
}

template<typename Dtype>
MPIGossipParamsCPU7<Dtype>::MPIGossipParamsCPU7(
    shared_ptr<Solver<Dtype> > root_solver,
    const SolverParameter& param,
    bool cube,
    bool rotate)
  : CPUParams<Dtype>(root_solver),
    comm_rank_(),
    comm_size_(),
    logp_(0),
    hci_(0),
    mci_(0),
    send_pair_(0),
    recv_pair_(0),
    solver_(),
    sgdsolver_(),
    adamsolver_(),
    params_(root_solver->net()->learnable_params()),
    comms_(),
    requests_(),
    time_comm_(),
    time_comp_(),
    stats_comm_(),
    stats_comp_(),
    data_all_(),
    history_(),
    history_all_(),
    history_size_(),
    cube_(cube),
    rotate_(rotate),
    first_time_(true)
{
  int count = 0;
  int node_rank = 0;
  int node_size = 0;

  stats_clear(&stats_comm_);
  stats_clear(&stats_comp_);

  // one MPI_Comm per rank
  Timer timer_comm_create_;
  timer_comm_create_.Start();
  comm_size_ = caffe::mpi::comm_size();
  node_rank = caffe::mpi::node_rank();
  node_size = caffe::mpi::node_size();
  comms_.resize(comm_size_);
  comms_[0] = caffe::mpi::comm_dup();
  comm_rank_orig_ = caffe::mpi::comm_rank(comms_[0]);
  vector<int> ranks(comm_size_);
  for (int i = 0; i < comm_size_; ++i) {
    ranks[i] = i;
  }
  for (int i = 1; i < comm_size_; ++i) {
    if (0 == comm_rank_orig_) {
      std::random_shuffle(ranks.begin(), ranks.end());
    }
    caffe::mpi::bcast(ranks, 0, comms_[0]);
    comms_[i] = caffe::mpi::comm_create(ranks);
    LOG(INFO) << "my rank " << caffe::mpi::comm_rank(comms_[i]);
  }
  LOG(INFO) << "comm creation time " << timer_comm_create_.MilliSeconds();

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);

  sgdsolver_ = boost::dynamic_pointer_cast<SGDSolver<Dtype> >(root_solver);
  if (NULL == sgdsolver_) {
      LOG(FATAL) << "dynamic cast of SGDSolver failed";
  }
  adamsolver_ = boost::dynamic_pointer_cast<AdamSolver<Dtype> >(root_solver);
  if (NULL == adamsolver_) {
      LOG(INFO) << "dynamic cast of AdamSolver failed";
  }

  comm_rank_ = caffe::mpi::comm_rank(comms_[0]);
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  data_all_ = new Dtype[size_];

  /* AdamSolver has history that is twice the size */
  if (NULL != adamsolver_) {
    history_size_ = size_ * 2;
  }
  else {
    history_size_ = size_;
  }

  history_ = new Dtype[history_size_];
  caffe_set(history_size_, Dtype(0), history_);

  history_all_ = new Dtype[history_size_];
  caffe_set(history_size_, Dtype(0), history_all_);

  apply_buffers(sgdsolver_->history(), history_, history_size_, replace_cpu);

  LOG(INFO) << "buffer size_ " << size_*sizeof(Dtype);
  LOG(INFO) << "buffer history_size_ " << history_size_*sizeof(Dtype);
}

template<typename Dtype>
MPIGossipParamsCPU7<Dtype>::~MPIGossipParamsCPU7() {
  delete [] data_all_;
  delete [] history_;
  delete [] history_all_;
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
  CPUTimer timer;

  // wait for comm to finish and update buffers 
  if (first_time_) {
    LOG(INFO) << "first iteration doesn't wait for comm";
    first_time_ = false;
  }
  else {
    //solver_->DataShuffleEnd();
    timer.Start();
    CHECK_EQ(requests_.size(), 4);
    caffe::mpi::waitall(requests_);
    timer.Stop();
    time_comm_ += timer.MilliSeconds();
    timer.Start();
    caffe_cpu_axpby(size_, Dtype(0.5), data_all_, Dtype(0.5), data_);
    caffe_cpu_axpby(history_size_, Dtype(0.5), history_all_, Dtype(0.5), history_);
    timer.Stop();
    time_comp_ = timer.MilliSeconds();
    stats_sample_value(&stats_comm_, time_comm_);
    stats_sample_value(&stats_comp_, time_comp_);
    LOG_EVERY_N(INFO, 20) << "time comm sample " << time_comm_;
    LOG_EVERY_N(INFO, 20) << "time comp sample " << time_comp_;
    LOG_EVERY_N(INFO, 20) << "time comm " << stats_comm_._mean
      << " += " << stats_stddev(&stats_comm_)
      << " min " << stats_comm_._min
      << " max " << stats_comm_._max;
    LOG_EVERY_N(INFO, 20) << "time comp " << stats_comp_._mean
      << " += " << stats_stddev(&stats_comp_)
      << " min " << stats_comp_._min
      << " max " << stats_comp_._max;
  }

  // select next exchange partners
  next();

  // begin exchange of samples, data, and history
  {
    //solver_->DataShuffleBegin();
    timer.Start();
    MPI_Comm comm = comms_[mci_];
    requests_.assign(4, MPI_REQUEST_NULL);
    caffe::mpi::irecv(requests_[0], data_all_,    size_, recv_pair_, 1234, comm);
    caffe::mpi::isend(requests_[1], data_,        size_, send_pair_, 1234, comm);
    caffe::mpi::irecv(requests_[2], history_all_, history_size_, recv_pair_, 2345, comm);
    caffe::mpi::isend(requests_[3], history_,     history_size_, send_pair_, 2345, comm);
    timer.Stop();
    time_comm_ = timer.MilliSeconds();
  }

  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::make_progress() {
  CPUTimer timer;

  //solver_->DataShuffleTest();

  timer.Start();
  caffe::mpi::testall(requests_);
  timer.Stop();
  time_comm_ += timer.MilliSeconds();
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::on_forward(int param_id) {
  DLOG(INFO) << "on_forward(param_id)";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(param_id)";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
  make_progress();
}

template<typename Dtype>
int MPIGossipParamsCPU7<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  make_progress();
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::on_update() {
  DLOG(INFO) << "on_update()";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsCPU7<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsCPU7);

}  // namespace caffe

