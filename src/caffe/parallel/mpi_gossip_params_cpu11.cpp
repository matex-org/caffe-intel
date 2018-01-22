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
#include "caffe/parallel/mpi_gossip_params_cpu11.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"

namespace caffe {

#define CAN_USE_PRV_DATA(param) (param->prv_data() && (param->prv_data_count() == param->count()))
#define CAN_USE_PRV_DIFF(param) (param->prv_diff() && (param->prv_diff_count() == param->count()))

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::next() {
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
void MPIGossipParamsCPU11<Dtype>::next_cube() {
  if (hci_ > logp_) {
    hci_ = 0;
  }
  send_pair_ = comm_rank_ ^ int(pow(2,hci_));
  recv_pair_ = send_pair_;
  ++hci_;
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::next_cube_rotate() {
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
void MPIGossipParamsCPU11<Dtype>::next_diffuse() {
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
void MPIGossipParamsCPU11<Dtype>::next_diffuse_rotate() {
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
MPIGossipParamsCPU11<Dtype>::MPIGossipParamsCPU11(
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
    hist_recv_(),
    data_recv_(),
    layers_(root_solver->net()->layers()),
    layer_param_ids_(),
    net_(root_solver->net()),
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

  layer_param_ids_.resize(layers_.size());
  for (int layer_id = 0; layer_id < layers_.size(); layer_id++) {
    shared_ptr<Layer<Dtype> > layer = layers_[layer_id];
    /* cache param ids */
    layer_param_ids_[layer_id] = net_->get_layer_learnable_param_ids(layer_id);
  }

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
  for (size_t i=0; i<params_.size(); ++i) {
      if (CAN_USE_PRV_DATA(params_[i])) {
        caffe::mpi::bcast(params_[i]->mutable_prv_data(), params_[i]->prv_data_count(), 0, comms_[0]);
      }
      else {
        caffe::mpi::bcast(params_[i]->mutable_cpu_data(), params_[i]->count(), 0, comms_[0]);
      }
  }

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  data_recv_.assign(params_.size(), NULL);
  for (size_t i=0; i<params_.size(); ++i) {
      data_recv_[i] = new Dtype[params_[i]->count()];
      caffe_set(params_[i]->count(), Dtype(0), data_recv_[i]);
  }

  hist_recv_.assign(sgdsolver_->history().size(), NULL);
  for (size_t i=0; i<sgdsolver_->history().size(); ++i) {
      hist_recv_[i] = new Dtype[sgdsolver_->history()[i]->count()];
      caffe_set(sgdsolver_->history()[i]->count(), Dtype(0), hist_recv_[i]);
  }
}

template<typename Dtype>
MPIGossipParamsCPU11<Dtype>::~MPIGossipParamsCPU11() {
}

template<typename Dtype>
bool MPIGossipParamsCPU11<Dtype>::param_needs_reduce(int param_id) {
    pair<int,int> tmp = net_->param_layer_indices()[param_id];
    int index_layer = tmp.first;
    int index_param = tmp.second;
    boost::shared_ptr<Layer<Dtype>> &layer = layers_[index_layer];
    return layer->ParamNeedReduce(index_param);
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
  CPUTimer timer;

  // wait for comm to finish and update buffers 
  if (first_time_) {
    LOG(INFO) << "first iteration doesn't wait for comm";
    first_time_ = false;
  }
  else {
    solver_->DataShuffleEnd();
    timer.Start();
    caffe::mpi::waitall(requests_);
    timer.Stop();
    time_comm_ += timer.MilliSeconds();
    timer.Start();
    for (size_t i=0; i<params_.size(); ++i) {
        if (!param_needs_reduce(i)) continue;
        if (CAN_USE_PRV_DATA(params_[i])) {
            caffe_cpu_axpby(params_[i]->prv_data_count(), Dtype(0.5), data_recv_[i], Dtype(0.5), params_[i]->mutable_prv_data());
        }
        else {
            caffe_cpu_axpby(params_[i]->count(), Dtype(0.5), data_recv_[i], Dtype(0.5), params_[i]->mutable_cpu_data());
        }
    }
    for (size_t i=0; i<sgdsolver_->history().size(); ++i) {
        if (!param_needs_reduce(i)) continue;
        if (CAN_USE_PRV_DATA(sgdsolver_->history()[i])) {
            caffe_cpu_axpby(sgdsolver_->history()[i]->prv_data_count(), Dtype(0.5), hist_recv_[i], Dtype(0.5), sgdsolver_->history()[i]->mutable_prv_data());
        }
        else {
            caffe_cpu_axpby(sgdsolver_->history()[i]->count(), Dtype(0.5), hist_recv_[i], Dtype(0.5), sgdsolver_->history()[i]->mutable_cpu_data());
        }
    }
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
    int req = 0;
    int tag = 0;
    const int TAG_BASE = 2345;
    solver_->DataShuffleBegin();
    timer.Start();
    MPI_Comm comm = comms_[mci_];
    requests_.assign(2*(params_.size()+sgdsolver_->history().size()), MPI_REQUEST_NULL);
    for (size_t i=0; i<params_.size(); ++i) {
        if (!param_needs_reduce(i)) continue;
        caffe::mpi::irecv(requests_[req], data_recv_[i], params_[i]->count(), recv_pair_, TAG_BASE+tag, comm);
        req += 1;
        if (CAN_USE_PRV_DATA(params_[i])) {
            caffe::mpi::isend(requests_[req], params_[i]->mutable_prv_data(), params_[i]->prv_data_count(), send_pair_, TAG_BASE+tag, comm);
        }
        else {
            caffe::mpi::isend(requests_[req], params_[i]->mutable_cpu_data(), params_[i]->count(), send_pair_, TAG_BASE+tag, comm);
        }
        req += 1;
        tag += 1;
    }
    for (size_t i=0; i<sgdsolver_->history().size(); ++i) {
        if (!param_needs_reduce(i)) continue;
        caffe::mpi::irecv(requests_[req], hist_recv_[i], sgdsolver_->history()[i]->count(), recv_pair_, TAG_BASE+tag, comm);
        req += 1;
        if (CAN_USE_PRV_DATA(sgdsolver_->history()[i])) {
            caffe::mpi::isend(requests_[req], sgdsolver_->history()[i]->mutable_prv_data(), sgdsolver_->history()[i]->prv_data_count(), send_pair_, TAG_BASE+tag, comm);
        }
        else {
            caffe::mpi::isend(requests_[req], sgdsolver_->history()[i]->mutable_cpu_data(), sgdsolver_->history()[i]->count(), send_pair_, TAG_BASE+tag, comm);
        }
        req += 1;
        tag += 1;
    }
    //CHECK_EQ(req, (2*(params_.size()+sgdsolver_->history().size())));
    timer.Stop();
    time_comm_ = timer.MilliSeconds();
  }

  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::make_progress() {
  CPUTimer timer;

  solver_->DataShuffleTest();

  timer.Start();
  caffe::mpi::testall(requests_);
  timer.Stop();
  time_comm_ += timer.MilliSeconds();
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::on_forward(int param_id) {
  DLOG(INFO) << "on_forward(param_id)";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(param_id)";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
  make_progress();
}

template<typename Dtype>
int MPIGossipParamsCPU11<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  make_progress();
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::on_update() {
  DLOG(INFO) << "on_update()";
  make_progress();
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsCPU11<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsCPU11);

}  // namespace caffe

