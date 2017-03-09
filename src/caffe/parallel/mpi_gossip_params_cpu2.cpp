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
#include "caffe/parallel/mpi_gossip_params_cpu2.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
class MPIGossipParamsCPU2<Dtype>::Reducer : public InternalThread {
  public:
    MPIGossipParamsCPU2<Dtype> *sync_;
    int tid_;
    Timer timer_queue_;
    double time_in_queue_;
    Timer timer_comm_;
    double time_in_comm_;
    vector<double> time_per_param_;
    stats_t stats_queue_;
    stats_t stats_comm_;

    Reducer(MPIGossipParamsCPU2<Dtype> *sync, int tid)
        : sync_(sync), tid_(tid),
        timer_queue_(), time_in_queue_(0.0),
        timer_comm_(), time_in_comm_(0.0),
        time_per_param_(),
        stats_queue_(),
        stats_comm_()
    { 
      time_per_param_.resize(sync_->params_.size());
      stats_clear(&stats_queue_);
      stats_clear(&stats_comm_);
    }

    void InternalThreadEntry() {
      try {
        while (!must_stop()) {
          timer_queue_.Start();
          int param_id = sync_->param_solo_.pop("solo param not yet ready");
          time_in_queue_ += timer_queue_.MilliSeconds();
          Blob<Dtype> *blob = sync_->params_[param_id];
          MPI_Comm comm = sync_->comms_[param_id];
          Dtype *recvdiff = sync_->param_diffs_[param_id];
          Dtype *recvdata = sync_->param_datas_[param_id];
          sync_->next();
#ifdef USE_MPI
          timer_comm_.Start();
          // exchange data
#if 1
          caffe::mpi::sendrecv(
              (const Dtype*)blob->cpu_diff(), blob->count(), sync_->pair_, 1234,
              recvdiff, blob->count(), sync_->pair_, 1234, comm);
          caffe::mpi::sendrecv(
              (const Dtype*)blob->cpu_data(), blob->count(), sync_->pair_, 1234,
              recvdata, blob->count(), sync_->pair_, 1234, comm);
#else
          vector<MPI_Request> requests(4);
          caffe::mpi::irecv(requests[0], recvdiff,
              blob->count(), sync_->pair_, 2222, comm);
          caffe::mpi::irecv(requests[1], recvdata,
              blob->count(), sync_->pair_, 3333, comm);
          caffe::mpi::isend(requests[2], (const Dtype*)blob->cpu_diff(),
              blob->count(), sync_->pair_, 2222, comm);
          caffe::mpi::isend(requests[3], (const Dtype*)blob->cpu_data(),
              blob->count(), sync_->pair_, 3333, comm);
          caffe::mpi::waitall(requests);
#endif
          // postpone average local data and diff into secondary buffers
          time_per_param_[param_id] += timer_comm_.MilliSeconds();
          time_in_comm_ += timer_comm_.MilliSeconds();
#else       
          NO_MPI;
#endif        
          timer_queue_.Start();
          sync_->param_all_[param_id]->push(tid_);
          time_in_queue_ += timer_queue_.MilliSeconds();
        }
      } catch (boost::thread_interrupted&) {
        // Interrupted exception is expected on shutdown
      }
    }
};

template<typename Dtype>
static void get_pointers(const vector<Blob<Dtype>*>& blobs,
    Dtype *ptr, vector<Dtype*>& ptrs)
{
  for (int i = 0; i < blobs.size(); ++i) {
    ptrs[i] = ptr;
    ptr += blobs[i]->count();
  }
}

template<typename Dtype>
int MPIGossipParamsCPU2<Dtype>::next() {
  pair_ = comm_rank_ ^ int(pow(2,hci_));
  //LOG(INFO) << "next() for " << comm_rank_ << " returned " << pair_;
  ++hci_;
  if (hci_ > logp_) {
    hci_ = 0;
  }
  return pair_;
}

template<typename Dtype>
MPIGossipParamsCPU2<Dtype>::MPIGossipParamsCPU2(
    shared_ptr<Solver<Dtype> > root_solver,
    int comm_threads)
  : CPUParams<Dtype>(root_solver),
    comm_rank_(),
    comm_size_(),
    logp_(0),
    hci_(0),
    pair_(0),
    solver_(),
    params_(root_solver->net()->learnable_params()),
    param_solo_(),
    param_all_(),
    comms_(),
    reducers(),
    diff_all_(),
    data_all_(),
    param_diffs_(),
    param_datas_()
{
#ifdef USE_MPI
  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);
  // one MPI_Comm per parameter
  comms_.resize(params_.size());
  for (int i = 0; i < params_.size(); ++i) {
    comms_[i] = caffe::mpi::comm_dup();
  }
  comm_rank_ = caffe::mpi::comm_rank(comms_[0]);
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  // check that comm_size_ is a power of 2
  CHECK_EQ((comm_size_ & (comm_size_ - 1)), 0);
  logp_ = int(log2(comm_size_))-1;

  diff_all_ = new Dtype[size_];
  caffe_set(size_, Dtype(0), diff_all_);
  param_diffs_.resize(params_.size());
  get_pointers(params_, diff_all_, param_diffs_);

  data_all_ = new Dtype[size_];
  caffe_set(size_, Dtype(0), data_all_);
  param_datas_.resize(params_.size());
  get_pointers(params_, data_all_, param_datas_);

  // create queue, one per param
  param_all_.resize(params_.size());
  for (int i = 0; i < params_.size(); ++i) {
    param_all_[i] = new BlockingQueue<int>;
  }
  
  // Start the gradient allreduce threads
  reducers.resize(comm_threads);
  for (int i = 0; i < comm_threads; ++i) {
    reducers[i] = new Reducer(this, i);
    reducers[i]->StartInternalThread();
  }
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIGossipParamsCPU2<Dtype>::~MPIGossipParamsCPU2() {
  for (int i = 0; i < reducers.size(); ++i) {
    reducers[i]->StopInternalThread();
    delete reducers[i];
  }
  delete [] diff_all_;
  delete [] data_all_;
  for (int i = 0; i < params_.size(); ++i) {
    delete param_all_[i];
  }
}

template<typename Dtype>
void MPIGossipParamsCPU2<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
  for (int i=0; i<reducers.size(); ++i) {
    stats_sample_value(&reducers[i]->stats_queue_, reducers[i]->time_in_queue_);
    stats_sample_value(&reducers[i]->stats_comm_, reducers[i]->time_in_comm_);
    //LOG(INFO) << "reducer[" << i << "] time queue " << reducers[i]->time_in_queue_ << " time comm " << reducers[i]->time_in_comm_;
    LOG_EVERY_N(INFO, 20) << "reducer[" << i << "] time queue " << reducers[i]->stats_queue_._mean << " time comm " << reducers[i]->stats_comm_._mean;
#if 0
    if (solver_->iter() > 0) {
      for (int j=params_.size()-1; j >= 0; --j) {
        LOG(INFO) << j << ": " << reducers[i]->time_per_param_[j]/solver_->iter();
      }
    }
#endif
    reducers[i]->time_in_queue_ = 0.0;
    reducers[i]->time_in_comm_ = 0.0;
  }
}

template<typename Dtype>
void MPIGossipParamsCPU2<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
}

template<typename Dtype>
void MPIGossipParamsCPU2<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(param_id)";
  param_solo_.push(param_id);
}

template<typename Dtype>
int MPIGossipParamsCPU2<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  int who_did_the_work = param_all_[param_id]->pop("waiting in apply");
  Blob<Dtype> *blob = params_[param_id];

  // average pairwise exhange
  caffe_cpu_axpby(blob->count(), Dtype(0.5), blob->cpu_diff(), Dtype(0.5), param_diffs_[param_id]);
  caffe_cpu_axpby(blob->count(), Dtype(0.5), blob->cpu_data(), Dtype(0.5), param_datas_[param_id]);

  // swap diff and data pointers with reduction pointers
  Dtype *swap;
  swap = blob->mutable_cpu_diff();
  blob->diff()->set_cpu_data(param_diffs_[param_id]);
  param_diffs_[param_id] = swap;
  swap = blob->mutable_cpu_data();
  blob->data()->set_cpu_data(param_datas_[param_id]);
  param_datas_[param_id] = swap;
  return param_id;
}

template<typename Dtype>
void MPIGossipParamsCPU2<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIGossipParamsCPU2<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIGossipParamsCPU2);

}  // namespace caffe

