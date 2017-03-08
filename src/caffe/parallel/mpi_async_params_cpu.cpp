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
#include "caffe/parallel/mpi_async_params_cpu.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template<typename Dtype>
class MPIAsyncParamsCPU<Dtype>::Reducer : public InternalThread {
  public:
    MPIAsyncParamsCPU<Dtype> *sync_;
    int tid_;
    Timer timer_queue_;
    double time_in_queue_;
    Timer timer_comm_;
    double time_in_comm_;
    vector<double> time_per_param_;
    stats_t stats_queue_;
    stats_t stats_comm_;

    Reducer(MPIAsyncParamsCPU<Dtype> *sync, int tid)
        : sync_(sync), tid_(tid),
        timer_queue_(), time_in_queue_(0.0),
        timer_comm_(), time_in_comm_(0.0),
        time_per_param_(),
        stats_queue_(),
        stats_comm_()
  { 
    time_per_param_.resize(sync->params_.size());
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
          Dtype *sum = sync_->param_diffs_[param_id];
#ifdef USE_MPI
#if 0
          // sum gradients
          if (sync_->params_[param_id]->prv_diff()
              && (sync_->params_[param_id]->prv_diff_count()
                == sync_->params_[param_id]->count())) {
            timer_comm_.Start();
            caffe::mpi::allreduce_copy((const Dtype*)blob->prv_diff(),
                sum, blob->count(), MPI_SUM, comm);
            time_in_comm_ += timer_comm_.MilliSeconds();
          }
          else {
#endif
#if 1
            timer_comm_.Start();
            caffe::mpi::allreduce_copy((const Dtype*)blob->cpu_diff(),
                sum, blob->count(), MPI_SUM, comm);
            time_per_param_[param_id] += timer_comm_.MilliSeconds();
            time_in_comm_ += timer_comm_.MilliSeconds();
#else
            timer_comm_.Start();
            caffe::mpi::allreduce(blob->mutable_cpu_diff(),
                blob->count(), MPI_SUM, comm);
            time_in_comm_ += timer_comm_.MilliSeconds();
#endif
#if 0
          }
#endif
          //caffe_scal(blob->count(), Dtype(1.0 / sync_->comm_size_), sum);
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
MPIAsyncParamsCPU<Dtype>::MPIAsyncParamsCPU(
    shared_ptr<Solver<Dtype> > root_solver,
    int comm_threads)
  : CPUParams<Dtype>(root_solver),
    comm_size_(),
    solver_(),
    params_(root_solver->net()->learnable_params()),
    param_solo_(),
    param_all_(),
    comms_(),
    reducers(),
    diff_all_(),
    param_diffs_()
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
  comm_size_ = caffe::mpi::comm_size(comms_[0]);
  caffe::mpi::bcast(data_, size_, 0, comms_[0]);

  diff_all_ = new Dtype[size_];
  caffe_set(size_, Dtype(0), diff_all_);
  param_diffs_.resize(params_.size());
  get_pointers(params_, diff_all_, param_diffs_);

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

  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));
#else
  NO_MPI;
#endif
}

template<typename Dtype>
MPIAsyncParamsCPU<Dtype>::~MPIAsyncParamsCPU() {
  for (int i = 0; i < reducers.size(); ++i) {
    reducers[i]->StopInternalThread();
    delete reducers[i];
  }
  delete [] diff_all_;
  for (int i = 0; i < params_.size(); ++i) {
    delete param_all_[i];
  }
}

template<typename Dtype>
void MPIAsyncParamsCPU<Dtype>::on_start() {
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
void MPIAsyncParamsCPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
}

template<typename Dtype>
void MPIAsyncParamsCPU<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(param_id)";
  param_solo_.push(param_id);
}

template<typename Dtype>
int MPIAsyncParamsCPU<Dtype>::on_apply(int param_id) {
  DLOG(INFO) << "on_apply(param_id)";
  int who_did_the_work = param_all_[param_id]->pop("waiting in apply");
  Blob<Dtype> *blob = params_[param_id];
  Dtype *swap = blob->mutable_cpu_diff();
  blob->diff()->set_cpu_data(param_diffs_[param_id]);
  param_diffs_[param_id] = swap;
  return param_id;
}

template<typename Dtype>
void MPIAsyncParamsCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPIAsyncParamsCPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPIAsyncParamsCPU);

}  // namespace caffe

