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
#include "caffe/parallel.hpp"
#include "caffe/parallel/ga_sync_cpu2.hpp"

#include "armci.h"

namespace caffe {

#define CAN_USE_PRV_DATA(param) (param->prv_data() && (param->prv_data_count() == param->count()))
#define CAN_USE_PRV_DIFF(param) (param->prv_diff() && (param->prv_diff_count() == param->count()))

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

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
GASyncCPU2<Dtype>::GASyncCPU2(shared_ptr<Solver<Dtype> > root_solver)
    : comm_rank_(),
      comm_size_(),
      solver_(),
      sgdsolver_(),
      params_(root_solver->net()->learnable_params()),
      time_comm_(),
      time_comp_(),
      stats_comm_(),
      stats_comp_(),
      size_(total_size<Dtype>(params_)),
      data_recv_(),
      hist_recv_(),
      data_hdl_(),
      hist_hdl_(),
      layers_(root_solver->net()->layers()),
      net_(root_solver->net()),
      first_time_(true)
{
  stats_clear(&stats_comm_);
  stats_clear(&stats_comp_);

  solver_ = root_solver;
  solver_->add_callback(this);

  sgdsolver_ = boost::dynamic_pointer_cast<SGDSolver<Dtype> >(root_solver);
  if (NULL == sgdsolver_) {
    LOG(FATAL) << "dynamic cast of SGDSolver failed";
  }

  comm_rank_ = caffe::mpi::comm_rank();
  comm_size_ = caffe::mpi::comm_size();

  LOG(INFO) << "params_.size() = " << params_.size();
  LOG(INFO) << "comm_rank_ = " << comm_rank_;
  LOG(INFO) << "comm_size_ = " << comm_size_;

  /* allocate ARMCI buffers for model data */
  data_pointers_.resize(comm_size_);
  /* allocate memory */
  ARMCI_Malloc(reinterpret_cast<void**>(&data_pointers_[0]),
               sizeof(Dtype)*size_);
  /* init memory to current value of param */
  apply_buffers(params_, data_pointers_[comm_rank_], size_, copy);
  /* replace param pointer */
  apply_buffers(params_, data_pointers_[comm_rank_], size_, replace_cpu);

  /* allocate ARMCI buffers for model history */
  hist_pointers_.resize(comm_size_);
  /* allocate memory */
  ARMCI_Malloc(reinterpret_cast<void**>(&hist_pointers_[0]),
               sizeof(Dtype)*size_);
  /* init memory to 0 */
  caffe_set(size_, Dtype(0), hist_pointers_[comm_rank_]);
  /* replace hist pointer */
  apply_buffers(sgdsolver_->history(), hist_pointers_[comm_rank_], size_, replace_cpu);

  /* allocate local receive buffers */
  data_recv_ = reinterpret_cast<Dtype*>(ARMCI_Malloc_local(sizeof(Dtype)*size_));
  hist_recv_ = reinterpret_cast<Dtype*>(ARMCI_Malloc_local(sizeof(Dtype)*size_));

  /* broadcast model from rank 0 so they all match */
  for (size_t i=0; i<params_.size(); ++i) {
    if (CAN_USE_PRV_DATA(params_[i])) {
      caffe::mpi::bcast(params_[i]->mutable_prv_data(), params_[i]->prv_data_count(), 0);
    }
    else {
      caffe::mpi::bcast(params_[i]->mutable_cpu_data(), params_[i]->count(), 0);
    }
  }
}

template<typename Dtype>
GASyncCPU2<Dtype>::~GASyncCPU2() {
}

template<typename Dtype>
bool GASyncCPU2<Dtype>::param_needs_reduce(int param_id) {
    pair<int,int> tmp = net_->param_layer_indices()[param_id];
    int index_layer = tmp.first;
    int index_param = tmp.second;
    boost::shared_ptr<Layer<Dtype>> &layer = layers_[index_layer];
    return layer->ParamNeedReduce(index_param);
}   

template<typename Dtype>
void GASyncCPU2<Dtype>::on_start() {
  int victim = rand() % comm_size_;
  DLOG(INFO) << "on_start() victim=" << victim;

  int data_test = 0;
  int hist_test = 0;

  if (first_time_) {
    first_time_ = false;
    /* prefetch data and history of random victim */
    ARMCI_NbGet(data_pointers_[victim],
        data_recv_,
        sizeof(Dtype)*size_,
        victim,
        &data_hdl_);
    ARMCI_NbGet(hist_pointers_[victim],
        hist_recv_,
        sizeof(Dtype)*size_,
        victim,
        &hist_hdl_);
  }
  else {
    /* blend with local, this also copies it to cpu from prv */
    data_test = ARMCI_Test(&data_hdl_);
    if (0 == data_test) {
      caffe_cpu_axpby(size_,
          Dtype(0.5), data_recv_,
          Dtype(0.5), data_pointers_[comm_rank_]);
      ARMCI_NbGet(data_pointers_[victim],
          data_recv_,
          sizeof(Dtype)*size_,
          victim,
          &data_hdl_);
    }
    hist_test = ARMCI_Test(&hist_hdl_);
    if (0 == hist_test) {
      caffe_cpu_axpby(size_,
          Dtype(0.5), hist_recv_,
          Dtype(0.5), hist_pointers_[comm_rank_]);
      ARMCI_NbGet(hist_pointers_[victim],
          hist_recv_,
          sizeof(Dtype)*size_,
          victim,
          &hist_hdl_);
    }
  }
}

template<typename Dtype>
void GASyncCPU2<Dtype>::on_forward(int param_id) {
  DLOG(INFO) << "on_forward(" << param_id << ")";
}

template<typename Dtype>
void GASyncCPU2<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
}

template<typename Dtype>
void GASyncCPU2<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(" << param_id << ")";
}

template<typename Dtype>
void GASyncCPU2<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void GASyncCPU2<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(GASyncCPU2);

}  // namespace caffe

