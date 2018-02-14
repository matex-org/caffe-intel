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
#include "caffe/parallel/ga_sync_cpu3.hpp"

#include "armci.h"

namespace caffe {

#define CAN_USE_PRV_DATA(param) (param->prv_data() && (param->prv_data_count() == param->count()))
#define CAN_USE_PRV_DIFF(param) (param->prv_diff() && (param->prv_diff_count() == param->count()))

template<typename Dtype>
GASyncCPU3<Dtype>::GASyncCPU3(shared_ptr<Solver<Dtype> > root_solver)
    : comm_rank_(),
      comm_size_(),
      solver_(),
      sgdsolver_(),
      params_(root_solver->net()->learnable_params()),
      time_comm_(),
      time_comp_(),
      stats_comm_(),
      stats_comp_(),
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

  LOG(INFO) << "params_.size() = " << params_.size();

  sgdsolver_ = boost::dynamic_pointer_cast<SGDSolver<Dtype> >(root_solver);
  if (NULL == sgdsolver_) {
    LOG(FATAL) << "dynamic cast of SGDSolver failed";
  }

  /* allocate ARMCI buffers for model data */
  comm_rank_ = caffe::mpi::comm_rank();
  comm_size_ = caffe::mpi::comm_size();
  data_pointers_.resize(params_.size());
  for (size_t i=0; i<params_.size(); ++i) {
    data_pointers_[i].resize(comm_size_);
    /* allocate memory */
    ARMCI_Malloc(reinterpret_cast<void**>(&data_pointers_[i][0]),
        sizeof(Dtype)*params_[i]->count());
    /* init memory to current value of param */
    caffe_copy(params_[i]->count(),
        reinterpret_cast<const Dtype*>(params_[i]->data()->cpu_data()),
        data_pointers_[i][comm_rank_]);
    /* replace param pointer */
    params_[i]->data()->set_cpu_data(data_pointers_[i][comm_rank_]);
  }

  /* allocate ARMCI buffers for model history */
  hist_pointers_.resize(params_.size());
  for (size_t i=0; i<params_.size(); ++i) {
    hist_pointers_[i].resize(comm_size_);
    /* allocate memory */
    ARMCI_Malloc(reinterpret_cast<void**>(&hist_pointers_[i][0]),
        sizeof(Dtype)*params_[i]->count());
    /* init memory to 0 */
    caffe_set(sgdsolver_->history()[i]->count(),
        Dtype(0), hist_pointers_[i][comm_rank_]);
    /* replace hist pointer */
    sgdsolver_->history()[i]->data()->set_cpu_data(hist_pointers_[i][comm_rank_]);
  }

  /* allocate local receive buffers */
  data_recv_.resize(params_.size());
  hist_recv_.resize(params_.size());
  for (size_t i=0; i<params_.size(); ++i) {
    data_recv_[i] = reinterpret_cast<Dtype*>(ARMCI_Malloc_local(sizeof(Dtype)*params_[i]->count()));
    hist_recv_[i] = reinterpret_cast<Dtype*>(ARMCI_Malloc_local(sizeof(Dtype)*params_[i]->count()));
  }

  data_hdl_.resize(params_.size());
  hist_hdl_.resize(params_.size());

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
GASyncCPU3<Dtype>::~GASyncCPU3() {
}

template<typename Dtype>
bool GASyncCPU3<Dtype>::param_needs_reduce(int param_id) {
    pair<int,int> tmp = net_->param_layer_indices()[param_id];
    int index_layer = tmp.first;
    int index_param = tmp.second;
    boost::shared_ptr<Layer<Dtype>> &layer = layers_[index_layer];
    return layer->ParamNeedReduce(index_param);
}   

template<typename Dtype>
void GASyncCPU3<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
}

template<typename Dtype>
void GASyncCPU3<Dtype>::on_forward(int param_id) {
  int victim = rand() % comm_size_;
  DLOG(INFO) << "on_forward(" << param_id << ") victim=" << victim;

  /* prefetch data and history of random victim */
  ARMCI_NbGet(data_pointers_[param_id][victim],
      data_recv_[param_id],
      sizeof(Dtype)*params_[param_id]->count(),
      victim,
      &data_hdl_[param_id]);
  ARMCI_NbGet(hist_pointers_[param_id][victim],
      hist_recv_[param_id],
      sizeof(Dtype)*params_[param_id]->count(),
      victim,
      &hist_hdl_[param_id]);

  /* blend with local, this also copies it to cpu from prv */
  ARMCI_Wait(&data_hdl_[param_id]);
  caffe_cpu_axpby(params_[param_id]->count(),
      Dtype(0.5), data_recv_[param_id],
      Dtype(0.5), params_[param_id]->mutable_cpu_data());
  ARMCI_Wait(&hist_hdl_[param_id]);
  caffe_cpu_axpby(params_[param_id]->count(),
      Dtype(0.5), hist_recv_[param_id],
      Dtype(0.5), sgdsolver_->history()[param_id]->mutable_cpu_data());
}

template<typename Dtype>
void GASyncCPU3<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
  first_time_ = false;
}

template<typename Dtype>
void GASyncCPU3<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(" << param_id << ")";
}

template<typename Dtype>
void GASyncCPU3<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void GASyncCPU3<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(GASyncCPU3);

}  // namespace caffe

