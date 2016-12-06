#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/internode/mpiutil.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/mpi_server_cpu.hpp"

#define MYTAG 22333

DECLARE_bool(scale_on_apply);

DEFINE_bool(random_worker, true,
    "if par == MPIServerCPU, select worker at random");

namespace caffe {

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
SGDSolverServer<Dtype>::SGDSolverServer(const SolverParameter& param)
: SGDSolver<Dtype>(param),
#ifdef USE_MPI
  comm_(),
#endif
  comm_rank_(0),
  comm_size_(0),
  current_worker_(1),
  size_(total_size<Dtype>(net_->learnable_params())),
  data_(),
  diff_()
{
#ifdef USE_MPI
  comm_ = caffe::mpi::comm_dup();
  comm_rank_ = caffe::mpi::comm_rank(comm_);
  comm_size_ = caffe::mpi::comm_size(comm_);
  CHECK(comm_size_ > 1) << "must use at least 2 ranks with SGDSolverServer";

  const vector<Blob<Dtype>*>& params = net_->learnable_params();
  LOG(INFO) << "before buffer allocation";
  data_ = new Dtype[size_];
  apply_buffers(params, data_, size_, copy);
  LOG(INFO) << "copied net to data_";
  diff_ = new Dtype[size_];
  caffe_set(size_, Dtype(0), diff_);
  LOG(INFO) << "after buffer allocation";
  apply_buffers(params, data_, size_, replace_cpu);
  LOG(INFO) << "applied data_ buffer";
  apply_buffers(params, diff_, size_, replace_cpu_diff);
  LOG(INFO) << "applied diff_ buffer";

  if (0 == comm_rank_) {
    for (int i=1; i<comm_size_; ++i) {
      caffe::mpi::send(data_, size_, i, MYTAG, comm_);
    }
  }
  LOG(INFO) << "after pseudo broadcast";
#else
  NO_MPI;
#endif
}

template<typename Dtype>
SGDSolverServer<Dtype>::~SGDSolverServer() {
  delete [] data_;
  delete [] diff_;
}

template<typename Dtype>
void SGDSolverServer<Dtype>::ApplyUpdate() {
  if (0 == comm_rank_) {
    SGDSolver<Dtype>::ApplyUpdate();
  }
}

template<typename Dtype>
void SGDSolverServer<Dtype>::Step(int iters) {
  // basically copied from Solver<Dtype>::Step
  // however, rank 0 doesn't compute forward/backward
  // instead it recv/send gradients and params
  if (0 == comm_rank_) {
    const int start_iter = iter_;
    const int stop_iter = iter_ + iters*(comm_size_-1);
    int average_loss = this->param_.average_loss();
    losses_.clear();
    smoothed_loss_ = 0;
    iteration_timer_.Start();
    float lapse_total = 0;

    net_->SetSolver(this);

    while (iter_ < stop_iter) {
      if (param_.test_interval() && iter_ % param_.test_interval() == 0
          && (iter_ > 0 || param_.test_initialization())
          && Caffe::root_solver()) {
        TestAll();
        if (requested_early_exit_) {
          // Break out of the while loop because stop was requested while testing.
          break;
        }
      }

      for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->on_start();
      }
      const bool display = param_.display() && iter_ % param_.display() == 0;
      net_->set_debug_info(display && param_.debug_info());

      /* GET LOSS FROM SOMEONE */
      Dtype loss;
      caffe::mpi::recv(loss, current_worker_, MYTAG, comm_);

      // average the loss across iterations for smoothed reporting
      UpdateSmoothedLoss(loss, start_iter, average_loss);
      if (display) {
        float lapse = iteration_timer_.Seconds();
        float per_s = (iter_ - iterations_last_) / (lapse ? lapse : 1);
        lapse_total += lapse;
        float total_per_s = iter_ / lapse_total;
#ifdef USE_MPI
        LOG_IF(INFO, Caffe::root_solver())
          << caffe::internode::mpi_get_current_proc_rank_as_string()
          << " Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() <<" iter), loss = " << smoothed_loss_
          << ", " << total_per_s << " iter/s avg";
#else
        LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << " (" << per_s << " iter/s, " << lapse << "s/"
          << param_.display() <<" iter), loss = " << smoothed_loss_;
#endif
        iteration_timer_.Start();
        iterations_last_ = iter_;
        const vector<Blob<Dtype>*>& result = net_->output_blobs();
        int score_index = 0;
        for (int j = 0; j < result.size(); ++j) {
          const Dtype* result_vec = result[j]->cpu_data();
          const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
          const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
          for (int k = 0; k < result[j]->count(); ++k) {
            ostringstream loss_msg_stream;
            if (loss_weight) {
              loss_msg_stream << " (* " << loss_weight
                << " = " << loss_weight * result_vec[k] << " loss)";
            }
            LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
          }
        }
      }

      /* GET GRADIENTS FROM SAME SOMEONE */
      caffe::mpi::recv(diff_, size_, current_worker_, MYTAG, comm_);

      for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->on_gradients_ready();
      }
      if (!param().disabled_update()) {
        ApplyUpdate();
      }

      /* SEND NEW NET PARAMS */
      caffe::mpi::send(data_, size_, current_worker_, MYTAG, comm_);

      // Increment the internal iter_ counter -- its value should always indicate
      // the number of times the weights have been updated.
      ++iter_;

      SolverAction::Enum request = GetRequestedAction();

      // Save a snapshot if needed.
      if ((param_.snapshot()
            && iter_ % param_.snapshot() == 0
            && Caffe::root_solver()) ||
          (request == SolverAction::SNAPSHOT)) {
        Snapshot();
      }
      if (SolverAction::STOP == request) {
        requested_early_exit_ = true;
        // Break out of training loop.
        break;
      }

      current_worker_ += 1;
      if (current_worker_ >= (comm_size_-1)) {
        current_worker_ = 1;
      }
    }
  }
  else {
    const int stop_iter = iter_ + iters;
    losses_.clear();

    net_->SetSolver(this);

    /* get first net params */
    caffe::mpi::recv(data_, size_, 0, MYTAG, comm_);

    while (iter_ < stop_iter) {
      LOG(INFO) << "client " << comm_rank_ << " step " << iter_;
      Dtype loss;
      /* compute next batch */
      loss = forward_backward_();
      /* send loss */
      caffe::mpi::send(loss, 0, MYTAG, comm_);
      /* send gradients */
      caffe::mpi::send(diff_, size_, 0, MYTAG, comm_);
      /* get latest net params */
      caffe::mpi::recv(data_, size_, 0, MYTAG, comm_);
    }
  }
}

template<typename Dtype>
void SGDSolverServer<Dtype>::ApplyUpdate(int param_id) {
  if (0 == comm_rank_) {
    SGDSolver<Dtype>::ApplyUpdate(param_id);
  }
}

template<typename Dtype>
void SGDSolverServer<Dtype>::SnapshotSolverState(const string& model_filename) {
  if (0 == comm_rank_) {
    SGDSolver<Dtype>::SnapshotSolverState(model_filename);
  }
}

template<typename Dtype>
void SGDSolverServer<Dtype>::RestoreSolverStateFromBinaryProto(const string& state_file) {
  if (0 == comm_rank_) {
    SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(state_file);
  }
}

template<typename Dtype>
void SGDSolverServer<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  if (0 == comm_rank_) {
    SGDSolver<Dtype>::RestoreSolverStateFromHDF5(state_file);
  }
}

INSTANTIATE_CLASS(SGDSolverServer);

}  // namespace caffe

