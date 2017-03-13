/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <cstdio>
#include <string>
#include <vector>

#include <numeric>

#include "boost/bind.hpp"
#include "caffe/internode/mpiutil.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/performance.hpp"
#include "caffe/util/upgrade_proto.hpp"

#ifdef USE_MLSL
#include <mlsl.h>
#include <mpi.h>
#endif /* USE_MLSL */

#ifdef ADAPTIVE_BATCH
#include <cstdlib>
#include <mpi.h>
#include <math.h> // temp
#endif

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false),
      scale_on_apply_(1.0),
#ifdef ADAPTIVE_BATCH
      forward_backward_(boost::bind(&Solver<Dtype>::ForwardBackward, this, _1) )  {
#else
      forward_backward_(boost::bind(&Solver<Dtype>::ForwardBackward, this) )  {
#endif
  Init(param);
  Caffe::set_iter_size(param_.iter_size());
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false),
      scale_on_apply_(1.0),
#ifdef ADAPTIVE_BATCH
      forward_backward_(boost::bind(&Solver<Dtype>::ForwardBackward, this, _1) )  {
#else
      forward_backward_(boost::bind(&Solver<Dtype>::ForwardBackward, this) )  {
#endif
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
  Caffe::set_iter_size(param_.iter_size());
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
#if !defined(USE_MPI) && !defined(USE_MLSL)
  CheckSnapshotWritePermissions();
#endif
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;

#ifdef CAFFE_PER_LAYER_TIMINGS
  InitTimers();
#endif

}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  if (param_.engine() != "")
    net_param.set_engine(param_.engine());
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);

    if (param_.engine() != "")
      net_params[i].set_engine(param_.engine());

    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

#ifdef ADAPTIVE_BATCH
template <typename Dtype>
void Solver<Dtype>::AssignItersize(std::size_t itersize) {
    Caffe::set_iter_size(itersize);
    newitersize_ = itersize;
}

template <typename Dtype>
Dtype Solver<Dtype>::ForwardBackward(int iter_size) {
  // zero-init the params
  net_->ClearParamDiffs();

  Dtype loss = Dtype();
  vector<Blob<Dtype>*> bottom_vec;

  AssignItersize(iter_size);

  // accumulate the loss and gradient
  for (int i = 0; i < iter_size; ++i) {
    loss += net_->ForwardBackward();
  }
  return loss / iter_size;
}

#else
template <typename Dtype>
Dtype Solver<Dtype>::ForwardBackward() {
  // zero-init the params
  net_->ClearParamDiffs();

  Dtype loss = Dtype();
  vector<Blob<Dtype>*> bottom_vec;

  // accumulate the loss and gradient
  for (int i = 0; i < param_.iter_size(); ++i) {
    loss += net_->ForwardBackward();
  }
  return loss / param_.iter_size();
}
#endif

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;

#ifdef ADAPTIVE_BATCH
  // std::size_t batch_icnt;
 #ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 #endif
  int new_iter_size = param_.iter_size(); // 4
  std::size_t batch_iter_count = 0;
  bool batch_h_update = false;
  // bool all_reduce_onceData = false;
  // bool all_reduce_onceHistory = false;
  Dtype lastLoss = 0;
  int temp = 0;
  int randomThres;
  float lossThres;
  float CToCThres;  // ratio of communication to computation < 1.
  float currentCToC = 1;
  std::mt19937 gen; // seed for random no. generator.
  // Choose the Hieuristic type: random, ratioCTC,etc.
  // For AllReduce batches.
  std::vector<int> tempBatchSizes;
  std::vector<int>::iterator itrB;
  for( int i = 0; i< 3; ++i) {
    tempBatchSizes.push_back(pow(4.0, i));
    DLOG(INFO) << "tempBatchSizes ----- : "<< tempBatchSizes[i] << "\n";
    DLOG(INFO) << "Power ----- : "<< int(pow(4.0, i)) << "\n";
  }
  itrB = tempBatchSizes.begin();

  std::string hieuristic_OptType(std::getenv("ADAPTIVEB_OPTION"));
  DLOG(INFO) << "LOSSRATE_Hieuristic-----\n";

  char const* hieuristic_RandomThres = std::getenv("RANDOMTHRES");
  char const* hieuristic_LossThres = std::getenv("LOSSTHRES");
  char const* hieuristic_CToCThres = std::getenv("CTOCTHRES");

  typedef AdaptiveBatchOption::Random batchOptionRan;
  typedef AdaptiveBatchOption::LossRate batchOptionLR;
  typedef AdaptiveBatchOption::RatioCToC batchOptionRatioCToC;

  if(hieuristic_OptType == "RANDOM") {
    randomThres =
      (hieuristic_RandomThres !=NULL) ? atoi(hieuristic_RandomThres) : 1;
  }
  else if (hieuristic_OptType == "LOSSRATE") {
    lossThres =
      (hieuristic_LossThres != NULL) ? atof(hieuristic_LossThres) : 1;
    DLOG(INFO) << "LossThres-------: " << lossThres << "\n";
  }
  else if (hieuristic_OptType == "RATIOCTOC") {
    CToCThres =
      (hieuristic_CToCThres != NULL) ? atof(hieuristic_CToCThres) : 1;
  }

  int batch_apply_iter = 1; // starting default value

#else
  int batch_ongradients_iter = 1;
  // std::size_t lossesHistorySize = 20;
#endif

  while (iter_ < stop_iter) {
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {

      LOG(INFO) << "TestAll";

      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }

#ifdef ADAPTIVE_BATCH
  if(batch_iter_count < 1) {
    // all_reduce_onceData = false;
    // all_reduce_onceHistory = false;

    if(hieuristic_OptType == "RANDOM") {
      batch_apply_iter = NewBatchSize<batchOptionRan>::get(randomThres,gen);
      DLOG(INFO) << "BATCHAPPLYITER value:" << batch_apply_iter;
      temp  = stop_iter - iter_;
      if(temp < randomThres)
        batch_apply_iter = (temp%2) == 0? temp/2 : 1;
    }
    else if (hieuristic_OptType == "LOSSRATE") {
      int last_batchApplyIter = batch_apply_iter;
      DLOG(INFO) << "lastBatchApplyIter : ------------" << last_batchApplyIter;
      // batch_apply_iter = *itrB;
      if((deltaLosses_.size() > 20) && (iter_ > 300))
      {
        batch_apply_iter = NewBatchSize<batchOptionLR>::get(deltaLosses_
        , lossThres, last_batchApplyIter);
      }
      // ++itrB;
      // if(itrB == tempBatchSizes.end())
      //  itrB = tempBatchSizes.begin();
    }
    else if(hieuristic_OptType == "RATIOCTOC") {
      //TODO: Need to revisit
      batch_apply_iter =
        NewBatchSize<batchOptionRatioCToC>::get(CToCThres, currentCToC);
    }
    else if(hieuristic_OptType == "FIXEDSTEP") {
      batch_apply_iter = *itrB;
      DLOG(INFO) << "BATCHAPPLYITER (FIXED STEP) value:" << batch_apply_iter;
      ++itrB;
      if(itrB == tempBatchSizes.end())
       itrB = tempBatchSizes.begin();
    }

    batch_iter_count = batch_apply_iter;
  #ifdef USE_MPI
    MPI_Bcast(&batch_apply_iter, 1, MPI_INT, 0, MPI_COMM_WORLD);
  #endif
  }
  DLOG(INFO) << "BATCH_ITER_COUNT :" << batch_iter_count << " BATCHAPPLYITER VAL " << batch_apply_iter << " IterVal:" << iter_;
#endif /* ADAPTIVE_BATCH */
  Timer total_timer, comm_timer;
  double total_time = 0, comm_time = 0;
  // total_timer.Start();
  comm_timer.Start();

  for (int i = 0; i < callbacks_.size(); ++i) {
#ifdef ADAPTIVE_BATCH
    // if(((iter_ % batch_apply_iter) == 0) && !all_reduce_onceData) {
    if((iter_ % batch_apply_iter) == 0) {
      callbacks_[i]->on_start(iter_);
      // all_reduce_onceData = true;
    }
#else
    callbacks_[i]->on_start();
#endif
  }
  double temp_ctime = 0;
  temp_ctime += comm_timer.MilliSeconds();
  comm_time += temp_ctime;
  total_time += temp_ctime;
  // comm_timer.Stop();
  const bool display = param_.display() && iter_ % param_.display() == 0;
  net_->set_debug_info(display && param_.debug_info());

  Timer iter_timer;
  double iter_time = 0;
  iter_timer.Start();

#ifdef ADAPTIVE_BATCH
  Dtype loss = forward_backward_(new_iter_size);

  if(deltaLosses_.size() < 21) {
    deltaLosses_.push_front(lastLoss - loss);
  }
  else {
    deltaLosses_.pop_back();
    deltaLosses_.push_front(lastLoss - loss);
  }
  lastLoss = loss;
#else
  Dtype loss = forward_backward_();
#endif

  double temp_time = 0;
  temp_time += iter_timer.MilliSeconds();
  iter_time += temp_time;
  total_time += temp_time;

    // average the loss across iterations for smoothed reporting
  UpdateSmoothedLoss(loss, start_iter, average_loss);
    if (display) {
#ifdef USE_MPI
      LOG_IF(INFO, Caffe::root_solver())
             << caffe::internode::mpi_get_current_proc_rank_as_string()
             << " Iteration " << iter_ << ", loss = " << smoothed_loss_;
#else
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", loss = " << smoothed_loss_;
#endif
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

#ifdef CAFFE_PER_LAYER_TIMINGS
      PrintTimers(false);
      ResetTimers();
//      MLSL::print_mlsl_time();
#endif

    }

    iter_timer.Start();

#ifdef ADAPTIVE_BATCH
// comm_timer.Start();
    if (!param().disabled_update()) {
      if((iter_ > 0)
        && ((iter_ % batch_apply_iter) == 0)) {
        // && !all_reduce_onceHistory) {
          batch_h_update = true; // false: force no AllReduce(history)
        DLOG(INFO) << "ApplyUpdate(History) called. ";
        // all_reduce_onceHistory = true;
      }
      else { batch_h_update = false; }
      ApplyUpdate(batch_h_update);
    }
// comm_time += comm_timer.MilliSeconds();
#else
    // if((iter_ > 0)
    //     && ((iter_ % batch_ongradients_iter) == 0)) {
      for (int i = 0; i < callbacks_.size(); ++i) {
        callbacks_[i]->on_gradients_ready();
      }
    // }

    if (!param().disabled_update()) {
        ApplyUpdate();
        #if 0
        DLOG(INFO) << "ApplyUpdate(History) called. ";
        batch_h_update = false; // true;
        ApplyUpdate(batch_h_update);
        #endif
     }
#endif
    
    temp_time = 0;
    temp_time += iter_timer.MilliSeconds();
    iter_time += temp_time;
    comm_time += temp_time;
    total_time += temp_time;

#ifdef CAFFE_PER_LAYER_TIMINGS
    if (MLSL::GetNodeId() == 0)
        LOG(INFO) << "iter " << iter_ << ", forward_backward_update_time: " << iter_time << " ms";
#endif

#ifdef ADAPTIVE_BATCH
  #ifdef USE_MPI
    if(rank == 0)
      LOG(INFO) << "iter " << iter_ << ", forward_backward_update_time: " << iter_time << " ms";
      LOG(INFO) << "iter " << iter_ << ", communication_time: " << comm_time << " ms";
      LOG(INFO) << "iter " << iter_ << ", total_time: " << total_time << " ms";

  #endif
#endif

    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;
#ifdef ADAPTIVE_BATCH
    --batch_iter_count;
    lastLoss = loss;
#endif
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
  }

#ifdef CAFFE_PER_LAYER_TIMINGS
  ResetTimers();
  PrintTimers(true);
#endif

}

#ifdef CAFFE_PER_LAYER_TIMINGS

template <typename Dtype>
void Solver<Dtype>::InitTimers() {

  int layer_count = net_->layers().size();

  this->forward_time_per_layer.resize(layer_count, 0.0);
  this->backward_time_per_layer.resize(layer_count, 0.0);
  this->update_time_per_layer.resize(layer_count, 0.0);

  this->forward_time_per_layer_total.resize(layer_count, 0.0);
  this->backward_time_per_layer_total.resize(layer_count, 0.0);
  this->update_time_per_layer_total.resize(layer_count, 0.0);
}

template <typename Dtype>
void Solver<Dtype>::ResetTimers() {

  std::transform(this->forward_time_per_layer_total.begin(),
                 this->forward_time_per_layer_total.end(),
                 this->forward_time_per_layer.begin(),
                 this->forward_time_per_layer_total.begin(),
                 std::plus<int>());

  std::transform(this->backward_time_per_layer_total.begin(),
                 this->backward_time_per_layer_total.end(),
                 this->backward_time_per_layer.begin(),
                 this->backward_time_per_layer_total.begin(),
                 std::plus<int>());

  std::transform(this->update_time_per_layer_total.begin(),
                 this->update_time_per_layer_total.end(),
                 this->update_time_per_layer.begin(),
                 this->update_time_per_layer_total.begin(),
                 std::plus<int>());

  std::fill(this->forward_time_per_layer.begin(), this->forward_time_per_layer.end(), 0);
  std::fill(this->backward_time_per_layer.begin(), this->backward_time_per_layer.end(), 0);
  std::fill(this->update_time_per_layer.begin(), this->update_time_per_layer.end(), 0);
}

template <typename Dtype>
void Solver<Dtype>::PrintTimers(bool printTotal) {

#ifdef USE_MPI
    if (caffe::internode::mpi_get_current_proc_rank())
        return;
#endif

#ifdef USE_MLSL
    if (MLSL::GetNodeId())
       return;
#endif

    LOG(WARNING) << std::endl;
    LOG(WARNING) << "####################################################";

    std::vector<double>& forward_timers = printTotal ? forward_time_per_layer_total : forward_time_per_layer;
    std::vector<double>& backward_timers = printTotal ? backward_time_per_layer_total : backward_time_per_layer;
    std::vector<double>& update_timers = printTotal ? update_time_per_layer_total : update_time_per_layer;
    std::string prefix = printTotal ? "TOTAL " : "DELTA ";

    double forward_time = std::accumulate(forward_timers.begin(), forward_timers.end(), 0) / 1000;
    LOG(WARNING) << prefix << "FORWARD TIME: " << forward_time << " ms";
    for (int layer_idx = 0; layer_idx < net_->layers().size(); layer_idx++) {
        LOG(WARNING) << "LAYER-" << layer_idx << " "
                     << net_->layers()[layer_idx]->type()
                     << ": forward_time: " << forward_timers[layer_idx] / 1000 << " ms";
    }
    LOG(WARNING) << std::endl;

    double backward_time = std::accumulate(backward_timers.begin(), backward_timers.end(), 0) / 1000;
    LOG(WARNING) << prefix << "BACKWARD TIME: " << backward_time << " ms";
    for (int layer_idx = 0; layer_idx < net_->layers().size(); layer_idx++) {
        LOG(WARNING) << "LAYER-" << layer_idx << " "
                     << net_->layers()[layer_idx]->type()
                     << ": backward_time: " << backward_timers[layer_idx] / 1000 << " ms";
    }
    LOG(WARNING) << std::endl;

    double update_time = std::accumulate(update_timers.begin(), update_timers.end(), 0) / 1000;
    LOG(WARNING) << prefix << "UPDATE TIME: " << update_time << " ms";
    for (int layer_idx = 0; layer_idx < net_->layers().size(); layer_idx++) {
        LOG(WARNING) << "LAYER-" << layer_idx << " "
                     << net_->layers()[layer_idx]->type()
                     << ": update_time: " << update_timers[layer_idx] / 1000 << " ms";
    }
    LOG(WARNING) << std::endl;

    LOG(WARNING) << prefix << "TIME (F+B+U): " << (forward_time + backward_time + update_time) / 1000 << " sec";
    LOG(WARNING) << "####################################################";
    LOG(WARNING) << std::endl;
}

#endif /* CAFFE_PER_LAYER_TIMINGS */

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  PERFORMANCE_INIT_MONITOR();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);
  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);

    UpdateSmoothedLoss(loss, start_iter, average_loss);

#ifdef USE_MPI
    LOG(INFO) << caffe::internode::mpi_get_current_proc_rank_as_string()
              << " Iteration " << iter_ << ", loss = " << smoothed_loss_;
#else
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
#endif
  }

#if !defined(USE_MPI) && !defined(USE_MLSL) // in multinode last test must be done after weights update
  if (param_.test_interval() && iter_ % param_.test_interval() == 0)
    TestAll();
#endif

  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }

    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO)     << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
#ifdef USE_MLSL
    MPI_Allreduce(MPI_IN_PLACE, &loss, 1, sizeof(Dtype) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    loss /= (param_.test_iter(test_net_id) * MLSL::GetNumNodes());
    if(MLSL::GetNodeId() == 0) LOG(INFO) << "Test loss: " << loss;
#else /* !USE_MLSL */
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
#endif /* USE_MLSL */
  }
#ifdef USE_MLSL
  MPI_Allreduce(MPI_IN_PLACE, test_score.data(), test_score.size(), sizeof(Dtype) == 4 ? MPI_FLOAT : MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if(MLSL::GetNodeId() == 0)
#endif /* USE_MLSL */
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
#ifdef USE_MLSL
    const Dtype mean_score = test_score[i] / (param_.test_iter(test_net_id) * MLSL::GetNumNodes());
#else /* !USE_MLSL */
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
#endif /* USE_MLSL */
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());

#ifdef USE_MLSL
  if(MLSL::GetNodeId() != 0) return;
#endif /* USE_MLSL */

  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe
