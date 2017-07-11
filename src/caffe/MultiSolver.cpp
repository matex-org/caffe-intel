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

#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <vector>
#include "caffe/internode/tree_cluster.hpp"
#include "caffe/MultiSolver.hpp"

namespace caffe {

template <typename Dtype>
MultiSolver<Dtype>::MultiSolver(shared_ptr<Solver<Dtype> > root_solver)
  : root_solver_(root_solver)
  , iter_size(root_solver->param().iter_size()) {
  root_solver->set_forward_backward(
    boost::bind(&MultiSolver<Dtype>::ForwardBackward, this));
}

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackwardImpl(bool first, bool last) {
  Dtype loss = 0;
  Net<Dtype>& net = *root_solver_->net();

#ifdef CAFFE_PER_LAYER_TIMINGS
  Timer& timer = root_solver_->timer;
  std::vector<double>& forward_time_per_layer = root_solver_->forward_time_per_layer;
  std::vector<double>& backward_time_per_layer = root_solver_->backward_time_per_layer;
  std::vector<double>& update_time_per_layer = root_solver_->update_time_per_layer;
#endif

  for (int i = 0; i < net.layers().size(); ++i) {

#ifdef CAFFE_PER_LAYER_TIMINGS
    timer.Start();
#endif

    if (first) {
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_start(i);
      }
    }

    shared_ptr<Layer<Dtype> > layer = net.layers()[i];

    for (int bottom_idx = 0; bottom_idx < net.bottom_vecs()[i].size(); bottom_idx++)
        LOG_BLOB(layer, net.bottom_vecs()[i][bottom_idx], data, bottom_idx, "fprop: input data values:");


    for (int param_idx = 0; param_idx < layer->blobs().size(); param_idx++) {
          LOG_BLOB(layer, layer->blobs()[param_idx], data, param_idx, "fprop: weights:");
    }

    Dtype layer_loss = root_solver_->net()->ForwardFromTo(i, i);

    for (int top_idx = 0; top_idx < net.top_vecs()[i].size(); top_idx++)
        LOG_BLOB(layer, net.top_vecs()[i][top_idx], data, top_idx, "fprop: output data values:");

    loss += layer_loss;

    DLOG(WARNING) << "iter " << root_solver_->iter() 
                  << ", layer_id " << i
                  << ", layer_type " << net.layers()[i]->type()
                  << ", layer_loss " << layer_loss;

    if (last) {
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_forward_finished(i);
      }
    }

#ifdef CAFFE_PER_LAYER_TIMINGS
    forward_time_per_layer[i] += timer.MicroSeconds();
#endif

  }

  for (int i = net.layers().size() - 1; i >= 0; --i) {

#ifdef CAFFE_PER_LAYER_TIMINGS
    timer.Start();
#endif

    if (first) {
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_backward_start(i);
      }
    }

    shared_ptr<Layer<Dtype> > layer = net.layers()[i];

    for (int top_idx = 0; top_idx < net.top_vecs()[i].size(); top_idx++)
        LOG_BLOB(layer, net.top_vecs()[i][top_idx], diff, top_idx, "bprop: input diff values:");

    root_solver_->net()->BackwardFromTo(i, i);

    for (int bottom_idx = 0; bottom_idx < net.bottom_vecs()[i].size(); bottom_idx++)
        LOG_BLOB(layer, net.bottom_vecs()[i][bottom_idx], diff, bottom_idx, "bprop: output diff values:");


    for (int param_idx = 0; param_idx < layer->blobs().size(); param_idx++) {
        LOG_BLOB(layer, layer->blobs()[param_idx], diff, param_idx, "bprop: delwt:");
    }


#ifdef CAFFE_PER_LAYER_TIMINGS
    backward_time_per_layer[i] += timer.MicroSeconds();
    timer.Start();
#endif

    if (last) {
      for (int j = 0; j < callbacks_.size(); ++j) {
        callbacks_[j]->on_gradients_ready(i);
      }
    }

#ifdef CAFFE_PER_LAYER_TIMINGS
    update_time_per_layer[i] += timer.MicroSeconds();
#endif

  }

  DLOG(WARNING) << "iter " << root_solver_->iter() << ", loss " << loss;

  return loss;
}

template <typename Dtype>
Dtype MultiSolver<Dtype>::ForwardBackward() {
  Dtype loss = 0;
  for (int i = 0; i < iter_size; ++i) {
    loss += ForwardBackwardImpl(
      (i == 0), (i + 1 == iter_size));
  }
  return loss / iter_size;
}

template <typename Dtype>
void MultiSolver<Dtype>::Solve() {
  root_solver_->Solve();
}

INSTANTIATE_CLASS(MultiSolver);

}  // namespace caffe

