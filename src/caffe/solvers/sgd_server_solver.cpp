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

#include <vector>

#include "caffe/mpi.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
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

template <typename Dtype>
void SGDServerSolver<Dtype>::Init() {
  LOG(INFO) << "IN INIT";

  params_ = this->net_->learnable_params();
  size_ = total_size(params_);
  data_ = new Dtype[size_];
  diff_ = new Dtype[size_];

  apply_buffers(params_, data_, size_, copy);
  caffe::mpi::bcast(data_, size_, 0);
  caffe_set(size_, Dtype(0), diff_);
  apply_buffers(params_, data_, size_, replace_cpu);
  apply_buffers(params_, diff_, size_, replace_cpu_diff);

  int comm_rank = caffe::mpi::comm_rank();
  int comm_size = caffe::mpi::comm_size();
  if (comm_size-1 == comm_rank) {
    Server();
  }
}

template <typename Dtype>
void SGDServerSolver<Dtype>::ApplyUpdate() {
  DLOG(INFO) << "IN APPLY UPDATE";

  int comm_rank = caffe::mpi::comm_rank();
  int comm_size = caffe::mpi::comm_size();

#if 0
  ClipGradients();
  for (int i = 0; i < params_.size(); ++i) {
    caffe::mpi::send(params_[i]->cpu_diff(), params_[i]->count(), comm_size-1);
  }
  for (int i = 0; i < params_.size(); ++i) {
    caffe::mpi::recv(params_[i]->mutable_cpu_data(), params_[i]->count(), comm_size-1);
  }
#else
  caffe::mpi::send(diff_, size_, comm_size-1);
  caffe::mpi::recv(data_, size_, comm_size-1);
#endif
}

template <typename Dtype>
void SGDServerSolver<Dtype>::Server() {
  DLOG(INFO) << "SERVER STARTED";

  int comm_rank = caffe::mpi::comm_rank();
  int comm_size = caffe::mpi::comm_size();

  while (true) {
    for (int worker = 0; worker < comm_size-1; ++worker) {
#if 0
      for (int i = 0; i < params_.size(); ++i) {
        caffe::mpi::recv(params_[i]->mutable_cpu_diff(), params_[i]->count(), worker);
      }
      SGDSolver<Dtype>::ApplyUpdate();
      for (int i = 0; i < params_.size(); ++i) {
        caffe::mpi::send(params_[i]->cpu_data(), params_[i]->count(), worker);
      }
#else
      caffe::mpi::recv(diff_, size_, worker);
      SGDSolver<Dtype>::ApplyUpdate();
      caffe::mpi::send(data_, size_, worker);
#endif
    }
  }

}

INSTANTIATE_CLASS(SGDServerSolver);
REGISTER_SOLVER_CLASS(SGDServer);

}  // namespace caffe
