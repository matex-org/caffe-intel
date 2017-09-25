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

#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>
#ifdef USE_DEEPMEM
#include <iostream>
#include <fstream>
#include "caffe/util/cache.hpp"
#include <cstdlib>
#endif

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  // Data layers should be shared by multiple solvers in parallel
  virtual inline bool ShareInParallel() const { return true; }
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  
 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

#ifndef USE_DEEPMEM
template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};
#endif

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // Prefetches batches (asynchronously if to GPU memory)
  static const int PREFETCH_COUNT = 3;
#ifdef USE_DEEPMEM
  virtual void Pass_Value_To_Layer(Dtype value, unsigned int position) {
    //LOG(INFO) << "Base Pass";
    //ignoreAccuracy_=false;
    historical_accuracy_.push_back(value);
  }
  int cache_size_;
#endif

 protected:
#ifdef USE_DEEPMEM
  bool prefetch;
  void refill_cache(int current_cache);
#endif
  virtual void InternalThreadEntry();
#ifdef USE_DEEPMEM
  virtual void load_batch(Batch<Dtype>* batch, bool in_thread) = 0;
#else
  virtual void load_batch(Batch<Dtype>* batch) = 0;
#endif

  virtual void GetBatch();
#ifdef USE_DEEPMEM
  void rate_replace_policy(int next_cache);
  void thread_rate_replace_policy(int next_cache);

  GenRandNumbers randomGen;
  volatile long prefetch_count;
  volatile long reuse_count;
#endif
  // Batch<Dtype> prefetch_[PREFETCH_COUNT];
  std::vector<Batch<Dtype>* > prefetch_;
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;

  std::vector<PopBatch<Dtype>* > pop_prefetch_;
  BlockingQueue<PopBatch<Dtype> > pop_prefetch_free_;
  BlockingQueue<PopBatch<Dtype> > pop_prefetch_full_;

#ifdef USE_DEEPMEM
  Cache<Dtype> ** caches_;
  vector<Dtype> historical_accuracy_;
#endif

  Blob<Dtype> transformed_data_;

#ifdef USE_DEEPMEM
  friend class Cache<Dtype>;
  friend class MemoryCache<Dtype>;
  friend class DiskCache<Dtype>;
#endif
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
