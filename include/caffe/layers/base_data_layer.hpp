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
#include <iostream>

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

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer;


template <typename Dtype>
class Cache
{
  public:
  int size;
  int eviction_rate;
  int current_shuffle_count;
  bool ignoreAccuracy;
  void (BasePrefetchingDataLayer<Dtype>::*refill_policy)(int);  
  //void (Cache<Dtype>::*refill_policy)(Cache<Dtype> * next_cache);  
  virtual void create( void * ptr ) { };
  virtual bool empty() { return false; };
  virtual Batch<Dtype> * pop() { return NULL; };
  virtual void shuffle (BasePrefetchingDataLayer<Dtype> * data_helper){}
  virtual void fill(BasePrefetchingDataLayer<Dtype> * filler) {};
  virtual void refill(Cache<Dtype> * next_cache) {};
  virtual void reshape(vector<int> * top_shape, vector<int> * label_shape) {};
  virtual void mutate_data(bool labels) {};
};

template <typename Dtype>
class MemoryCache : public Cache <Dtype>
{
  public:
  //Batch<Dtype> * cache_;
  //BlockingQueue<Batch<Dtype>*> cache_full;
  //virtual void shuffle();
  Batch<Dtype> * cache;
  BlockingQueue<Batch<Dtype>*> cache_full;
  void shuffle_cache(Batch<Dtype>* batch1, int batchPos1, Batch<Dtype>*  batch2, int batchPos2) {
    const int datum_channels = batch1->data_.shape(1);
    const int datum_height = batch1->data_.shape(2);
    const int datum_width = batch1->data_.shape(3);
    
    Dtype * data1 = batch1->data_.mutable_cpu_data();
    Dtype * data2 = batch2->data_.mutable_cpu_data();
    Dtype * label1 = batch1->label_.mutable_cpu_data();
    Dtype * label2 = batch2->label_.mutable_cpu_data();
    int offset1 = batch1->data_.offset(batchPos1);
    int offset2 = batch2->data_.offset(batchPos2);
    int top_index;
    data1+=offset1;
    data2+=offset2;

    int height = datum_height;
    int width = datum_width;

    int h_off = 0;
    int w_off = 0;
    std::swap(label1[batchPos1], label2[batchPos2]); 
    for (int c = 0; c < datum_channels; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          top_index = (c * height + h) * width + w;
          std::swap(data1[top_index], data2[top_index]);
        }
      }
    }
  }
  virtual void create( void * ptr )
  {
    cache = static_cast<Batch<Dtype> *> (ptr); 
  };
  virtual bool empty()
  { 
    return cache_full.size() == 0;
  };
  virtual Batch<Dtype> * pop()
  { 
    return cache_full.pop();
  }
  virtual void shuffle (BasePrefetchingDataLayer<Dtype> * data_helper)
  {
    for(int i=0; i< Cache<Dtype>::size; i++)
    {
      for(int j=0; j< cache[i].data_.shape(0); j++)
      {
        shuffle_cache(&cache[i], j, &cache[data_helper->randomGen(size)], data_helper->randomGen(cache[i].data_.shape(0)));
      }
      cache_full.push(&cache[i]);
    }
  }
  virtual void fill(BasePrefetchingDataLayer<Dtype> * filler)
  {
    for (int j = 0; j < Cache<Dtype>::size; ++j) {
      filler->load_batch(&cache[j]);
      cache_full.push(&cache[j]);
    }
  };
  virtual void refill(Cache<Dtype> * next_cache)
  {
    Batch<Dtype>* batch;
    for(int i=0; i< Cache<Dtype>::size; i++)
    {
      //LOG(INFO) << position;
      batch = next_cache->pop(); //->cache_full_.pop("Data layer cache queue empty");
      cache[i].data_.CopyFrom( batch->data_ );
      cache[i].label_.CopyFrom( batch->label_ );
      cache_full.push(&cache[i]);
    }
  }  
  virtual void reshape(vector<int> * top_shape, vector<int> * label_shape)
  {
    for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].data_.Reshape(*top_shape);
    }
    if (label_shape) {
      for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].label_.Reshape(*label_shape);
      }
    }
  }
  virtual void mutate_data(bool labels)
  {
    for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].data_.mutable_cpu_data();
    }
    if (labels) {
      for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].label_.mutable_cpu_data();
      }
    }
#ifndef CPU_ONLY
    for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].data_.mutable_gpu_data();
      }
    if (labels) {
      for(int i=0; i< Cache<Dtype>::size; i++) {
        cache[i].label_.mutable_gpu_data();
      }
    }
#endif
  }
};

template <typename Dtype>
class DiskCache : public Cache <Dtype>
{
  public:
  //File stream
  string disk_cache_location;
  unsigned int current_cache_offset;
  virtual void shuffle();
  virtual void refill(Cache<Dtype> * next_cache);
};

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
  virtual void Pass_Value_To_Layer(Dtype value, unsigned int position) {
    //LOG(INFO) << "Base Pass";
    //ignoreAccuracy_=false;
    historical_accuracy_.push_back(value);
  }

 protected:
  void refill_cache(int current_cache);
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  virtual void GetBatch();
  void rate_replace_policy(int next_cache);
  

  GenRandNumbers randomGen;
  Batch<Dtype> prefetch_[PREFETCH_COUNT];
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  
  Cache<Dtype> ** caches_;
  int cache_size_;
  vector<Dtype> historical_accuracy_;

  Blob<Dtype> transformed_data_;
  
  friend class Cache<Dtype>;
  friend class MemoryCache<Dtype>;
  friend class DiskCache<Dtype>;
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
