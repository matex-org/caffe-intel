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

#include <unistd.h> 
#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#ifdef KNL
#include <hbwmalloc.h>
#endif

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_(){
  //LOG(INFO) << "Cache size" << param.data_param().cache_size(0);
  cache_size_ = param.data_param().cache_size();
  LOG(INFO) << "Caches " << cache_size_;
  prefetch=false;
  if(cache_size_)
  {
    #ifdef KNL
      void * ptr = hbw_malloc(sizeof(Cache<Dtype> *) * cache_size_);
      caches_ = new (ptr) Cache<Dtype> * [cache_size_];
    #else  
      caches_ = new Cache<Dtype> * [cache_size_];
    #endif

  }
  for(int i = cache_size_, j=0; i > 0; i--, j++)
  {
    bool thread_safe = param.data_param().cache(j).thread_safe();
    
    if(thread_safe)
      prefetch = true;

    if(param.data_param().cache(j).type() == CacheParameter::HEAP)
    {
      caches_[i-1] = new MemoryCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      caches_[i-1]->create( new Batch<Dtype>[caches_[i-1]->size], new bool[caches_[i-1]->size], thread_safe );
    }
  #ifdef KNL
    else if(param.data_param().cache(j).type() == CacheParameter::HBM)
    {
      void * ptr = hbw_malloc(sizeof(MemoryCache<Dtype>));
      caches_[i-1] = new (ptr) MemoryCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      ptr = hbw_malloc(sizeof(Batch<Dtype>)*caches_[i-1]->size);
      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
      caches_[i-1]->create( new (ptr) Batch<Dtype>[caches_[i-1]->size], ptr2, thread_safe );
    }
    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
    {
      void * ptr = hbw_malloc(sizeof(DiskCache<Dtype>));
      caches_[i-1] = new (ptr) DiskCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      ptr = hbw_malloc(sizeof(Batch<Dtype>)*2);
      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
      caches_[i-1]->create( new (ptr) Batch<Dtype>[2], ptr2, thread_safe );
    }
  #else
    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
    {
      caches_[i-1] = new DiskCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      caches_[i-1]->create( new Batch<Dtype>[2], new bool[caches_[i-1]->size], thread_safe );
    }
  #endif
    else
    {
      LOG(INFO) << "Cache Type not supported";
      exit(1);
    }
    if(i-1==cache_size_-1)
      caches_[i-1]->next = NULL; 
    else  
      caches_[i-1]->next = caches_[i]; 
    
    caches_[i-1]->data_layer = this;
    caches_[i-1]->used = caches_[i-1]->size;
    caches_[i-1]->refill_start = 0;
    caches_[i-1]->current_shuffle_count = 0;
    caches_[i-1]->eviction_rate = param.data_param().cache(j).eviction_rate();
    //caches_[i-1]->thread_refill_policy = Cache<Dtype>::thread_rate_replace_policy;
    caches_[i-1]->refill_policy = &Cache<Dtype>::rate_replace_policy;
    caches_[i-1]->local_refill_policy = &Cache<Dtype>::local_rate_replace_policy;
    caches_[i-1]->disk_location = param.data_param().cache(j).disk_location();
    LOG(INFO) << "Cacher " <<  param.data_param().cache(j).disk_location() << " " << caches_[i-1]->disk_location;
  }
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_free_.push(&prefetch_[i]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  randomGen.Init();
  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    prefetch_[i].data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i].label_.mutable_cpu_data();
    }
  }
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->mutate_data(this->output_labels_);
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < PREFETCH_COUNT; ++i) {
      prefetch_[i].data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i].label_.mutable_gpu_data();
      }
    }
    for (int i = 0; i < cache_size_; ++i) {
      for (int j = 0; j < caches_[i]->size; ++j) {
        caches_[i]->cache[j].data_.mutable_gpu_data();
        if (this->output_labels_) {
          caches_[i]->cache[j].label_.mutable_gpu_data();
        }
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();

  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->fill(false);
  }
  // Only if GPU mode on then we use background threads
  //if (Caffe::mode() == Caffe::GPU) {
  if (prefetch) {
    StartInternalThread();
  }
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  /*try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }*/
  while (!must_stop()) {
    if(cache_size_)
    {
      //usleep(1000000);
      for(int i=cache_size_-1; i>= 0; i--)
      //for(int i=0; i < cache_size_; i++)
      {
        if(caches_[i]->prefetch)
          (caches_[i]->*(caches_[i]->refill_policy))(1);
      }
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

// TODO: Make it properlly implemented/integrated with above solution
template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::GetBatch() {
  try {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch, false);
      prefetch_full_.push(batch);
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

/*template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::thread_rate_replace_policy(int next_cache)
{
 
  if(next_cache == cache_size_) //Last level -> refill
  {
    caches_[next_cache-1]->fill(this);
  }
  else
  {
    (this->*(caches_[next_cache]->thread_refill_policy))(next_cache+1);
    
    caches_[next_cache-1]->refill(caches_[next_cache]);
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::rate_replace_policy(int next_cache)
{
 
  //LOG(INFO) << "replace " << caches_[next_cache-1]->current_shuffle_count; // << " " << caches_[next_cache-1]->cache_full.size();

  if(caches_[next_cache-1]->current_shuffle_count < caches_[next_cache-1]->eviction_rate)
  {
    LOG(INFO) << "Shuffling Level " << next_cache-1 << " " << caches_[next_cache-1]->size;
    caches_[next_cache-1]->current_shuffle_count++;
    caches_[next_cache-1]->shuffle(this);
  }
  else if(next_cache == cache_size_) //Last level -> refill
  {
    LOG(INFO) << "Refilling last level " << next_cache-1 << " " << caches_[next_cache-1]->size;
    caches_[next_cache-1]->current_shuffle_count=0;
    caches_[next_cache-1]->fill(this);
    
  }
  else
  {
    caches_[next_cache-1]->current_shuffle_count=0;
    //Refill higher levels
    //LOG(INFO) << "Recurse level " << next_cache << " " << caches_[next_cache]->size;
    //LOG(INFO) << "Recurse Queue Size " << caches_[next_cache]->current_shuffle_count; // << " " << caches_[next_cache]->cache_full.size();
    if(caches_[next_cache]->empty() ) //empty cache
      (this->*(caches_[next_cache]->refill_policy))(next_cache+1);
    
    LOG(INFO) << "Refilling level " << next_cache-1 << " " << caches_[next_cache-1]->size;
    caches_[next_cache-1]->refill(caches_[next_cache]);
  }
}*/

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype> * batch;
  PopBatch<Dtype> pop_batch;
  if(cache_size_)
  {
    if(!caches_[0]->prefetch && caches_[0]->empty()) //empty cache
    {
      //LOG(INFO) << "Local Refill "; 
      (caches_[0]->*(caches_[0]->local_refill_policy))(1);
    }
    pop_batch = caches_[0]->pop();
    batch = pop_batch.batch;
  }
  else
  {
    //int accuracySize = historical_accuracy.size();
    //for(int i=0; i< accuracySize; i++)
    //  LOG(INFO) << "ACC" << historical_accuracy[i];
    // Here for CPU we do transformation
    //if (Caffe::mode() == Caffe::CPU) {
    if (!prefetch) {
      this->GetBatch();
    }
    batch = prefetch_full_.pop("Prefetch cache queue empty");
  }
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  if(cache_size_)
    *pop_batch.dirty = true;
  if(cache_size_ == 0 || caches_[0]->size == 0)
    prefetch_free_.push(batch);

  // TODO: Consider prefetch_data_array and prefetch_label_array
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
