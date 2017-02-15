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
  LOG(INFO) << "Caches " << param.data_param().cache_size();
  cache_size_ = param.data_param().cache_size();
  if(cache_size_)
  {
    caches_ = new cache[cache_size_];
  }
  for(int i = cache_size_, j=0; i > 0; i--, j++)
  {
    caches_[i-1].current_shuffle_count_ = 0;
    caches_[i-1].size_ = param.data_param().cache(j).size();
    caches_[i-1].eviction_rate_ = param.data_param().cache(j).eviction_rate();
    caches_[i-1].refill_policy = &BasePrefetchingDataLayer<Dtype>::rate_replace_policy;
  #ifndef KNL
    if(param.data_param().cache(j).type() == CacheParameter::HEAP)
      caches_[i-1].cache_ = new Batch<Dtype>[caches_[i-1].size_];
    else
    {
      LOG(INFO) << "Cache Type not supported";
      exit(1);
    }
  #else
    if(param.data_param().cache(j).type() == CacheParameter::HBM)
    {
      void * ptr = hbw_malloc(sizeof(Batch<Dtype>)*caches_[i-1].size_);
      caches_[i-1].cache_ = new (ptr) Batch<Dtype>[caches_[i-1].size_];
    }
    else if(param.data_param().cache(j).type() == CacheParameter::HEAP)
    {
      caches_[i-1].cache_ = new Batch<Dtype>[caches_[i-1].size_];
    }
    else
    {
      LOG(INFO) << "Cache Type not supported";
      exit(1);
    }
  #endif
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
    for (int j = 0; j < caches_[i].size_; ++j) {
      caches_[i].cache_[j].data_.mutable_cpu_data();
      if (this->output_labels_) {
        caches_[i].cache_[j].label_.mutable_cpu_data();
      }
    }
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
      for (int j = 0; j < caches_[i].size_; ++j) {
        caches_[i].cache_[j].data_.mutable_gpu_data();
        if (this->output_labels_) {
          caches_[i].cache_[j].label_.mutable_gpu_data();
        }
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();

  for (int i = 0; i < cache_size_; ++i) {
    for (int j = 0; j < caches_[i].size_; ++j) {
      load_batch(&caches_[i].cache_[j]);
      caches_[i].cache_full_.push(&caches_[i].cache_[j]);
    }
  }
  // Only if GPU mode on then we use background threads
  if (Caffe::mode() == Caffe::GPU) {
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

  try {
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
      load_batch(batch);
      prefetch_full_.push(batch);
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template<typename Dtype>
void BasePrefetchingDataLayer<Dtype>::refill_cache(int current_cache)
{
  Batch<Dtype>* batch;
  int next_cache = current_cache+1;
  for(int i=0; i< caches_[current_cache].size_; i++)
  {
    //LOG(INFO) << position;
    batch = caches_[next_cache].cache_full_.pop("Data layer cache queue empty");
    caches_[current_cache].cache_[i].data_.CopyFrom( batch->data_ );
    caches_[current_cache].cache_[i].label_.CopyFrom( batch->label_ );
    caches_[current_cache].cache_full_.push(&caches_[next_cache].cache_[i]);
  }
}

template<typename Dtype>
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

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::rate_replace_policy(int next_cache)
{
 
  LOG(INFO) << "replace " << caches_[next_cache-1].current_shuffle_count_ << " " << caches_[next_cache-1].cache_full_.size();

  if(caches_[next_cache-1].current_shuffle_count_ < caches_[next_cache-1].eviction_rate_)
  {
    LOG(INFO) << "Shuffling Level " << next_cache-1 << " " << caches_[next_cache-1].size_;
    caches_[next_cache-1].current_shuffle_count_++;
    for(int i=0; i< caches_[next_cache-1].size_; i++)
    {
      for(int j=0; j< caches_[next_cache-1].cache_[i].data_.shape(0); j++)
      {
        shuffle_cache(&caches_[next_cache-1].cache_[i], j, &caches_[next_cache-1].cache_[randomGen(caches_[next_cache-1].size_)], randomGen(caches_[next_cache-1].cache_[i].data_.shape(0)));
      }
      caches_[next_cache-1].cache_full_.push(&caches_[next_cache-1].cache_[i]);
    }
  }
  else if(next_cache == cache_size_) //Last level -> refill
  {
    LOG(INFO) << "Refilling last level " << next_cache-1 << " " << caches_[next_cache-1].size_;
    caches_[next_cache-1].current_shuffle_count_=0;
    for (int i = 0; i < caches_[next_cache-1].size_; ++i)
    {
      load_batch(&caches_[next_cache-1].cache_[i]);
      caches_[next_cache-1].cache_full_.push(&caches_[next_cache-1].cache_[i]);
    }
  }
  else
  {
    caches_[next_cache-1].current_shuffle_count_=0;
    //Refill higher levels
    LOG(INFO) << "Recurse level " << next_cache << " " << caches_[next_cache].size_;
    LOG(INFO) << "Recurse Queue Size " << caches_[next_cache].current_shuffle_count_ << " " << caches_[next_cache].cache_full_.size();
    if(caches_[next_cache].cache_full_.size() == 0) //empty cache
      (this->*(caches_[0].refill_policy))(next_cache+1);
    LOG(INFO) << "Refilling level " << next_cache-1 << " " << caches_[next_cache-1].size_;
    //for (int i = 0; i < caches_[next_cache-1].size_; ++i)
    {
      refill_cache(next_cache-1);
    }
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch;
  /*if(cache_size_.size() > 0 && cache_size_[0])
  {
    if(cache_full_[0].size() == 0)
    {
      int accuracySize = historical_accuracy.size();
      if( ignoreAccuracy || accuracySize < 2 || historical_accuracy[accuracySize-1] > historical_accuracy[accuracySize-2])
      {
        //for(int i=0; i< accuracySize; i++)
        //  LOG(INFO) << "ACC" << historical_accuracy[i];
        
        //LOG(INFO) << "Shuffling Cache";
        for(int i=0; i< cache_size_[0]; i++)
        {
          for(int j=0; j< cache_[0][i].data_.shape(0); j++)
          {
              shuffle_cache(&cache_[0][i], j, &cache_[0][randomGen(cache_size_[0])], randomGen(cache_[0][i].data_.shape(0)));
          }
          cache_full_[0].push(&cache_[0][i]);
        }
        //LOG(INFO) << "Shuffling Cache Done";
      }
      else
      {
        ignoreAccuracy = true;
        LOG(INFO) << "Refilling Cache";
        for (int i = 0; i < cache_size_[0]; ++i)
        {
          load_batch(&cache_[0][i]);
          cache_full_.push(&cache_[0][i]);
        }
        //LOG(INFO) << "Refilling Cache Done";
      }
      //Don't forget prefetch_free_
    }
    batch = cache_full_[0].pop("Data layer cache queue empty");
  }*/
  if(cache_size_)
  {
    if(caches_[0].cache_full_.size() == 0) //empty cache
    {
        (this->*(caches_[0].refill_policy))(1);
    }
    batch = caches_[0].cache_full_.pop("Data layer cache queue empty");
  }
  else
  {
    //int accuracySize = historical_accuracy.size();
    //for(int i=0; i< accuracySize; i++)
    //  LOG(INFO) << "ACC" << historical_accuracy[i];
    // Here for CPU we do transformation
    if (Caffe::mode() == Caffe::CPU) {
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
  if(cache_size_ == 0 || caches_[0].size_ == 0)
    prefetch_free_.push(batch);

  // TODO: Consider prefetch_data_array and prefetch_label_array
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
