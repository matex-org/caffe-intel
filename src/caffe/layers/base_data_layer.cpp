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

#ifdef USE_DEEPMEM
#include <unistd.h>
#endif  
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

#ifdef USE_MLSL
using namespace MLSL;
#endif /* USE_MLSL */

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


#ifdef USE_MLSL

  DataType dt = (sizeof(Dtype) == 4)? DT_FLOAT : DT_DOUBLE;
  ComputeOpRegInfo *myRegInfo;
  myRegInfo = new ComputeOpRegInfo(COMP_OP_TYPE_DATA);
  myRegInfo->SetName(this->layer_param_.name().c_str());
  myRegInfo->AddOutputFeatureMap(top[0]->channels(), top[0]->width()*top[0]->height(), dt);
  myRegInfo->AddOutputFeatureMap(top[1]->channels(), top[1]->width()*top[1]->height(), dt);
  
  myRegInfo->Validate();
  this->layerOp = new ComputeOp(myRegInfo, caffe::internode::data_parallelism);
  delete myRegInfo;

#endif /* USE_MLSL */

}

#ifdef USE_MLSL

template <typename Dtype>
void BaseDataLayer<Dtype>::pack_buffer(FeatureMap *fm, Dtype *to, const Dtype *from) {
      for (int i = 0; i < fm->NumPackBlocks(); i++) {
          BlockInfo * bi = fm->GetPackBlock(i);
          int bMBLen = bi->MBLen();
          int bMBStart = bi->MBStart();
          int bFMLen = bi->FMLen();
          int bFMStart = bi->FMStart();
          Dtype *src = (Dtype*) from;
          Dtype *dst = (Dtype*) (to + bi->BufOffset());
          for (int mb = 0; mb < bMBLen; mb++) {
              for (int fm = 0; fm < bFMLen; fm++) {
                  for (int s = 0 ; s < bi->FMSize(); s++) {
                      dst[(fm*bMBLen + mb)*bi->FMSize() + s] =
                          src[((bMBStart+mb)*bFMLen + bFMStart+fm)*bi->FMSize() + s];
                  }
              }
          }
      }
  }

  template <typename Dtype>
  void BaseDataLayer<Dtype>::unpack_buffer(FeatureMap *fm, const Dtype *from, Dtype *to) {
      for (int i = 0; i < fm->NumUnpackBlocks(); i++) {
          BlockInfo * bi = fm->GetUnpackBlock(i);
          int bMBLen = bi->MBLen();
          int bMBStart = bi->MBStart();
          int bFMLen = bi->FMLen();
          int bFMStart = bi->FMStart();
          Dtype *dst = (Dtype*) to;
          Dtype *src = (Dtype*) (from + bi->BufOffset());
          for (int mb = 0; mb < bMBLen; mb++) {
              for (int fm = 0; fm < bFMLen; fm++) {
                  for (int s = 0 ; s < bi->FMSize(); s++) {
                    dst[((bMBStart+mb)*bFMLen + bFMStart+fm)*bi->FMSize() + s] = src[(fm*bMBLen + mb)*bi->FMSize() + s];
                  }
              }
          }
      }
  }

#endif /* USE_MLSL */

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_free_(), prefetch_full_() {
#ifdef USE_DEEPMEM
  //LOG(INFO) << "Cache size" << param.data_param().cache_size(0);
  const char* env_prefetch_count = std::getenv("ENV_PREFETCH_COUNT");
  const char* env_reuse_count = std::getenv("ENV_REUSE_COUNT");
  char *garbage1 = NULL, *garbage2 = NULL;
  prefetch_count =
    (env_prefetch_count != NULL) ? strtol(env_prefetch_count,&garbage1,0):PREFETCH_COUNT;
  LOG(INFO) << "Prefetch Count: " << prefetch_count;
  reuse_count = (env_reuse_count != NULL) ? strtol (env_reuse_count,&garbage2,0):0;
  LOG(INFO) << "Reuse Count: " << reuse_count;
   
  for(std::size_t i = 0; i < prefetch_count; ++i) {
    prefetch_.push_back(new Batch<Dtype>());
  }

  cache_size_ = param.data_param().cache_size();
  LOG(INFO) << "Caches " << cache_size_;
  prefetch=false;
  if(cache_size_)
  {
    //Allocate array to hold caches on heap or in hbm
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
    
    //If one cache is thread_save then set this global to turn the prefetcher
    //Thread
    if(thread_safe)
      prefetch = true;

    if(param.data_param().cache(j).type() == CacheParameter::HEAP)
    {
      //Create a new cache, set size, a dirty structure
      caches_[i-1] = new MemoryCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      caches_[i-1]->create( new Batch<Dtype>[caches_[i-1]->size], new bool[caches_[i-1]->size], thread_safe );
    }
  #ifdef KNL
    else if(param.data_param().cache(j).type() == CacheParameter::HBM)
    {
      //Use hbm to create a new cache, set size, and dirty structure
      void * ptr = hbw_malloc(sizeof(MemoryCache<Dtype>));
      caches_[i-1] = new (ptr) MemoryCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      ptr = hbw_malloc(sizeof(Batch<Dtype>)*caches_[i-1]->size);
      bool * ptr2 = (bool *)hbw_malloc(sizeof(bool)*caches_[i-1]->size);
      caches_[i-1]->create( new (ptr) Batch<Dtype>[caches_[i-1]->size], ptr2, thread_safe );
    }
    else if(param.data_param().cache(j).type() == CacheParameter::DISK)
    {
      //Use hbm to create a new cache, read/write bufs, set size, and dirty structure
      void * ptr = hbw_malloc(sizeof(DiskCache<Dtype>));
      caches_[i-1] = new (ptr) DiskCache<Dtype>;
      caches_[i-1]->size = param.data_param().cache(j).size();
      //Read/write buffer
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
   
    //Setup cache to point one level above 
    if(i-1==cache_size_-1)
      caches_[i-1]->next = NULL; 
    else  
      caches_[i-1]->next = caches_[i]; 
   
    // Pass data_layer (used for filling)
    caches_[i-1]->data_layer = this;
    //Initially needs to be filled
    caches_[i-1]->used = caches_[i-1]->size;
    caches_[i-1]->refill_start = 0;
    caches_[i-1]->current_shuffle_count = 0;
    caches_[i-1]->eviction_rate = param.data_param().cache(j).eviction_rate();
    caches_[i-1]->refill_policy = &Cache<Dtype>::rate_replace_policy;
    caches_[i-1]->local_refill_policy = &Cache<Dtype>::local_rate_replace_policy;
    caches_[i-1]->disk_location = param.data_param().cache(j).disk_location();
    LOG(INFO) << "Cacher " <<  param.data_param().cache(j).disk_location() << " " << caches_[i-1]->disk_location;
  }
  
  //Setup cache to point one level below
  for(int j=0; j < cache_size_; j++)
  {
    if(j==0)
      caches_[j]->prev = NULL; 
    else
      caches_[j]->prev = caches_[j-1]; 
  }

  if(cache_size_) {
    typedef MemoryCache<Dtype> MemCacheType;
    MemCacheType *memcache;

    if(cache_size_ && ((memcache = dynamic_cast<MemCacheType *>(caches_[0])))) {
      for (int i = 0; i < memcache->size ; ++i) {
        LOG(INFO) << "MEMCACHE size: " << memcache->size;
        memcache->cache[i].count = this->reuse_count;
        prefetch_free_.push(&memcache->cache[i]);
      }
     }
  } else {
#endif 

  for (int i = 0; i < PREFETCH_COUNT; ++i) {
    // prefetch_free_.push(&prefetch_[i]);
    prefetch_free_.push(prefetch_[i]);
  }
#ifdef USE_DEEPMEM
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
#ifdef USE_DEEPMEM
  randomGen.Init();
#endif 
  // for (int i = 0; i < PREFETCH_COUNT; ++i) {
  for (int i = 0; i < this->prefetch_count; ++i) {
    // prefetch_[i].data_.mutable_cpu_data();
    prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      // prefetch_[i].label_.mutable_cpu_data();
      prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifdef USE_DEEPMEM
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->mutate_data(this->output_labels_);
  }
#endif
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    // for (int i = 0; i < PREFETCH_COUNT; ++i) {
    for (int i = 0; i < this->prefetch_count; ++i) {
      // prefetch_[i].data_.mutable_gpu_data();
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        // prefetch_[i].label_.mutable_gpu_data();
        prefetch_[i]->label_.mutable_gpu_data();
      }
    }
// #ifdef USE_DEEPMEM
//     //Setup the caches data
//     for (int i = 0; i < cache_size_; ++i) {
//       for (int j = 0; j < caches_[i]->size; ++j) {
//         caches_[i]->cache[j].data_.mutable_gpu_data();
//         if (this->output_labels_) {
//           caches_[i]->cache[j].label_.mutable_gpu_data();
//         }
//       }
//     }
// #endif
  }
#endif

  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();

#ifdef USE_DEEPMEM
  for (int i = 0; i < cache_size_; ++i) {
    caches_[i]->fill(false);
  }
#endif

  // Only if GPU mode on then we use background threads
#ifdef USE_DEEPMEM
//If the global prefetch is set create a prefetch thread which is just below
  if (prefetch) {
#else
  if (Caffe::mode() == Caffe::GPU) {
#endif
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

#ifdef USE_DEEPMEM
 while (!must_stop()) {
    if(cache_size_)
    {
      for(int i=cache_size_-1; i>= 0; i--)
      {
        //If we handle the refilling apply the member pointer to the current
        //Cache class
        if(caches_[i]->prefetch)
          (caches_[i]->*(caches_[i]->refill_policy))(1);
      }
      DLOG(INFO) << "Prefetch_free_queue size before pop:" << prefetch_free_.size();
      Batch<Dtype>* batch = prefetch_free_.pop("DEEPMEMCACHE DataLayer(CH) Free Queue Empty");
      prefetch_full_.push(batch);
    }
  }
#else
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
#endif

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
#ifdef USE_DEEPMEM
      load_batch(batch, false);
#else
      load_batch(batch);
#endif
      prefetch_full_.push(batch);
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
#ifdef USE_DEEPMEM
  Batch<Dtype> * batch;
  PopBatch<Dtype> pop_batch;
  //If there are any caches
  if(cache_size_)
  {
    DLOG(INFO) << "Cache_size: " << cache_size_
               << " --cache used to fetch batch!";
    //Do we handle the refill on l1 cache?
    if(!caches_[0]->prefetch && caches_[0]->empty()) //empty cache
    {
      //LOG(INFO) << "Local Refill ";
      //Refill before poping using the policy we have
      (caches_[0]->*(caches_[0]->local_refill_policy))(1);
    }
    DLOG(INFO) << "L0 Cache SlotSize: " << caches_[0]->size
               << " Current SlotNo: " << caches_[0]->slot;
    pop_batch = caches_[0]->pop();
    batch = pop_batch.batch;
  }
  else //Use the original unmofified code to get a batch
  {
    DLOG(INFO) << " --cache not used to fetch batch!";
    //int accuracySize = historical_accuracy.size();
    //for(int i=0; i< accuracySize; i++)
    //  LOG(INFO) << "ACC" << historical_accuracy[i];
    // Here for CPU we do transformation
    //if (Caffe::mode() == Caffe::CPU) {
    if (!prefetch) {
      DLOG(INFO) << "FCPU_GetBatch called!!!";
      this->GetBatch();
    }
    batch = prefetch_full_.pop("Prefetch cache queue empty");
  }
#else
  // Here for CPU we do transformation
  if (Caffe::mode() == Caffe::CPU) {
    this->GetBatch();
  }
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
#endif

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
#ifdef USE_DEEPMEM
  if(cache_size_) // We finished copy the batch so mark it for replacement
    *pop_batch.dirty = true;
  //Use the orginal code if caches are turned off
  // if(cache_size_ == 0 || caches_[0]->size == 0)
    batch->count -= 1;
    if(batch->count > 0) {
      DLOG(INFO) << "Batch Reuse Count: " << batch->count;
      batch->dirty = false;
      prefetch_full_.push(batch);
    } else {
      batch->dirty = true;
      prefetch_free_.push(batch);
    }
    // prefetch_free_.push(batch);
#else
  // TODO: Consider prefetch_data_array and prefetch_label_array
  prefetch_free_.push(batch);
#endif
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe
