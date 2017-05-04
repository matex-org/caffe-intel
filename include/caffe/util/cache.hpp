#ifndef CAFFE_CACHE_HPP_
#define CAFFE_CACHE_HPP_

#include <vector>
#include <iostream>
#include <fstream>
#include <boost/atomic.hpp>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
class BasePrefetchingDataLayer;

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
  //class sync;
  //shared_ptr<sync> sync_var;
  //void lock();
  //void unlock();
};

template <typename Dtype>
struct PopBatch 
{
  Batch<Dtype>* batch;
  volatile bool * dirty;
};

template <typename Dtype>
class Cache
{
  public:
  volatile bool * dirty;
  class Cache * next;
  class Cache * prev;
  string disk_location;
  bool prefetch;
  volatile bool full_replace;
  int size;
  int refill_start;
  mutable boost::atomic<int> used;
  int eviction_rate;
  int current_shuffle_count;
  int last_i;
  int slot;
  bool ignoreAccuracy;
  void rate_replace_policy(int next_cache);
  void local_rate_replace_policy(int next_cache);
  void (Cache<Dtype>::*refill_policy)(int);  
  void (Cache<Dtype>::*local_refill_policy)(int);  
  BasePrefetchingDataLayer<Dtype> * data_layer;
  //void (BasePrefetchingDataLayer<Dtype>::*refill_policy)(int);  
  //void (BasePrefetchingDataLayer<Dtype>::*thread_refill_policy)(int);  
  //void (Cache<Dtype>::*refill_policy)(Cache<Dtype> * next_cache);  
  virtual void create( void * ptr, bool * ptr2, bool thread_safe ) { };
  virtual bool empty() { return false; };
  virtual PopBatch<Dtype> pop() { PopBatch<Dtype> nothing; return nothing; };
  virtual void shuffle (){}
  virtual void fill(bool in_cache) {};
  virtual void refill(Cache<Dtype> * next_cache) {};
  virtual void reshape(vector<int> * top_shape, vector<int> * label_shape) {};
  virtual void mutate_data(bool labels) {};
};


template <typename Dtype>
class MemoryCache : public Cache <Dtype>
{
  public:
  Batch<Dtype> * cache;
  BlockingQueue<Batch<Dtype>*> cache_full;
  void shuffle_cache(Batch<Dtype>* batch1, int batchPos1, Batch<Dtype>*  batch2, int batchPos2);
  virtual void create( void * ptr, bool * ptr2,bool thread_safe );
  virtual bool empty();
  virtual PopBatch<Dtype> pop();
  virtual void shuffle ();
  virtual void fill(bool in_cache);
  virtual void refill(Cache<Dtype> * next_cache);
  virtual void reshape(vector<int> * top_shape, vector<int> * label_shape);
  virtual void mutate_data(bool labels);
};

template <typename Dtype>
class DiskCache : public Cache <Dtype>
{
  public:
  
  //File stream
  bool open;
  fstream cache;
  fstream cache_read;
  Batch<Dtype> * cache_buffer;
  Batch<Dtype> * cache_read_buffer;
  unsigned int current_offset;
  void shuffle_cache(int batch1, int batchPos1, int  batch2, int batchPos2, int image_count, int data_count, int label_count);
  virtual void create( void * ptr, bool * ptr2, bool thread_safe);
  virtual bool empty();
  virtual PopBatch<Dtype> pop();
  virtual void fill(bool in_cache);
  virtual void shuffle ();
  virtual void refill(Cache<Dtype> * next_cache);
  virtual void reshape(vector<int> * top_shape, vector<int> * label_shape);
  virtual void mutate_data(bool labels);
};

}
#endif
