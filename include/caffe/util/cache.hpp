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

//Pop Batch strucutre is like a batch but includes a pointer to dirty structure
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
  volatile bool * dirty; //Tells if cache position can be written over  
  class Cache * next; //The cache above me
  class Cache * prev; //The cache below me
  string disk_location; //File location for disk caching
  bool prefetch; //Is this cache replaced in the prefetcher thread
  volatile bool full_replace; //Has the whole cache been replaced
  int size; //Number of batches this cache stores
  int refill_start; 
  mutable boost::atomic<int> used; //Tells your dirty slot or how many slots are dirty
  int eviction_rate; //Reuse count
  int current_shuffle_count; //Increments when full_replace is true
  int last_i; //Stores the i for refill/fill/shuffle loops between function calls
  int slot;
  bool ignoreAccuracy;
  void rate_replace_policy(int next_cache); //Generic prefetch thread policy that replaces cache at the eviction rate
  void local_rate_replace_policy(int next_cache); //Same as above but inside of forward cpu
  void (Cache<Dtype>::*refill_policy)(int); //Function pointer to thread replacement policy
  void (Cache<Dtype>::*local_refill_policy)(int); //Function pointer to forward cpu replacement policy
  BasePrefetchingDataLayer<Dtype> * data_layer;
  
  //Inits Cache: ptr is the batch buffer memory, pt2 is the dirty bit memory, thread_safe tells if cache is on prefetch
  virtual void create( void * ptr, bool * ptr2, bool thread_safe ) { };
  virtual bool empty() { return false; };
  //Pops a batch from the cache -> includes ptr to dirty structure
  virtual PopBatch<Dtype> pop() { PopBatch<Dtype> nothing; return nothing; };
  virtual void shuffle (){}
  //Fills data from data_layer ptr -> in_cach
  virtual void fill(bool in_cache) {};
  //Refills from cache above you (basically next_cache is what uses though)
  virtual void refill(Cache<Dtype> * next_cache) {};
  virtual void reshape(vector<int> * top_shape, vector<int> * label_shape) {};
  virtual void mutate_data(bool labels) {};
};


template <typename Dtype>
class MemoryCache : public Cache <Dtype>
{
  public:

  //Batches memory 
  Batch<Dtype> * cache;
  //Queue for poping from
  BlockingQueue<Batch<Dtype>*> cache_full;
  
  //Swaps image in batch1 at batchPos1 with image in batch2 at batchPos2
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
