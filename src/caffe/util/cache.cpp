
#include <boost/thread.hpp>
#include <boost/atomic.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/util/cache.hpp"
#ifdef KNL
#include <hbwmalloc.h>
#endif

namespace caffe {
/*template<typename T>
class Batch<T>::sync {
 public:
  mutable boost::mutex mutex_;
  boost::condition_variable condition_;
};

template <typename Dtype>
void Batch<Dtype>::lock() {

  if(sync_var)
    sync_var->mutex_.lock();
}

template <typename Dtype>
void Batch<Dtype>::unlock() {

  if(sync_var)
    sync_var->mutex_.unlock();
}*/

template <typename Dtype>
void Cache<Dtype>::rate_replace_policy(int next_cache)
{
  if(current_shuffle_count < eviction_rate)
  {
    //LOG(INFO) << "Shuffling Level " << next_cache-1 << " " << size;
    shuffle();
    
    if(full_replace)
    {
      current_shuffle_count++;
      full_replace=0;
    }
  }
  else if(next == NULL) //Last level -> refill
  {
    //LOG(INFO) << "Filling Level " << next_cache-1 << " " << size;
    fill(true);
    if(full_replace)
    {
      current_shuffle_count=0;
      full_replace=0;
    }
  }
  else
  {
    //Refill higher levels
    //if(next->empty() ) //empty cache
    //  (next->*(next->refill_policy))(next_cache+1);
    
    //LOG(INFO) << "Refilling Level " << next_cache-1 << " " << size;
    refill(next);
    
    if(full_replace)
    {
      current_shuffle_count=0;
      full_replace=0;
    }
  }
}
template <typename Dtype>
void Cache<Dtype>::local_rate_replace_policy(int next_cache)
{
  if(current_shuffle_count < eviction_rate)
  {
    LOG(INFO) << "Shuffling Level " << next_cache-1 << " " << size;
    shuffle();
    if(full_replace)
    {
      current_shuffle_count++;
      full_replace=0;
    }
  }
  else if(next == NULL) //Last level -> refill
  {
    fill(false);
    if(full_replace)
    {
      current_shuffle_count=0;
      full_replace=0;
    }
  }
  else
  {
    LOG(INFO) << "Refilling Level " << next_cache-1 << " " << size;
    //Refill higher levels
    if(!next->prefetch && next->empty() ) //empty cache
      (next->*(next->local_refill_policy))(next_cache+1);
    
    refill(next);
    
    if(full_replace)
    {
      current_shuffle_count=0;
      full_replace=0;
    }
  }
}
template <typename Dtype>
void MemoryCache<Dtype>::shuffle_cache(Batch<Dtype>* batch1, int batchPos1, Batch<Dtype>*  batch2, int batchPos2) {
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
void MemoryCache<Dtype>::create( void * ptr, bool * ptr2, bool thread_safe )
{

  cache = static_cast<Batch<Dtype> *> (ptr);
  Cache<Dtype>::prefetch = thread_safe;
  Cache<Dtype>::full_replace = false;
  Cache<Dtype>::dirty = ptr2;
  for(int i=0; i< Cache<Dtype>::size; i++)
    dirty[i] = true;
  
  /*for(int i=0; i< Cache<Dtype>::size; i++)
  {
    if(thread_safe)
      cache[i].sync_var = boost::make_shared<Batch<Dtype>::sync>();
    else
      cache[i].sync_var = NULL;
  }*/
}
template <typename Dtype>
bool MemoryCache<Dtype>::empty()
{ 
  //int bounds = Cache<Dtype>::used.fetch_add(0, boost::memory_order_relaxed);
  

  //LOG(INFO)  << bounds << " " << Cache<Dtype>::size;
  
  LOG(INFO) << " empty  " << Cache<Dtype>::used;

  return Cache<Dtype>::used == Cache<Dtype>::size;
}
template <typename Dtype>
Batch<Dtype> * MemoryCache<Dtype>::pop()
{
  //Cache<Dtype>::lock();
  //Cache<Dtype>::used++;
  //Cache<Dtype>::unlock();
  int slot = Cache<Dtype>::used.fetch_add(1, boost::memory_order_relaxed);
  Batch<Dtype> *batch = cache_full.pop();
  Cache<Dtype>::dirty[slot] = true;
  LOG(INFO)  << "Pop used "  << Cache<Dtype>::used;
  
  return batch;
}
template <typename Dtype>
void MemoryCache<Dtype>::shuffle ()
{
  //Cache<Dtype>::lock();
  static int last_i=0;
  //int bounds = Cache<Dtype>::used.fetch_add(0, boost::memory_order_relaxed);
  int rand;
  for (int i = last_i; i < Cache<Dtype>::size; ++i) {
   
    last_i=i;
    if(Cache<Dtype>::dirty[i] == true)
    { 
      for(int j=0; j< cache[i].data_.shape(0); j++)
      {
        rand = Cache<Dtype>::data_layer->randomGen(Cache<Dtype>::size);
        while(!dirty[rand])
          rand = Cache<Dtype>::data_layer->randomGen(Cache<Dtype>::size);
        shuffle_cache(&cache[i], j, &cache[rand], Cache<Dtype>::data_layer->randomGen(cache[i].data_.shape(0)));
      }
      Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      cache_full.push(&cache[i]);
      LOG(INFO)  << "Shuffle used "  << Cache<Dtype>::used;
      dirty[i] = false;
      last_i++;
    }
    else
      break;
  }
  //Cache<Dtype>::used = 0;
  if(last_i == Cache<Dtype>::size)
  {
    //bounds=0;
    Cache<Dtype>::full_replace = true;
    last_i=0;
  }
  //refill_start = bounds;
  //Cache<Dtype>::unlock();
}
template <typename Dtype>
void MemoryCache<Dtype>::fill(bool in_thread)
{
  //Cache<Dtype>::lock();
  static int last_i=0;
  for (int j = last_i; j < Cache<Dtype>::size; ++j) {
    last_i=j;
    if(Cache<Dtype>::dirty[j] == true)
    {  
      Cache<Dtype>::data_layer->load_batch(&cache[j], in_thread);
      Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      cache_full.push(&cache[j]);
      dirty[j] = false;
      last_i++;
      LOG(INFO)  << "Fill used "  << Cache<Dtype>::used;
    }
    else
      break;
  }
  if(last_i == Cache<Dtype>::size)
  {
    //bounds=0;
    Cache<Dtype>::full_replace = true;
    last_i=0;
  }
  //refill_start = bounds;
  //Cache<Dtype>::used = 0;
  //Cache<Dtype>::unlock();
}
template <typename Dtype>
void MemoryCache<Dtype>::refill(Cache<Dtype> * next_cache)
{
  //Cache<Dtype>::lock();
  Batch<Dtype>* batch;
  static int last_i=0;
  for (int j = last_i; j < Cache<Dtype>::size; ++j) {
    //LOG(INFO) << position;
    last_i=j;
    if(Cache<Dtype>::dirty[j] == true)
    {  
      batch = next_cache->pop(); //->cache_full_.pop("Data layer cache queue empty");
      cache[j].data_.CopyFrom( batch->data_ );
      cache[j].label_.CopyFrom( batch->label_ );
      Cache<Dtype>::used.fetch_sub(1, boost::memory_order_relaxed);
      cache_full.push(&cache[j]);
      LOG(INFO)  << "Refill used "  << Cache<Dtype>::used;
      dirty[j] = false;
      last_i++;
    }
    else
      break;
  }
  if(last_i == Cache<Dtype>::size)
  {
    //bounds=0;
    Cache<Dtype>::full_replace = true;
    last_i=0;
  } 
  //refill_start = bounds;
  //Cache<Dtype>::unlock();
}  
template <typename Dtype>
void MemoryCache<Dtype>::reshape(vector<int> * top_shape, vector<int> * label_shape)
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
template <typename Dtype>
void MemoryCache<Dtype>::mutate_data(bool labels)
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
template <typename Dtype>
void DiskCache<Dtype>::shuffle_cache(int batch1, int batchPos1, int  batch2, int batchPos2, int image_count, int data_count, int label_count) {
  
  LOG(INFO) << "Error: Disk Caching Shuffle Disabled";
  /*unsigned int image1_loc = (batch1*(image_count*(data_count+1))*sizeof(Dtype))+(batchPos1*(data_count+1)*sizeof(Dtype));
  unsigned int image2_loc = (batch2*(image_count*(data_count+1))*sizeof(Dtype))+(batchPos2*(data_count+1)*sizeof(Dtype));
  int offset1 = cache_buffer->data_.offset(batchPos1);
  int offset2 = cache_buffer->data_.offset(batchPos2);
  //unsigned int last_loc = (image_count*(image_count*(data_count))*sizeof(Dtype));
  Dtype * data = cache_buffer->data_.mutable_cpu_data();  
  Dtype * label = cache_buffer->label_.mutable_cpu_data();  
  
  cache.seekg (image1_loc);
  for (int i = 0; i < data_count; ++i)
    cache >> data[i]; 
  
  cache.seekg (image2_loc);
  for (int i = 0; i < data_count; ++i)
    cache >> data[i+data_count]; 
  
  cache.seekg (last_loc+sizeof(Dtype)*batch1*image_count + sizeof(Dtype)*batchPos1);
  cache >> label[0];
  
  cache.seekg (last_loc+sizeof(Dtype)*batch2*image_count + sizeof(Dtype)*batchPos2);
  cache >> label[1];
 
  //Write stuff
  cache.seekg (image1_loc);
  for (int i = 0; i < data_count; ++i)
    cache << data[i+data_count]; 
  
  cache.seekg (image2_loc);
  for (int i = 0; i < data_count; ++i)
    cache << data[i]; 
  
  cache.seekg (last_loc+sizeof(Dtype)*batch1);
  cache << label[1];
  
  cache.seekg (last_loc+sizeof(Dtype)*batch2);
  cache << label[0];*/
}
template <typename Dtype>
void DiskCache<Dtype>::create( void * ptr, bool * ptr2, bool thread_safe )
{
  Cache<Dtype>::prefetch = thread_safe;
  Cache<Dtype>::full_replace = false;
  Cache<Dtype>::dirty = ptr2;
  for(int i=0; i< Cache<Dtype>::size; i++)
    dirty[i] = true;

  //if(thread_safe)
  //  Cache<Dtype>::sync_var = boost::make_shared<Cache<Dtype>::sync>();
  //else
  //  Cache<Dtype>::sync_var = NULL;
  open = false;
  cache_buffer = static_cast<Batch<Dtype> *> (ptr); 
  current_offset = 0;
  if( Cache<Dtype>::eviction_rate!=0)
  {
    LOG(INFO) << "Error: Disk Caching shuffle is not supported / auto turning off shuffling";
    Cache<Dtype>::eviction_rate=0;
  }
}
template <typename Dtype>
bool DiskCache<Dtype>::empty()
{ 
  
  return current_offset == Cache<Dtype>::size;
}
template <typename Dtype>
Batch<Dtype> * DiskCache<Dtype>::pop() {
  //Cache<Dtype>::lock();
  Dtype * data = cache_buffer->data_.mutable_cpu_data();  
  Dtype * label = cache_buffer->label_.mutable_cpu_data();  
  int image_count;
  int datum_size;
  char * bytes;
  char byte;

  cache.read( (char *)&image_count, sizeof(int)); 
  cache.read( (char *)&datum_size, sizeof(int)); 
  for (int i = 0; i < image_count; ++i)
  {
    int offset = cache_buffer->data_.offset(i);
    bytes = (char*) (data+offset);
    //for (int k = 0; k < datum_size; ++k)
    cache.read( bytes, datum_size); 
    bytes = (char*) (label+i);
    //for (int k = 0; k < sizeof(Dtype); ++k)
    cache.read( bytes, sizeof(Dtype)); 
  }

  current_offset++;
  Cache<Dtype>::used++;

  //Cache<Dtype>::unlock();
  return cache_buffer;
}
template <typename Dtype>
void DiskCache<Dtype>::fill(bool in_thread)
{
  //Cache<Dtype>::lock();
  if(!open)
  {
    LOG(INFO) << "Cache Location" << Cache<Dtype>::disk_location;
    cache.open (Cache<Dtype>::disk_location, ios::trunc| ios::in | ios::out | ios::binary );
    open = true;
    if(!cache.is_open())
    {
      LOG(INFO) << "Couldn't open disk cache location: " << Cache<Dtype>::disk_location;
      exit(1);
    }

  } 
  Dtype * data = cache_buffer->data_.mutable_cpu_data();  
  Dtype * label = cache_buffer->label_.mutable_cpu_data();
  char * bytes;
  cache.seekg (0, ios::beg);
  for (int j = 0; j < Cache<Dtype>::used; ++j) {
    Cache<Dtype>::data_layer->load_batch(cache_buffer, in_thread);
    int image_count = cache_buffer->data_.shape(0);
    int datum_size = cache_buffer->data_.shape(1);
    datum_size *= cache_buffer->data_.shape(2);
    datum_size *= cache_buffer->data_.shape(3);
    datum_size *= sizeof(Dtype);

    cache.write( (char *)&image_count, sizeof(int)); 
    cache.write( (char *)&datum_size, sizeof(int)); 
    for (int i = 0; i < image_count; ++i)
    {
      int offset = cache_buffer->data_.offset(i);
      bytes = (char*) (data+offset);
      cache.write( bytes, datum_size); 
      bytes = (char*) (label+i);
      cache.write( bytes, sizeof(Dtype)); 
    }
  }
  cache.seekg (0, ios::beg);
  current_offset = 0;
  Cache<Dtype>::used=0;
  //Cache<Dtype>::unlock();
}
template <typename Dtype>
void DiskCache<Dtype>::shuffle ()
{
  //Cache<Dtype>::lock();
  cache.seekg (0, ios::beg);
  int image_count;
  int datum_size;
  cache.read( (char *)&image_count, sizeof(int)); 
  cache.read( (char *)&datum_size, sizeof(int)); 
  for(int i=0; i< Cache<Dtype>::size; i++)
  {
    for(int j=0; j< cache_buffer->data_.shape(0); j++)
    {
      shuffle_cache(i, j, Cache<Dtype>::data_layer->randomGen(size), Cache<Dtype>::data_layer->randomGen(image_count), image_count, datum_size, 1);
    }
  }
  cache.seekg (0, ios::beg);
  current_offset = 0;
  //Cache<Dtype>::unlock();
}
template <typename Dtype>
void DiskCache<Dtype>::refill(Cache<Dtype> * next_cache)
{
  //Cache<Dtype>::lock();
  Batch<Dtype>* batch;
  Dtype * data = cache_buffer->data_.mutable_cpu_data();  
  Dtype * label = cache_buffer->label_.mutable_cpu_data();  
  current_offset=0;
  cache.seekg (0, ios::beg);
  char * bytes;
  for (int j = 0; j < Cache<Dtype>::used; ++j) {
    batch = next_cache->pop(); //->cache_full_.pop("Data layer cache queue empty");
    data = batch->data_.mutable_cpu_data();  
    label = batch->label_.mutable_cpu_data();  
    //cache_buffer->data_.CopyFrom( batch->data_ );
    //cache_buffer->label_.CopyFrom( batch->label_ );
    int image_count = batch->data_.shape(0);
    int datum_size = batch->data_.shape(1);
    datum_size *= batch->data_.shape(2);
    datum_size *= batch->data_.shape(3);
    datum_size *= sizeof(Dtype);

    cache.write( (char *)&image_count, sizeof(int)); 
    cache.write( (char *)&datum_size, sizeof(int)); 
    for (int i = 0; i < image_count; ++i)
    {
      int offset = batch->data_.offset(i);
      bytes = (char*) (data+offset);
      cache.write( bytes, datum_size); 
      bytes = (char*) (label+i);
      cache.write( bytes, sizeof(Dtype)); 
    }
  }
  cache.seekg (0, ios::beg);
  Cache<Dtype>::used=0;
  //Cache<Dtype>::unlock();
}  
template <typename Dtype>
void DiskCache<Dtype>::reshape(vector<int> * top_shape, vector<int> * label_shape)
{
  //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->data_.Reshape(*top_shape);
  //}
  if (label_shape) {
    //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->label_.Reshape(*label_shape);
    //}
  }
}
template <typename Dtype>
void DiskCache<Dtype>::mutate_data(bool labels)
{
  //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->data_.mutable_cpu_data();
  //}
  if (labels) {
    //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->label_.mutable_cpu_data();
    //}
  }
#ifndef CPU_ONLY
  //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache[i].data_.mutable_gpu_data();
    //}
  if (labels) {
    //for(int i=0; i< Cache<Dtype>::size; i++) {
      cache_buffer->label_.mutable_gpu_data();
    //}
  }
#endif
}
INSTANTIATE_CLASS(Cache);
INSTANTIATE_CLASS(MemoryCache);
INSTANTIATE_CLASS(DiskCache);
}

