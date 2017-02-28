
#include <boost/thread.hpp>
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
void MemoryCache<Dtype>::create( void * ptr )
{
  cache = static_cast<Batch<Dtype> *> (ptr); 
}
template <typename Dtype>
bool MemoryCache<Dtype>::empty()
{ 
  return cache_full.size() == 0;
}
template <typename Dtype>
Batch<Dtype> * MemoryCache<Dtype>::pop()
{
  Cache<Dtype>::used++;
  return cache_full.pop();
}
template <typename Dtype>
void MemoryCache<Dtype>::shuffle (BasePrefetchingDataLayer<Dtype> * data_helper)
{
  for(int i=0; i< Cache<Dtype>::used; i++)
  {
    for(int j=0; j< cache[i].data_.shape(0); j++)
    {
      shuffle_cache(&cache[i], j, &cache[data_helper->randomGen(used)], data_helper->randomGen(cache[i].data_.shape(0)));
    }
    cache_full.push(&cache[i]);
  }
  Cache<Dtype>::used = 0;
}
template <typename Dtype>
void MemoryCache<Dtype>::fill(BasePrefetchingDataLayer<Dtype> * filler)
{
  for (int j = 0; j < Cache<Dtype>::used; ++j) {
    filler->load_batch(&cache[j]);
    cache_full.push(&cache[j]);
  }
  Cache<Dtype>::used = 0;
}
template <typename Dtype>
void MemoryCache<Dtype>::refill(Cache<Dtype> * next_cache)
{
  Batch<Dtype>* batch;
  for(int i=0; i< Cache<Dtype>::used; i++)
  {
    //LOG(INFO) << position;
    batch = next_cache->pop(); //->cache_full_.pop("Data layer cache queue empty");
    cache[i].data_.CopyFrom( batch->data_ );
    cache[i].label_.CopyFrom( batch->label_ );
    cache_full.push(&cache[i]);
  }
  Cache<Dtype>::used = 0;
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
void DiskCache<Dtype>::create( void * ptr )
{
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

  return cache_buffer;
}
template <typename Dtype>
void DiskCache<Dtype>::fill(BasePrefetchingDataLayer<Dtype> * filler)
{
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
    filler->load_batch(cache_buffer);
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
}
template <typename Dtype>
void DiskCache<Dtype>::shuffle (BasePrefetchingDataLayer<Dtype> * data_helper)
{
  cache.seekg (0, ios::beg);
  int image_count;
  int datum_size;
  cache.read( (char *)&image_count, sizeof(int)); 
  cache.read( (char *)&datum_size, sizeof(int)); 
  for(int i=0; i< Cache<Dtype>::size; i++)
  {
    for(int j=0; j< cache_buffer->data_.shape(0); j++)
    {
      shuffle_cache(i, j, data_helper->randomGen(size), data_helper->randomGen(image_count), image_count, datum_size, 1);
    }
  }
  cache.seekg (0, ios::beg);
  current_offset = 0;
}
template <typename Dtype>
void DiskCache<Dtype>::refill(Cache<Dtype> * next_cache)
{
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

