#include <stdint.h>
#include <string.h>

#include <vector>

#if USE_PNETCDF
#include <pnetcdf.h>
#endif

#define STRIDED 0

#include <boost/thread.hpp>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/pnetcdf_all_data_layer.hpp"
#include "caffe/mpi.hpp"
//#if CAFFE_FT
//#include <mpi-ext.h>
//#endif 
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
PnetCDFAllDataLayer<Dtype>::PnetCDFAllDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    current_row_(0),
    max_row_(0),
    datum_shape_(),
    data_(),
    label_(),
    data_byte_count_(0),
    label_byte_count_(0),
#ifdef CAFFE_FT
    padd_data_(),
    padd_label_(),
    padd_data_byte_count_(0),
    padd_label_byte_count_(0),
#endif 
    row_mutex_(),
    comm_(),
    comm_rank_(),
    comm_size_() {
#ifdef CAFFE_FT
  comm_ = caffe::mpi::get_working_comm();
  // MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);
  std::cout << "Working Comm PNETCDFALLDATALAYER.\n";
#else
  comm_ = caffe::mpi::comm_dup();
#endif 
  comm_rank_ = caffe::mpi::comm_rank(comm_);
  comm_size_ = caffe::mpi::comm_size(comm_);
  LOG(INFO) << "Rank PnetCDF file: " << comm_rank_ 
      << " Size: " << comm_size_;

}

template <typename Dtype>
PnetCDFAllDataLayer<Dtype>::~PnetCDFAllDataLayer() {
  this->StopInternalThread();
}

static void errcheck(int retval) {
#if USE_PNETCDF
  if (NC_NOERR != retval) {
    LOG(FATAL) << "pnetcdf error: " << ncmpi_strerror(retval);
  }
#else
  NO_PNETCDF;
#endif
}

template <typename Dtype>
inline static Dtype prod(vector<Dtype> vec) {
  Dtype val = 1;
  for (size_t i=0; i<vec.size(); i++) {
    val *= vec[i];
  }
  return val;
}

#ifdef CAFFE_FT
template <typename Dtype>
std::tuple<int, bool> PnetCDFAllDataLayer<Dtype>::fix_comm_error(MPI_Comm* comm, float* x)
{
  MPI_Comm* tcomm = comm;
  std::tuple<int, bool> retVal;
  int trank, tsize;
  caffe::mpi::fix_communicator();
  *comm = caffe::mpi::get_working_comm();
  trank = caffe::mpi::comm_rank(*tcomm);
  tsize = caffe::mpi::comm_size(*tcomm);
  DLOG(INFO) << "Communicator Fixed: Rank: " << trank 
      << ", Size: " << tsize;
  
  //float xallreduce = 1.0;
  std::get<1>(retVal) = true;
  std::get<0>(retVal) =  MPI_Allreduce(MPI_IN_PLACE, x, 1, MPI_FLOAT, MPI_SUM, *comm);
  if(std::get<0>(retVal) != MPI_SUCCESS) {
    DLOG(INFO) << "ERROR OCCURED BEFORE FILE ACCESS. Could not recover. Aborting";
    caffe::mpi::error_report(std::get<0>(retVal), comm);
    // MPI_Group group_f;
    // std::get<0>(retVal1) = MPIX_Comm_failure_ack(comm_);
    // std::get<0>(retVal2) = MPIX_Comm_failure_get_acked(comm_, &group_f);
    MPI_Abort(*comm, std::get<0>(retVal));
  }
  comm_ = caffe::mpi::get_working_comm();
  comm_rank_ = caffe::mpi::comm_rank(comm_);
  comm_size_ = caffe::mpi::comm_size(comm_);
  return retVal;
}
#endif 

template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::load_pnetcdf_file_data(const string& filename) {
#if USE_PNETCDF
#if STRIDED
  LOG(INFO) << "Loading PnetCDF file, strided: " << filename;
#else
  LOG(INFO) << "Loading PnetCDF file: " << filename;
#endif

  int retval;
  int ncid;
  int ndims;
  int nvars;
  int ngatts;
  int unlimdim;
  MPI_Offset total;
  MPI_Offset count_;
  MPI_Offset remain;
  MPI_Offset start;
  MPI_Offset stop;

  #ifdef CAFFE_FT
  // int rc, rc2;
  std::tuple<int, bool> retVal1, retVal2; 
  DLOG(INFO) << "PnetCDF Before Opening File:------- ";
  // Check if communicator is still valid.

  float xallreduce = 1.0;
  std::get<1>(retVal1) = false;
  std::get<0>(retVal1) = MPI_Allreduce(MPI_IN_PLACE, &xallreduce, 1, MPI_FLOAT, 
              MPI_SUM, comm_);
  if(std::get<0>(retVal1) != MPI_SUCCESS)
  {
    retVal2 = fix_comm_error(&comm_, &xallreduce);
    DLOG(INFO) << "ERROR OCCURED BEFORE FILE ACCESS";
  }
  DLOG(INFO) << "Test AllReduce Value: " << xallreduce;            
  retval = ncmpi_open(comm_, filename.c_str(),
          NC_NOWRITE, MPI_INFO_NULL, &ncid);
  #else
  retval = ncmpi_open(comm_, filename.c_str(),
          NC_NOWRITE, MPI_INFO_NULL, &ncid);
  #endif
  errcheck(retval);
  int rank = comm_rank_;
  int size = comm_size_;

  DLOG(INFO) << "PnetCDF After Opening File:-------w rank: " 
             << rank << ", size:" << size;

  retval = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &unlimdim);
  errcheck(retval);

  retval = ncmpi_inq_dimlen(ncid, unlimdim, &total);
  errcheck(retval);

  count_ = total / size;
  remain = total % size;
#if STRIDED
  start = rank;
  stop = rank; // dummy value, not used
  if (rank < remain) {
      count_ += 1;
  }
#else
  start = rank * count_;
  stop = rank * count_ + count_;
  if (rank < remain) {
    start += rank;
    stop += rank + 1;
  } else {
    start += remain;
    stop += remain;
  }
#endif

  DLOG(INFO) << "ncid " << ncid;
  DLOG(INFO) << "ndims " << ndims;
  DLOG(INFO) << "nvars " << nvars;
  DLOG(INFO) << "ngatts " << ngatts;
  DLOG(INFO) << "unlimdim " << unlimdim;
  DLOG(INFO) << "total images " << total;
  DLOG(INFO) << "start " << start;
  DLOG(INFO) << "stop " << stop;

  for (int varid = 0; varid < nvars; varid++) {
    int vartype;
    int varndims;
    vector<int> vardimids;
    vector<MPI_Offset> count;
    vector<MPI_Offset> offset;
    vector<MPI_Offset> stride;
    MPI_Offset chunksize = 2147483647L;
    MPI_Offset prodcount;

    retval = ncmpi_inq_vartype(ncid, varid, &vartype);
    errcheck(retval);

    retval = ncmpi_inq_varndims(ncid, varid, &varndims);
    errcheck(retval);

    vardimids.resize(varndims);
    count.resize(varndims);
    offset.resize(varndims);
    stride.resize(varndims);

    retval = ncmpi_inq_vardimid(ncid, varid, &vardimids[0]);
    errcheck(retval);

    for (int i = 0; i < varndims; i++) {
      retval = ncmpi_inq_dimlen(ncid, vardimids[i], &count[i]);
      errcheck(retval);
      offset[i] = 0;
      stride[i] = 1;
      if (count[i] > chunksize) {
        LOG(FATAL) << "dimension is too large for Blob";
      }
    }
    // MPI-IO can only read 2GB chunks due to "int" interface for indices
#if STRIDED
    count[0] = count_;
    offset[0] = start;
    stride[0] = size;
#else
    count[0] = stop-start;
    offset[0] = start;
#endif
    prodcount = prod(count);

    if (NC_BYTE == vartype) {
      this->data_ = shared_ptr<signed char>(new signed char[prodcount]);
      datum_shape_.resize(4);
      datum_shape_[0] = 1;
      datum_shape_[1] = count[1];
      datum_shape_[2] = count[2];
      datum_shape_[3] = count[3];
      DLOG(INFO) << "datum_shape_[0] " << datum_shape_[0];
      DLOG(INFO) << "datum_shape_[1] " << datum_shape_[1];
      DLOG(INFO) << "datum_shape_[2] " << datum_shape_[2];
      DLOG(INFO) << "datum_shape_[3] " << datum_shape_[3];
      if (prodcount < chunksize) {
        LOG(INFO) << "reading PnetCDF data whole " << count[0];
        LOG(INFO) << "offset={"<<offset[0]<<","<<offset[1]<<","<<offset[2]<<","<<offset[3]<<"}";
        LOG(INFO) << "count={"<<count[0]<<","<<count[1]<<","<<count[2]<<","<<count[3]<<"}";
#if STRIDED
        retval = ncmpi_get_vars_schar_all(ncid, varid, &offset[0],
            &count[0], &stride[0], this->data_.get());
#else
        retval = ncmpi_get_vara_schar_all(ncid, varid, &offset[0],
            &count[0], this->data_.get());
#endif
        data_byte_count_ += prodcount;
        errcheck(retval);
      }
      else {
        vector<MPI_Offset> newoffset = offset;
        vector<MPI_Offset> newcount = count;
        MPI_Offset data_offset = 0;
        newcount[0] = 1;
        MPI_Offset newprodcount = prod(newcount);
        newcount[0] = chunksize/newprodcount;
        newprodcount = prod(newcount);
        if (newprodcount >= chunksize) {
          LOG(FATAL) << "newprodcount >= chunksize";
        }
        MPI_Offset cur = 0;
        shared_ptr<signed char> chunk = shared_ptr<signed char>(
            new signed char[newprodcount]);
        while (cur < count[0]) {
          if (cur+newcount[0] > count[0]) {
            newcount[0] = count[0]-cur;
            newprodcount = prod(newcount);
          }
          LOG(INFO) << "reading data chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
          retval = ncmpi_get_vars_schar_all(ncid, varid, &newoffset[0],
              &newcount[0], &stride[0], chunk.get());
#else
          retval = ncmpi_get_vara_schar_all(ncid, varid, &newoffset[0],
              &newcount[0], chunk.get());
#endif
          errcheck(retval);
          memcpy(this->data_.get() + data_offset, chunk.get(), newprodcount);
          data_byte_count_ += newprodcount;
          cur += newcount[0];
#if STRIDED
          newoffset[0] += newcount[0]*size;
#else
          newoffset[0] += newcount[0];
#endif
          data_offset += newprodcount;
        }
      }
    }
    else if (NC_INT == vartype && this->output_labels_) {
      max_row_ = count[0];
      this->label_ = shared_ptr<int>(new int[max_row_]);
      LOG(INFO) << "PnetCDF max_row_ = " << max_row_;
      if (prodcount < chunksize) {
        LOG(INFO) << "reading PnetCDF label whole " << count[0];
#if STRIDED
        retval = ncmpi_get_vars_int_all(ncid, varid, &offset[0],
            &count[0], &stride[0], this->label_.get());
#else
        retval = ncmpi_get_vara_int_all(ncid, varid, &offset[0],
            &count[0], this->label_.get());
#endif
        label_byte_count_ += prodcount;
        errcheck(retval);
      }
      else {
        vector<MPI_Offset> newoffset = offset;
        vector<MPI_Offset> newcount = count;
        MPI_Offset data_offset = 0;
        newcount[0] = 1;
        MPI_Offset newprodcount = prod(newcount);
        newcount[0] = chunksize/newprodcount;
        newprodcount = prod(newcount);
        if (newprodcount >= chunksize) {
          LOG(FATAL) << "newprodcount >= chunksize";
        }
        MPI_Offset cur = 0;
        shared_ptr<int> chunk = shared_ptr<int>(new int[newprodcount]);
        while (cur < count[0]) {
          if (cur+newcount[0] > count[0]) {
            newcount[0] = count[0]-cur;
            newprodcount = prod(newcount);
          }
          LOG(INFO) << "reading label chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
          retval = ncmpi_get_vars_int_all(ncid, varid, &newoffset[0],
              &newcount[0], &stride[0], chunk.get());
#else
          retval = ncmpi_get_vara_int_all(ncid, varid, &newoffset[0],
              &newcount[0], chunk.get());
#endif
          errcheck(retval);
          memcpy(this->label_.get() + data_offset, chunk.get(), newprodcount);
          label_byte_count_ += newprodcount;
          cur += newcount[0];
#if STRIDED
          newoffset[0] += newcount[0]*size;;
#else
          newoffset[0] += newcount[0];
#endif
          data_offset += newprodcount;
        }
      }
    }
    else {
      LOG(FATAL) << "unknown data type";
    }
  }

  retval = ncmpi_close(ncid);
  errcheck(retval);

  DLOG(INFO) << "Total Data Byte Count:" << data_byte_count_;
  DLOG(INFO) << "Total Label Byte Count:" << label_byte_count_;

  {
    const int batch_size = this->layer_param_.data_param().batch_size();
    Dtype label_sum = 0;
    for (int i=0; i<batch_size; i++) {
      label_sum += *(this->label_.get()+i);
    }
    caffe::mpi::allreduce(label_sum);
    LOG(INFO) << "Label Sum: " << label_sum;
  }
#else
  NO_PNETCDF;
#endif
}

//////////////////////////////////////// 
#ifdef CAFFE_FT
template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::reload_pnetcdf_file_data(const string& filename) {
#if USE_PNETCDF
  #if STRIDED
    LOG(INFO) << "Loading PnetCDF file, strided: " << filename;
  #else
    LOG(INFO) << "Loading PnetCDF file: " << filename;
  #endif
  
  int retval;
  int ncid;
  int ndims;
  int nvars;
  int ngatts;
  int unlimdim;
  MPI_Offset total;
  MPI_Offset count_;
  MPI_Offset remain;
  MPI_Offset faulted_start;
  MPI_Offset faulted_stop;

  std::tuple<int, bool> retVal1, retVal2;
  DLOG(INFO) << "PnetCDF Before Opening File:------- ";
  // Check if communicator is still valid.
  float xallreduce = 1.0;
  std::get<1>(retVal1) = false;
  std::get<0>(retVal1) = MPI_Allreduce(MPI_IN_PLACE, &xallreduce, 1, MPI_FLOAT, 
              MPI_SUM, comm_);
  if(std::get<0>(retVal1) != MPI_SUCCESS)
  {
    retVal2 = fix_comm_error(&comm_, &xallreduce);
    DLOG(INFO) << "ERROR OCCURED BEFORE FILE ACCESS";
  }
  DLOG(INFO) << "Test AllReduce Value: " << xallreduce;            
  retval = ncmpi_open(comm_, filename.c_str(),
          NC_NOWRITE, MPI_INFO_NULL, &ncid);
  errcheck(retval);
  
  DLOG(INFO) << "PnetCDF After Opening File:-------w rank: " 
             << comm_rank_ << ", size:" << comm_size_;
        
  retval = ncmpi_inq(ncid, &ndims, &nvars, &ngatts, &unlimdim);
  errcheck(retval);

  retval = ncmpi_inq_dimlen(ncid, unlimdim, &total);
  errcheck(retval);

  // Make universal for multiple faults
  if(caffe::mpi::last_rank_failed >= 0) {
    int frank = caffe::mpi::last_rank_failed;
    // int size = comm_size_;
    count_ = total / caffe::mpi::old_size; // 1250
    remain = total % caffe::mpi::old_size; // 0
      
    DLOG(INFO) << "Last Rank Failed :" << caffe::mpi::last_rank_failed;
    DLOG(INFO) << "Last COMM Size :" << caffe::mpi::old_size;
    DLOG(INFO) << "Count_ with prev size:" << count_;
    DLOG(INFO) << "Remain with prev size :" << remain;


    // Strided does not work for the moment. 
  #if STRIDED
    faulted_start = caffe::mpi::last_rank_failed;
    faulted_stop = caffe::mpi::last_rank_failed;
    if (caffe::mpi::rank < remain) 
      count_+=1;
  #else
    faulted_start = caffe::mpi::last_rank_failed * count_;
    faulted_stop = caffe::mpi::last_rank_failed * count_ + count_;
    if (caffe::mpi::last_rank_failed < remain) {
      faulted_start += caffe::mpi::last_rank_failed;
      faulted_stop += caffe::mpi::last_rank_failed + 1;
    } else {
      faulted_start += remain;
      faulted_stop += remain;
    }
  #endif 

    MPI_Offset padd_count_ = (faulted_stop - faulted_start)/comm_size_;
    MPI_Offset padd_remain = (faulted_stop - faulted_start)% comm_size_;

    MPI_Offset padd_start; 
    MPI_Offset padd_stop;

  #if STRIDED
    padd_start = faulted_start + comm_rank_;
    // padd_stop = faulted_start + comm_rank_;
    if ( comm_rank_ < padd_remain) {
      padd_count_ += 1;
    }
  #else
    padd_start = faulted_start + comm_rank_ * padd_count_;
    padd_stop = faulted_start + comm_rank_ * padd_count_ + padd_count_;
    if (comm_rank_ < padd_remain) {
      padd_start += comm_rank_;
      padd_stop += comm_rank_ + 1;
    } else {
      padd_start += padd_remain; 
      padd_stop += padd_remain; 
    }
  #endif  

    DLOG(INFO) << "ncid " << ncid;
    DLOG(INFO) << "ndims " << ndims;
    DLOG(INFO) << "nvars " << nvars;
    DLOG(INFO) << "ngatts " << ngatts;
    DLOG(INFO) << "unlimdim " << unlimdim;
    DLOG(INFO) << "total images " << total;
    DLOG(INFO) << "padd start (failed)" << padd_start;
    DLOG(INFO) << "padd stop (failed)" << padd_stop;

    for (int varid = 0; varid < nvars; varid++) {
      int vartype;
      int varndims;
      vector<int> vardimids;
      vector<MPI_Offset> count;
      vector<MPI_Offset> offset;
      vector<MPI_Offset> stride;
      MPI_Offset chunksize = 2147483647L;
      MPI_Offset prodcount;

      retval = ncmpi_inq_vartype(ncid, varid, &vartype);
      errcheck(retval);

      retval = ncmpi_inq_varndims(ncid, varid, &varndims);
      errcheck(retval);

      vardimids.resize(varndims);
      count.resize(varndims);
      offset.resize(varndims);
      stride.resize(varndims);

      retval = ncmpi_inq_vardimid(ncid, varid, &vardimids[0]);
      errcheck(retval);

      vardimids.resize(varndims);
      count.resize(varndims);
      offset.resize(varndims);
      stride.resize(varndims);

      retval = ncmpi_inq_vardimid(ncid, varid, &vardimids[0]);
      errcheck(retval);

      for (int i = 0; i < varndims; i++) {
        retval = ncmpi_inq_dimlen(ncid, vardimids[i], &count[i]);
        errcheck(retval);
        offset[i] = 0;
        stride[i] = 1;
        if (count[i] > chunksize) {
          LOG(FATAL) << "dimension is too large for Blob";
        } 
      }
      // MPI-IO can only read 2GB chunks due to "int" interface for indices
#if STRIDED
      count[0] = padd_count_; // count_;
      offset[0] = padd_start; // start; 
      stride[0] = comm_size_; // size
#else
      count[0] = padd_stop - padd_start;// stop-start;
      offset[0] = padd_start;// start;
#endif
      prodcount = prod(count);

      if (NC_BYTE == vartype) {
        this->padd_data_ = shared_ptr<signed char>(new signed char[prodcount]);
        datum_shape_.resize(4);
        datum_shape_[0] = 1;
        datum_shape_[1] = count[1];
        datum_shape_[2] = count[2];
        datum_shape_[3] = count[3];
        DLOG(INFO) << "datum_shape_[0] " << datum_shape_[0];
        DLOG(INFO) << "datum_shape_[1] " << datum_shape_[1];
        DLOG(INFO) << "datum_shape_[2] " << datum_shape_[2];
        DLOG(INFO) << "datum_shape_[3] " << datum_shape_[3];
        if (prodcount < chunksize) {
          LOG(INFO) << "reading PnetCDF data whole " << count[0];
          LOG(INFO) << "offset={"<<offset[0]<<","<<offset[1]<<","<<offset[2]<<","<<offset[3]<<"}";
          LOG(INFO) << "count={"<<count[0]<<","<<count[1]<<","<<count[2]<<","<<count[3]<<"}";
#if STRIDED
          retval = ncmpi_get_vars_schar_all(ncid, varid, &offset[0],
              &count[0], &stride[0], this->padd_data_.get());
#else
          retval = ncmpi_get_vara_schar_all(ncid, varid, &offset[0],
              &count[0], this->padd_data_.get());
#endif
          padd_data_byte_count_ += prodcount;
          errcheck(retval);
        }
        else {
          vector<MPI_Offset> newoffset = offset;
          vector<MPI_Offset> newcount = count;
          MPI_Offset data_offset = 0;
          newcount[0] = 1;
          MPI_Offset newprodcount = prod(newcount);
          newcount[0] = chunksize/newprodcount;
          newprodcount = prod(newcount);
          if (newprodcount >= chunksize) {
            LOG(FATAL) << "newprodcount >= chunksize";
          }
          MPI_Offset cur = 0;
          shared_ptr<signed char> chunk = shared_ptr<signed char>(
              new signed char[newprodcount]);
          while (cur < count[0]) {
            if (cur+newcount[0] > count[0]) {
              newcount[0] = count[0]-cur;
              newprodcount = prod(newcount);
            }
            LOG(INFO) << "reading data chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
            retval = ncmpi_get_vars_schar_all(ncid, varid, &newoffset[0],
                &newcount[0], &stride[0], chunk.get());
#else
            retval = ncmpi_get_vara_schar_all(ncid, varid, &newoffset[0],
                &newcount[0], chunk.get());
#endif
            errcheck(retval);
            memcpy(this->padd_data_.get() + data_offset, chunk.get(),newprodcount);
            padd_data_byte_count_ += newprodcount;
            cur += newcount[0];
#if STRIDED
            newoffset[0] += newcount[0]*size;
#else
            newoffset[0] += newcount[0];
#endif
            data_offset += newprodcount;
          }
        }
      }
      else if (NC_INT == vartype && this->output_labels_) {
        padd_max_row_ = count[0];
        // this->padd_label_ = shared_ptr<int>(new int[max_row_]);
        this->padd_label_ = shared_ptr<int>(new int[padd_max_row_]);

        LOG(INFO) << "PnetCDF padd max_row_ = " << padd_max_row_;
        if (prodcount < chunksize) {
          LOG(INFO) << "reading PnetCDF label whole " << count[0];
#if STRIDED
          retval = ncmpi_get_vars_int_all(ncid, varid, &offset[0],
              &count[0], &stride[0], this->padd_label_.get());
#else
          retval = ncmpi_get_vara_int_all(ncid, varid, &offset[0],
              &count[0], this->padd_label_.get());
#endif
          padd_label_byte_count_ += prodcount;
          errcheck(retval);
        }
        else {
          vector<MPI_Offset> newoffset = offset;
          vector<MPI_Offset> newcount = count;
          MPI_Offset data_offset = 0;
          newcount[0] = 1;
          MPI_Offset newprodcount = prod(newcount);
          newcount[0] = chunksize/newprodcount;
          newprodcount = prod(newcount);
          if (newprodcount >= chunksize) {
            LOG(FATAL) << "newprodcount >= chunksize";
          }
          MPI_Offset cur = 0;
          shared_ptr<int> chunk = shared_ptr<int>(new int[newprodcount]);
          while (cur < count[0]) {
            if (cur+newcount[0] > count[0]) {
              newcount[0] = count[0]-cur;
              newprodcount = prod(newcount);
            }
            LOG(INFO) << "reading label chunk " << cur << " ... " << cur+newcount[0];
#if STRIDED
            retval = ncmpi_get_vars_int_all(ncid, varid, &newoffset[0],
                &newcount[0], &stride[0], chunk.get());
#else
            retval = ncmpi_get_vara_int_all(ncid, varid, &newoffset[0],
                &newcount[0], chunk.get());
#endif
            errcheck(retval);
            memcpy(this->padd_label_.get() + data_offset, chunk.get(), newprodcount);
            padd_label_byte_count_ += newprodcount;
            cur += newcount[0];
#if STRIDED
            newoffset[0] += newcount[0]*size;;
#else
            newoffset[0] += newcount[0];
#endif
            data_offset += newprodcount;
          }
        }
      }
      else {
        LOG(FATAL) << "unknown data type";
      }
    }

    retval = ncmpi_close(ncid);
    errcheck(retval);

    DLOG(INFO) << "Total Padd Data Byte Count:" << padd_data_byte_count_;
    DLOG(INFO) << "Total Padd Label Byte Count:" << padd_label_byte_count_;

    {
      const int batch_size = this->layer_param_.data_param().batch_size();
      Dtype label_sum = 0;
      for (int i=0; i<batch_size; i++) {
        label_sum += *(this->padd_label_.get()+i);
      }
      caffe::mpi::allreduce(label_sum);
      LOG(INFO) << "Padd Label Sum: " << label_sum;
    }
    //}
  }

#else
  NO_PNETCDF;
#endif /*USE_PNETCDF*/
}
#endif /*CAFFE_FT*/
//////////////////////////////////////////////////////
template <typename Dtype>
size_t PnetCDFAllDataLayer<Dtype>::get_datum_size() {
  vector<int> top_shape = this->get_datum_shape();
  const size_t datum_channels = top_shape[1];
  const size_t datum_height = top_shape[2];
  const size_t datum_width = top_shape[3];
  return datum_channels*datum_height*datum_width;
}

template <typename Dtype>
vector<int> PnetCDFAllDataLayer<Dtype>::get_datum_shape() {
  CHECK(this->data_);
  CHECK(this->datum_shape_.size());
  return this->datum_shape_;
}

template <typename Dtype>
vector<int> PnetCDFAllDataLayer<Dtype>::infer_blob_shape() {
  vector<int> top_shape = this->get_datum_shape();
  const int crop_size = this->transform_param_.crop_size();
  const int datum_height = top_shape[2];
  const int datum_width = top_shape[3];
  // Check dimensions.
  CHECK_GE(datum_height, crop_size);
  CHECK_GE(datum_width, crop_size);
  // Build BlobShape.
  top_shape[2] = (crop_size)? crop_size: datum_height;
  top_shape[3] = (crop_size)? crop_size: datum_width;
  return top_shape;
}

template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();

  // Load the pnetcdf file into data_ and optionally label_
  load_pnetcdf_file_data(this->layer_param_.data_param().source());

  row_mutex_.reset(new boost::mutex());

  vector<int> top_shape = infer_blob_shape();
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

#ifdef CAFFE_FT
template <typename Dtype>
void PnetCDFAllDataLayer<Dtype>::DataLayerUpdate(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int batch_size = this->layer_param_.data_param().batch_size();

  reload_pnetcdf_file_data(this->layer_param_.data_param().source());
  // Update the data_ and optionally label_
  CHECK(this->data_);
  CHECK(this->padd_data_);
  CHECK(this->label_);
  CHECK(this->padd_label_);
  // row_mutex_.reset(new boost::mutex());
  row_mutex_->lock();

  DLOG(INFO) << "Checking Data, padd_data, label, padd_label" ;
  DLOG(INFO) << "Here--------------------" ;

  shared_ptr<signed char> temp_newdata = shared_ptr<signed char>(new signed char[this->data_byte_count_ + this->padd_data_byte_count_]);

  memcpy(temp_newdata.get(), this->data_.get(), this->data_byte_count_);
  memcpy(temp_newdata.get()+this->data_byte_count_, this->padd_data_.get(), this->padd_data_byte_count_);

  this->data_.swap(temp_newdata);
  temp_newdata.reset();

  shared_ptr<int> temp_newlabel = shared_ptr<int>(new int[this->label_byte_count_ + this->padd_label_byte_count_]);

  memcpy(temp_newlabel.get(), this->label_.get(), this->label_byte_count_);
  memcpy(temp_newlabel.get() + this->label_byte_count_, this->padd_label_.get(), this->padd_label_byte_count_);

  this->label_.swap(temp_newlabel);
  temp_newlabel.reset();

  /*vector<int> top_shape = infer_blob_shape();
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size);
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }*/
  
  this->data_byte_count_ += this->padd_data_byte_count_;
  this->label_byte_count_ += this->padd_label_byte_count_;
  DLOG(INFO) << "Total Data Byte Count, after padd:" << this->data_byte_count_;
  DLOG(INFO) << "Total Label Byte Count, after padd:" << this->label_byte_count_;
  this->padd_data_.reset();
  this->padd_label_.reset();
  this->padd_data_byte_count_ = 0;
  this->padd_label_byte_count_ = 0;

  row_mutex_->unlock();
}
#endif /*CAFFE_FT*/

// This function is called on prefetch thread
template<typename Dtype>
void PnetCDFAllDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());

#ifndef _OPENMP
  CHECK(this->transformed_data_.count());
#endif

  vector<int> top_shape = get_datum_shape();
  size_t datum_size = get_datum_size();
  Datum masterDatum;
  masterDatum.set_channels(top_shape[1]);
  masterDatum.set_height(top_shape[2]);
  masterDatum.set_width(top_shape[3]);

  // Reshape according to the first datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  top_shape = infer_blob_shape();
#ifndef _OPENMP
  this->transformed_data_.Reshape(top_shape);
#endif
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  // set up the Datum

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables

  if (this->output_labels_) {
    top_label = batch->label_.mutable_cpu_data();
  }
#ifdef _OPENMP
  #pragma omp parallel if (batch_size > 1)
  #pragma omp single nowait
#endif
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a datum
    size_t row = (current_row_+item_id) % this->max_row_;
    size_t pnetcdf_offset = row * datum_size;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply data transformations (mirror, scale, crop...)
    #pragma omp task firstprivate(masterDatum, top_label, item_id)
    {
      Datum datum = masterDatum;
      datum.set_data(this->data_.get() + pnetcdf_offset, datum_size);

      int offset = batch->data_.offset(item_id);
#ifdef _OPENMP
      Blob<Dtype> tmp_data;
      tmp_data.Reshape(top_shape);
      tmp_data.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &tmp_data);
#else
      this->transformed_data_.set_cpu_data(top_data + offset);
      this->data_transformer_->Transform(datum, &(this->transformed_data_));
#endif
      // Copy label.
      if (this->output_labels_) {
        top_label[item_id] = this->label_.get()[row];
      }
    }
    trans_time += timer.MicroSeconds();
  }

  current_row_+=batch_size;

  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template<typename Dtype>
size_t PnetCDFAllDataLayer<Dtype>::next_row() {
  size_t row;
  row_mutex_->lock();
  row = current_row_++;
  current_row_ = current_row_ % this->max_row_;
  row_mutex_->unlock();
  return row;
}

INSTANTIATE_CLASS(PnetCDFAllDataLayer);
REGISTER_LAYER_CLASS(PnetCDFAllData);

}  // namespace caffe

