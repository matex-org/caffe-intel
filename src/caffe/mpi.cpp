#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdlib.h>
#include <stdexcept>
#include <unistd.h> // for gethostid()
#include "caffe/mpi.hpp"

namespace caffe {
namespace mpi {

#ifdef USE_MPI

MPI_Comm default_comm_ = MPI_COMM_WORLD;

#ifdef CAFFE_FT
MPI_Comm wcomm, rcomm;
MPI_Errhandler errh;
char err_str[MPI_MAX_ERROR_STRING] = "";
int err_strlen;
int new_size;
int old_size;
// int fault_global_flag = 0;
// [Last Comm Size, Last Rank Failed] 
// int* last_ranks_failed;
int last_rank_failed = -1;

void update_faulted_processes(int rank) {
  last_rank_failed = rank;
}

#endif

MPI_Comm get_comm_default() {
  #ifdef CAFFE_FT
  return wcomm;
  #else
  return default_comm_;
  #endif
}

void set_comm_default(MPI_Comm comm) {
  if (MPI_COMM_NULL != comm) {
    default_comm_ = comm;
  }
  else {
    throw std::runtime_error("cannot assign MPI_COMM_NULL as default comm");
  }
}

void init(int *argc, char ***argv) {

#ifdef CAFFE_FT
  int rank = 0, size = 0, namelen = 0;
  char name[MPI_MAX_PROCESSOR_NAME];

  int rc;
  // working comm, recovery comm;
  // MPI_Comm wcomm,
  //         rcomm;

  //MPI_Init(&argc, &argv);
  MPI_Init(argc, argv);
  FTCommunicator ftComm;

  // MPI_Comm_dup(MPI_COMM_WORLD, &wcomm);
  duplicate_comm(&wcomm, MPI_COMM_WORLD);
  //std::cout << "Here!-------------------------------" << std::endl;
  // ftCaffe.duplicateComm(default_comm_);
  // int rc = duplicate_comm(mpi::default_comm_, &ftComm.working_comm);

  // Error Handler for Fault Tolerance
  // if(!rc) {
  //   mpi::set_comm_default(ftComm.working_comm);
  // }

  // MPI_Comm_size(ftComm.working_comm, &size);
  // MPI_Comm_rank(ftComm.working_comm, &rank);
  MPI_Comm_size(wcomm, &size);
  MPI_Comm_rank(wcomm, &rank);
  MPI_Get_processor_name(name, &namelen);
  caffe::mpi::old_size = size;
  caffe::mpi::new_size = size;

  MPI_Comm_create_errhandler(verbose_errhandler, &errh);
  // MPI_Comm_dup(MPI_COMM_WORLD, &wcomm);
  std::cout << "My Rank: " << rank << ", size: " << size << "----------------!!!\n";

  //MPI_Comm_set_errhandler(ftComm.working_comm, ftComm.errh);
  //MPI_Comm_set_errhandler(ftComm.working_comm, MPI_ERRORS_RETURN);
  // MPI_Comm_set_errhandler(fcomm, errh);
  MPI_Comm_set_errhandler(wcomm, MPI_ERRORS_RETURN);
  // MPI_Comm_set_errhandler(wcomm, errh);

  // std::cout << "Here!-------------------------------2, rank:" << rank << ", size:" << size <<std::endl;

  // Fault Injection
  /*int victim = (rank == (size - 1));
  // MPI_Comm_split(ftComm.working_comm, victim? MPI_UNDEFINED: 1, rank, &ftComm.split_comm);
  if( rank == victim ) {
    std::cout << "Victim Rank: " << victim << std::endl;
    raise(SIGKILL);
  }*/

  update_faulted_processes(rank);

  LOG(INFO) << "Process rank " << rank << " from number of " << size
            << " processes running on " << name;
#else
  if (!initialized()) {
    int provided;
    if (MPI_SUCCESS != MPI_Init_thread(
          // argc, argv, MPI_THREAD_MULTIPLE, &provided)) {
          argc, argv, MPI_THREAD_SINGLE, &provided)) {

      throw std::runtime_error("MPI_Init_thread failed");
    }
  }

  // if (MPI_THREAD_MULTIPLE != query_thread()) {
  //   throw std::runtime_error("MPI threading level must be == MPI_THREAD_MULTIPLE");
  if (MPI_THREAD_SINGLE != query_thread()) {
    throw std::runtime_error("MPI threading level must be == MPI_THREAD_SINGLE");
  }

  if (0 != atexit(finalize)) {
    throw std::runtime_error("atexit(caffe::mpi::finalize) failed");
  }
#endif // CAFFE_FT
}

bool initialized() {
  int flag;

  if (MPI_SUCCESS != MPI_Initialized(&flag)) {
    throw std::runtime_error("MPI_Initialized failed");
  }

  return flag;
}

void finalize() {
  #ifdef CAFFE_FT
    MPI_Comm_free(&wcomm);
  #endif
  if (MPI_SUCCESS != MPI_Finalize()) {
    throw std::runtime_error("MPI_Finalize failed");
  }
}

int query_thread() {
  int provided;

  if (MPI_SUCCESS != MPI_Query_thread(&provided)) {
    throw std::runtime_error("MPI_Query_thread failed");
  }

  return provided;
}

MPI_Comm comm_dup(MPI_Comm comm) {
  MPI_Comm newcomm;
  std::cout << "Second Mpi Comm Dup Called-----------------------------\n";

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_dup(comm, &newcomm)) {
    throw std::runtime_error("MPI_Comm_dup failed");
    return MPI_COMM_NULL;
  }

  return newcomm;
}

void comm_free(MPI_Comm comm) {
  if (MPI_SUCCESS != MPI_Comm_free(&comm)) {
    throw std::runtime_error("MPI_Comm_free failed");
  }
}

int comm_rank(MPI_Comm comm) {
  int rank = 0;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_rank(comm, &rank)) {
    throw std::runtime_error("MPI_Comm_size failed");
    return 0;
  }

  return rank;
}

int comm_size(MPI_Comm comm) {
  int size = 0;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_size(comm, &size)) {
    throw std::runtime_error("MPI_Comm_size failed");
    return 0;
  }

  return size;
}

int node_rank(MPI_Comm comm) {
  int size = 0;
  int rank = 0;
  int node = 0;
  long *hostid = NULL;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  size = comm_size(comm);
  rank = comm_rank(comm);
  hostid = new long[size];
  hostid[rank] = gethostid();
  if (MPI_SUCCESS != MPI_Allgather(MPI_IN_PLACE, 0, MPI_LONG,
        hostid, 1, MPI_LONG, comm)) {
    delete [] hostid;
    throw std::runtime_error("MPI_Allgather failed");
    return 0;
  }

  for (int i=0; i<rank; ++i) {
    if (hostid[i] == hostid[rank]) {
      ++node;
    }
  }

  delete [] hostid;
  return node;
}

int node_size(MPI_Comm comm) {
  int size = 0;
  int rank = 0;
  int node = 0;
  long *hostid = NULL;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  size = comm_size(comm);
  rank = comm_rank(comm);
  hostid = new long[size];
  hostid[rank] = gethostid();
  if (MPI_SUCCESS != MPI_Allgather(MPI_IN_PLACE, 0, MPI_LONG,
        hostid, 1, MPI_LONG, comm)) {
    delete [] hostid;
    throw std::runtime_error("MPI_Allgather failed");
    return 0;
  }

  for (int i=0; i<size; ++i) {
    if (hostid[i] == hostid[rank]) {
      ++node;
    }
  }

  delete [] hostid;
  return node;
}

template <>
MPI_Datatype datatype<float>() {
  return MPI_FLOAT;
}

template <>
MPI_Datatype datatype<double>() {
  return MPI_DOUBLE;
}

#ifdef CAFFE_FT

void allreduce_copy(const float& sendbuf, float& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 float)");
  }
}

void allreduce_copy(const double& sendbuf, double& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 double)");
  }
}

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  int rc;
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  rc = MPI_Allreduce((void*)sendbuf, recvbuf, count, MPI_FLOAT, op, comm);
  if(rc != MPI_SUCCESS)
    caffe::mpi::error_report(rc, &comm);

  if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy float)");
  }
}

void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  int rc;
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  rc = MPI_Allreduce((void*)sendbuf, recvbuf, count, MPI_DOUBLE, op, comm);
  if(rc != MPI_SUCCESS)
    caffe::mpi::error_report(rc, &comm);

  if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy double)");
  }
}

std::tuple<int,bool> allreduce(float& buffer, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Float Ref): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  
  std::get<0>(ret_val) 
      = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_FLOAT, op, comm);
  if(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    caffe::mpi::fix_communicator();
    test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(test_comm);
    tsize = caffe::mpi::comm_size(test_comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Float Ref): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_FLOAT, op, test_comm);
    // caffe::mpi::fault_global_flag = 1;
    // caffe::mpi::bcast(&caffe::mpi::fault_global_flag, 1, trank, test_comm);
    // caffe::mpi::bcast(&caffe::mpi::last_rank_failed, 1,)
    /*if (MPI_SUCCESS != rc2) {
      throw std::runtime_error(
        "MPI_Allreduce failed even after fault recovery (allreduce 1 float ref)");
    }*/
    return ret_val;
  }
  /*if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 float)");
  }*/
  return ret_val;
}

std::tuple<int,bool> allreduce(double& buffer, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Double Ref): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  
  std::get<0>(ret_val) 
      = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_DOUBLE, op, comm);
  if(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    caffe::mpi::fix_communicator();
    test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(test_comm);
    tsize = caffe::mpi::comm_size(test_comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Double Ref): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, &buffer, 1, MPI_DOUBLE, op, test_comm);
    // caffe::mpi::fault_global_flag = 1;
    // caffe::mpi::bcast(&caffe::mpi::fault_global_flag, 1, trank, test_comm);
    /*if (MPI_SUCCESS != rc2) {
      throw std::runtime_error(
        "MPI_Allreduce failed even after fault recovery (allreduce 1 double ref)");
    }*/
    return ret_val;
  }

  /*if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 double)");
  }*/
  return ret_val;
}

std::tuple<int, bool> allreduce(float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Float Ptr): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }
  
  std::get<0>(ret_val) 
      = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_FLOAT, op, comm);
  // trank = caffe::mpi::comm_rank(comm);
  // tsize = caffe::mpi::comm_size(comm);
  DLOG(INFO) << "------- RETVAL int:" << std::get<0>(ret_val) << ", size" << caffe::mpi::new_size;
  if(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    caffe::mpi::fix_communicator();
    test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(test_comm);
    tsize = caffe::mpi::comm_size(test_comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Float Ptr): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_FLOAT, op, test_comm);
    DLOG(INFO) << "------- RETVAL2 int:" << std::get<0>(ret_val) << ", size" << tsize;
    // caffe::mpi::fault_global_flag = 1;
    // caffe::mpi::bcast(&caffe::mpi::fault_global_flag, 1, trank, test_comm);
    /*if (MPI_SUCCESS != rc2) {
      throw std::runtime_error(
        "MPI_Allreduce failed even after fault recovery (allreduce 1 float ptr)");
    }*/
    return ret_val;
  }

  /*if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce float)");
  }*/
  return ret_val;
}

std::tuple<int, bool> allreduce(double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  int rc, trank, tsize;
  std::tuple<int, bool> ret_val;
  std::get<1>(ret_val) = false;
  MPI_Comm test_comm;
  if (MPI_COMM_NULL == comm) {
    std::cout << "AllReduce (Double Ptr): MPI_COMM_NULL \n";
    comm = get_comm_default();
  }

  std::get<0>(ret_val)
      = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, op, comm);
  if(std::get<0>(ret_val) != MPI_SUCCESS) {
    int rc2;
    caffe::mpi::error_report(std::get<0>(ret_val), &comm);
    caffe::mpi::fix_communicator();
    test_comm = caffe::mpi::get_working_comm();
    trank = caffe::mpi::comm_rank(test_comm);
    tsize = caffe::mpi::comm_size(test_comm);
    DLOG(INFO) << "Communicator Fixed AllReduce (Double Ptr): Rank: "
      << trank << ", Size: " << tsize;
    std::get<1>(ret_val) = true;  
    std::get<0>(ret_val) = MPI_Allreduce(MPI_IN_PLACE, buffer, count, MPI_DOUBLE, op, test_comm);
    // caffe::mpi::fault_global_flag = 1;
    // caffe::mpi::bcast(&caffe::mpi::fault_global_flag, 1, trank, test_comm);
    /*if (MPI_SUCCESS != rc2) {
      throw std::runtime_error(
        "MPI_Allreduce failed even after fault recovery (allreduce 1 double ptr)");
    }*/
    return ret_val;
  }

  /*if (MPI_SUCCESS != rc) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce double)");
  }*/
  return ret_val;
}

void bcast(int* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_INT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

void bcast(float* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_FLOAT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

void bcast(double* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

int mpix_comm_replace(MPI_Comm comm, MPI_Comm* newcomm)
{
  MPI_Comm shrinked;
  MPI_Group cgrp, sgrp, dgrp;
  int rc, flag, i, nc, ns, nd, nnew, crank, srank, drank;

  int *ranks_cc, *ranks_df;

  // Shrink/remove dead process/es
  MPIX_Comm_shrink(comm, &shrinked);
  MPI_Comm_size(comm, &nc);
  MPI_Comm_size(shrinked, &ns);
  if(ns == nc)
    return 0; 
  MPI_Comm_group(comm, &cgrp); MPI_Comm_group(shrinked, &sgrp);
  // Diff is the failed ranks
  MPI_Group_difference(cgrp, sgrp, &dgrp); MPI_Group_size(dgrp, &nd);

  ranks_cc = (int*)malloc(nd * sizeof(int));
  ranks_df = (int*)malloc(nd * sizeof(int));
  for (i = 0; i < nd; i++)
    ranks_df [i] = i; 
  MPI_Group_translate_ranks(dgrp, nd, ranks_df, cgrp, ranks_cc);
  for(i=0; i< nd; i++)
    last_rank_failed = ranks_cc[i];


  // for(i=0; i<ns-(nc-nd); i++) {
  //               if( i < nd ) MPI_Group_translate_ranks(dgrp, 1, &i, cgrp, &drank);
  //              else drank=-1; /* still a spare */
                /* sending their new assignment to all spares */
                //MPI_Send(&drank, 1, MPI_INT, i+nc-nd, 1, shrinked);
  //              std::cout << "------TEST " << i+nc-nd << "\n";
  //}
  MPI_Comm_size(comm, &nc);

  std::cout << "Last Rank Failed: -----------" << last_rank_failed;
  std::cout << " ,Old Comm size:" << nc; 
  std::cout << " ,New Comm size:" << ns;
  std::cout << "\n";


  // Set Error Handler: instantiated during MPI init
  MPI_Comm_set_errhandler(shrinked, errh);
  MPI_Comm_size(shrinked, &ns);
  MPI_Comm_rank(shrinked, &srank);

  DLOG (INFO) << "Shrinking Phase: shrunk comm size:" << ns;
  caffe::mpi::new_size = ns;

  // All agree on the reduced size of communicator;
  flag = MPIX_Comm_agree(shrinked, &caffe::mpi::new_size);
  if(flag == MPI_SUCCESS) {
    DLOG(INFO) << "All Agree on reduced Comm size ------new rank:" << srank <<" , old rank " << crank; 
    duplicate_comm(newcomm, shrinked);
    //newcomm = &shrinked;
  }
  MPI_Comm_size(*newcomm, &nnew);
  std::cout << "error here!!!\n";
  DLOG (INFO) << "Shrinking Phase copy: shrunk comm size:" << nnew;

  free(shrinked);
  return flag;
}

MPI_Comm get_working_comm() {
  return wcomm;
}

// duplicate communicator
int duplicate_comm(MPI_Comm* new_comm, MPI_Comm comm)
{
  int rc;
  int flag;
  int temp_rank, temp_size;
  temp_size = caffe::mpi::comm_size(comm);
  DLOG(INFO) << "Here!------------comm_dup, before comm_size" << temp_size;

  if(comm == MPI_COMM_NULL)
  {
    std::cout << "Primary Communicator is Null. ----------\n";
  }

  //  rc = MPI_Comm_dup(MPI_COMM_WORLD, new_comm);
  // else
  rc = MPI_Comm_dup(comm, new_comm);
  flag = (MPI_SUCCESS == rc);
  if(rc != MPI_SUCCESS)
    caffe::mpi::error_report(rc, &comm);

  MPIX_Comm_agree(comm, &flag);
  if(!flag) {
    if(rc == MPI_SUCCESS) {
      MPI_Comm_free(new_comm);
      rc = MPIX_ERR_PROC_FAILED;
    }
  }
  return rc;
}

// Use the same err_code (pointer), rather than copy
void error_report(int err_code, MPI_Comm* Comm)
{
  MPI_Comm comm_ = *Comm;
  char err_string_[MPI_MAX_ERROR_STRING];
  int err_slen_, err_code_;

  err_code_ = err_code;
  MPI_Error_string(err_code_, err_string_, &err_slen_);
  std::cout << "Error Report: " << err_code_ << " ," << err_string_ << std::endl;

  /* int i, rank, size, nf, len, eclass;
  MPI_Group group_c, group_f;
  int *ranks_gc, *ranks_gf;

  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  MPIX_Comm_failure_ack(comm_);
  MPIX_Comm_failure_get_acked(comm_, &group_f);
  MPI_Group_size(group_f, &nf);
  // MPI_Error_string(err_, err_string, &len);
  // std::cout << "Rank " << rank << "/" << size << ": Notified of error "
  //  << err_string << "." << nf << "found dead: {";

  ranks_gf = (int*)malloc(nf * sizeof(int));
  ranks_gc = (int*)malloc(nf * sizeof(int));
  MPI_Comm_group(comm_, &group_c);
  for(i = 0; i < nf; i++)
    ranks_gf[i] = i;
  MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);
  for(i = 0; i < nf; i++)
    last_rank_failed = ranks_gc[i];

  std::cout << "Last Rank Failed: -----------" << last_rank_failed << "\n";

  bcast(&caffe::mpi::last_rank_failed, 1, rank, comm_); */
    // std::cout << ranks_gc[i];
  // std::cout << "}\n";
  
}

void verbose_errhandler(MPI_Comm* comm, int* err, ...)
{
  int flag;
  MPI_Comm comm_ = *comm;
  int err_ = *err;
  char err_string[MPI_MAX_ERROR_STRING];
  int i, rank, size, nf, len, eclass;
  MPI_Group group_c, group_f;
  int *ranks_gc, *ranks_gf;

  MPI_Error_class(err_, &eclass);
  if (MPIX_ERR_PROC_FAILED != eclass) {
    // Not Aborting but continuing
    MPI_Abort(comm_, err_);
    // std::cout << "Error Occured, but not aborting \n";
  }

  std::cout << "Here!-----------------------------v_errHandler" << std::endl;

  MPI_Comm_rank(comm_, &rank);
  MPI_Comm_size(comm_, &size);

  MPIX_Comm_failure_ack(comm_);
  MPIX_Comm_failure_get_acked(comm_, &group_f);
  MPI_Group_size(group_f, &nf);
  MPI_Error_string(err_, err_string, &len);
  std::cout << "Rank " << rank << "/" << size << ": Notified of error "
    << err_string << "." << nf << "found dead: {";

  ranks_gf = (int*)malloc(nf * sizeof(int));
  ranks_gc = (int*)malloc(nf * sizeof(int));
  MPI_Comm_group(comm_, &group_c);
  for(i = 0; i < nf; i++)
    ranks_gf[i] = i;
  MPI_Group_translate_ranks(group_f, nf, ranks_gf, group_c, ranks_gc);
  for(i = 0; i < nf; i++)
    std::cout << ranks_gc[i];
  std::cout << "}\n";
  //MPI_Comm_free(wcomm);
  //duplicate_comm(&wcomm, wcomm_dup);
  //MPI_Comm_free(&wcomm_dup);
}

void fix_communicator()
{
  int flag, rsize, wsize;
  int wb_rank, wa_rank;
  flag = mpix_comm_replace(wcomm, &rcomm);
  rsize = caffe::mpi::comm_size(rcomm);
  wsize = caffe::mpi::comm_size(wcomm);
  wb_rank = caffe::mpi::comm_rank(wcomm);
  if (rsize != wsize) {
    DLOG(INFO) << "Replace comm size: " << rsize;

    /*if (flag != MPI_SUCCESS) { MPI_Abort(comm_, err_); // change error message: replacement not successful }*/ // (new, old) //MPI_Comm wcomm_dup;
  
    DLOG(INFO) << "Working comm size (before switch): " << wsize << ", rank" << wb_rank;

    // MPIX_Comm_revoke(wcomm);
    // MPI_Comm_free(&wcomm);
    duplicate_comm(&wcomm, rcomm);
    wsize = caffe::mpi::comm_size(wcomm);
    wa_rank = caffe::mpi::comm_rank(wcomm);
    DLOG(INFO) << "Working comm size (after switch): " << wsize << " , rank" << wa_rank;
    MPI_Comm_set_errhandler(wcomm, MPI_ERRORS_RETURN);

    flag = MPIX_Comm_agree(wcomm, &flag);
    MPI_Comm_free(&rcomm);

    /*if (flag != MPI_SUCCESS) {
      //MPI_Comm_free(&wcomm_dup);
      //MPI_Abort(wcomm, err_); // change error message: No agreement on new communicator.
    }*/
  }
}

#else /*CAFFE_FT*/

void allreduce_copy(const float& sendbuf, float& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 float)");
  }
}

void allreduce_copy(const double& sendbuf, double& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 double)");
  }
}

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)sendbuf, recvbuf, count,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy float)");
  }
}

void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce((void*)sendbuf, recvbuf, count,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy double)");
  }
}

void allreduce(float& buffer, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, &buffer, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 float)");
  }
}

void allreduce(double& buffer, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, &buffer, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 double)");
  }
}

void allreduce(float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, buffer, count,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce float)");
  }
}

void allreduce(double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, buffer, count,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce double)");
  }
}

void bcast(float* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_FLOAT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

void bcast(double* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}
#endif /*CAFFE_FT*/

#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe
