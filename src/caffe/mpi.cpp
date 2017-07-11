#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdlib.h>
#include <stdexcept>
#include <unistd.h> // for gethostid()
#include "caffe/mpi.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"

namespace caffe {
double allreducecopy_time=0.0;
double allreduce_time=0.0;
double bcast_time=0.0;
long unsigned int allreduce_data=0;
namespace mpi {

#ifdef USE_MPI

MPI_Comm default_comm_ = MPI_COMM_WORLD;
Timer comm_timer;

MPI_Comm get_comm_default() {
  return default_comm_;
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
  if (!initialized()) {
    int provided;
    if (MPI_SUCCESS != MPI_Init_thread(
          argc, argv, MPI_THREAD_MULTIPLE, &provided)) {

      throw std::runtime_error("MPI_Init_thread failed");
    }
  }

  if (MPI_THREAD_MULTIPLE != query_thread()) {
    throw std::runtime_error("MPI threading level must be == MPI_THREAD_MULTIPLE");
  }

  if (0 != atexit(finalize)) {
    throw std::runtime_error("atexit(caffe::mpi::finalize) failed");
  }
}

bool initialized() {
  int flag;

  if (MPI_SUCCESS != MPI_Initialized(&flag)) {
    throw std::runtime_error("MPI_Initialized failed");
  }

  return flag;
}

void finalize() {
  if (MPI_SUCCESS != MPI_Finalize()) {
    throw std::runtime_error("MPI_Finalize failed");
  }
  LOG(INFO) << "ALLREDUCE_COPY Time:" << allreducecopy_time << " s";
  LOG(INFO) << "ALLREDUCE Time:" << allreduce_time << " s";
  LOG(INFO) << "BCAST Time:" << bcast_time << " s";
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

void allreduce_copy(const float& sendbuf, float& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 float)");
  }
  allreducecopy_time += comm_timer.Seconds();
}

void allreduce_copy(const double& sendbuf, double& recvbuf, MPI_Op op,
    MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce((void*)&sendbuf, &recvbuf, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy 1 double)");
  }
  allreducecopy_time += comm_timer.Seconds();
}

void allreduce_copy(const float* sendbuf, float* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce((void*)sendbuf, recvbuf, count,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy float)");
  }
  allreducecopy_time += comm_timer.Seconds();
}

void allreduce_copy(const double* sendbuf, double* recvbuf, int count,
    MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce((void*)sendbuf, recvbuf, count,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce_copy double)");
  }
  allreducecopy_time += comm_timer.Seconds();
}

void allreduce(float& buffer, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, &buffer, 1,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 float)");
  }
  allreduce_data+=4;
  allreduce_time += comm_timer.Seconds();
}

void allreduce(double& buffer, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, &buffer, 1,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce 1 double)");
  }
  allreduce_data+=8;
  allreduce_time += comm_timer.Seconds();
}

void allreduce(float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, buffer, count,
              MPI_FLOAT, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce float)");
  }
  std::cout << count << std::endl;
  allreduce_data+=4*count;
  allreduce_time += comm_timer.Seconds();
}

void allreduce(double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Allreduce(MPI_IN_PLACE, buffer, count,
              MPI_DOUBLE, op, comm)) {
    throw std::runtime_error("MPI_Allreduce failed (allreduce double)");
  }
  allreduce_data+=8*count;
  allreduce_time += comm_timer.Seconds();
}

void bcast(float* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_FLOAT, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
  bcast_time += comm_timer.Seconds();
}

void bcast(double* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  comm_timer.Start();
  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
  bcast_time += comm_timer.Seconds();
}

#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe
