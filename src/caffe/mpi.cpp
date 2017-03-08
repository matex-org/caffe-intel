#ifdef USE_MPI
#include <mpi.h>
#endif
#include <stdlib.h>
#include <stdexcept>
#include <unistd.h> // for gethostid()
#include <vector>
#include "caffe/mpi.hpp"

namespace caffe {
namespace mpi {

#ifdef USE_MPI

MPI_Comm default_comm_ = MPI_COMM_WORLD;

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

MPI_Comm split(int color, int key, MPI_Comm comm) {
  MPI_Comm newcomm;

  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Comm_split(comm, color, key, &newcomm)) {
    throw std::runtime_error("MPI_Comm_split failed");
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

void iallreduce(MPI_Request &request, float* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Iallreduce(MPI_IN_PLACE, buffer, count,
              MPI_FLOAT, op, comm, &request)) {
    throw std::runtime_error("MPI_Iallreduce failed (allreduce float)");
  }
}

void iallreduce(MPI_Request &request, double* buffer, int count, MPI_Op op, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Iallreduce(MPI_IN_PLACE, buffer, count,
              MPI_DOUBLE, op, comm, &request)) {
    throw std::runtime_error("MPI_Iallreduce failed (allreduce double)");
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

void waitall(std::vector<MPI_Request> &requests) {
  if (MPI_SUCCESS != MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE)) {
    throw std::runtime_error("MPI_Waitall failed");
  }
}

bool test(MPI_Request &request) {
  int flag = 0;
  if (MPI_SUCCESS != MPI_Test(&request, &flag, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Test failed");
  }
  return flag;
}

void bcast(double* buffer, int count, int root, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Bcast(buffer, count, MPI_DOUBLE, root, comm)) {
    throw std::runtime_error("MPI_Bcast failed");
  }
}

void send(const float* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Send(buffer, count, MPI_FLOAT, dest, tag, comm)) {
    throw std::runtime_error("MPI_Send failed (float)");
  }
}

void send(const double* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Send(buffer, count, MPI_DOUBLE, dest, tag, comm)) {
    throw std::runtime_error("MPI_Send failed (double)");
  }
}

void recv(float *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Recv(buffer, count, MPI_FLOAT, source, tag, comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Recv failed (float)");
  }
}

void recv(double *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Recv(buffer, count, MPI_DOUBLE, source, tag, comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Recv failed (double)");
  }
}

void sendrecv(const float *sendbuf, int sendcount, int dest, int sendtag,
    float *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm)
{
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Sendrecv(sendbuf, sendcount, MPI_FLOAT, dest, sendtag,
        recvbuf, recvcount, MPI_FLOAT, source, recvtag,
        comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Sendrecv failed (float)");
  }
}

void sendrecv(const double *sendbuf, int sendcount, int dest, int sendtag,
    double *recvbuf, int recvcount, int source, int recvtag,
    MPI_Comm comm)
{
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Sendrecv(sendbuf, sendcount, MPI_DOUBLE, dest, sendtag,
        recvbuf, recvcount, MPI_DOUBLE, source, recvtag,
        comm, MPI_STATUS_IGNORE)) {
    throw std::runtime_error("MPI_Sendrecv failed (double)");
  }
}

void isend(MPI_Request &request, const float* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Isend(buffer, count, MPI_FLOAT, dest, tag, comm, &request)) {
    throw std::runtime_error("MPI_Isend failed (float)");
  }
}

void isend(MPI_Request &request, const double* buffer, int count, int dest, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Isend(buffer, count, MPI_DOUBLE, dest, tag, comm, &request)) {
    throw std::runtime_error("MPI_Isend failed (double)");
  }
}

void irecv(MPI_Request &request, float *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Irecv(buffer, count, MPI_FLOAT, source, tag, comm, &request)) {
    throw std::runtime_error("MPI_Irecv failed (float)");
  }
}

void irecv(MPI_Request &request, double *buffer, int count, int source, int tag, MPI_Comm comm) {
  if (MPI_COMM_NULL == comm) {
    comm = get_comm_default();
  }

  if (MPI_SUCCESS != MPI_Irecv(buffer, count, MPI_DOUBLE, source, tag, comm, &request)) {
    throw std::runtime_error("MPI_Irecv failed (double)");
  }
}


#else

int dummy();

#endif

} // namespace mpi
} // namespace caffe

