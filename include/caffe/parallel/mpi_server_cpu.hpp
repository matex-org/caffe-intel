#ifndef CAFFE_PARALLEL_MPI_SERVER_CPU_HPP_
#define CAFFE_PARALLEL_MPI_SERVER_CPU_HPP_

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Asynchronous data parallelism using param server between ranks. 
// Rank 0 is server.
template<typename Dtype>
class MPIServerCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIServerCPU(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~MPIServerCPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:
  void on_start();
  void on_gradients_ready();

#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int comm_rank_;
  int comm_size_;
  int current_worker_;
  int last_worker_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
