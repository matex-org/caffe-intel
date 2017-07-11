#ifndef CAFFE_PARALLEL_MPI_SYNC_CPU_HPP_
#define CAFFE_PARALLEL_MPI_SYNC_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Synchronous data parallelism using Allreduce between remote CPUs.
template<typename Dtype>
class MPISyncCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPISyncCPU(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~MPISyncCPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:
  void on_start();
  void on_gradients_ready();
#ifdef YY_SYNC
  void data_allreduce();
  void gradients_add();
#endif

#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
