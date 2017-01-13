#ifndef CAFFE_PARALLEL_MPI_SYNC_PARAMS_CPU_HPP_
#define CAFFE_PARALLEL_MPI_SYNC_PARAMS_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Synchronous data parallelism using Allreduce between remote CPUs.
template<typename Dtype>
class MPISyncParamsCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPISyncParamsCPU(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~MPISyncParamsCPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:
  void on_start();
  void on_gradients_ready();
  void on_apply(int param_id);

#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;
  const vector<Blob<Dtype>*>& params_;
  Timer timer_;
  double time_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif
