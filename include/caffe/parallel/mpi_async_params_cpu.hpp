#ifndef CAFFE_PARALLEL_MPI_ASYNC_PARAMS_CPU_HPP_
#define CAFFE_PARALLEL_MPI_ASYNC_PARAMS_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Asynchronous data parallelism using Allreduce between remote CPUs.
template<typename Dtype>
class MPIAsyncParamsCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIAsyncParamsCPU(shared_ptr<Solver<Dtype> > root_solver,
          int comm_threads);
  virtual ~MPIAsyncParamsCPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

  friend class Reducer;

 protected:
  class Reducer;

  void on_start();
  void on_gradients_ready();
  void on_gradients_ready(int param_id);
  int on_apply(int param_id);

  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;
  const vector<Blob<Dtype>*>& params_;
  BlockingQueue<int> param_solo_;
  vector<BlockingQueue<int>*> param_all_;
#ifdef USE_MPI
  vector<MPI_Comm> comms_;
#endif
  vector<Reducer*> reducers;
  Dtype *diff_all_;
  vector<Dtype*> param_diffs_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

