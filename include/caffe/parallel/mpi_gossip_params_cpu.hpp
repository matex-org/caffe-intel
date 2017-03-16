#ifndef CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_CPU_HPP_
#define CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Dtype>
class MPIGossipParamsCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIGossipParamsCPU(shared_ptr<Solver<Dtype> > root_solver,
          int comm_threads, bool cube, bool avgdata, bool rotate, bool batchwise);
  virtual ~MPIGossipParamsCPU();

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

  void next();
  void next_cube();
  void next_cube_rotate();
  void next_diffuse();
  void next_diffuse_rotate();

  int comm_rank_orig_;
  int comm_rank_;
  int comm_size_;
  int logp_;
  int hci_;
  int send_pair_;
  int recv_pair_;
  shared_ptr<Solver<Dtype> > solver_;
  const vector<Blob<Dtype>*>& params_;
  BlockingQueue<int> param_solo_;
  vector<BlockingQueue<int>*> param_all_;
#ifdef USE_MPI
  vector<MPI_Comm> comms_;
#endif
  vector<Reducer*> reducers;
  Dtype *diff_all_;
  Dtype *data_all_;
  vector<Dtype*> param_diffs_;
  vector<Dtype*> param_datas_;
  bool cube_;
  bool avgdata_;
  bool rotate_;
  bool batchwise_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

