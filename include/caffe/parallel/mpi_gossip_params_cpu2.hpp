#ifndef CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_CPU2_HPP_
#define CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_CPU2_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template<typename Dtype>
class MPIGossipParamsCPU2 : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIGossipParamsCPU2(shared_ptr<Solver<Dtype> > root_solver,
          const SolverParameter& param, bool cube, bool rotate);
  virtual ~MPIGossipParamsCPU2();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:

  void on_begin();
  void after_forward();
  void allreduce();
  void allreduce(int param_id);
  int on_apply(int param_id);
  void on_update();

  void next();
  void next_cube();
  void next_cube_rotate();
  void next_diffuse();
  void next_diffuse_rotate();

  void make_progress();

  int comm_rank_orig_;
  int comm_rank_;
  int comm_size_;
  int logp_;
  int hci_;
  int mci_;
  int send_pair_;
  int recv_pair_;
  shared_ptr<Solver<Dtype> > solver_;
  shared_ptr<SGDSolver<Dtype> > sgdsolver_;
  shared_ptr<AdamSolver<Dtype> > adamsolver_;
  const vector<Blob<Dtype>*>& params_;
  vector<MPI_Comm> comms_;
  vector<MPI_Request> requests_data_;
  double time_comm_;
  double time_comp_;
  stats_t stats_comm_;
  stats_t stats_comp_;
  Dtype *data_all_;
  Dtype *history_;
  Dtype *history_all_;
  size_t history_size_;
  bool cube_;
  bool rotate_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

