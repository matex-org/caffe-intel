#ifndef CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_CPU10_HPP_
#define CAFFE_PARALLEL_MPI_GOSSIP_PARAMS_CPU10_HPP_

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
class MPIGossipParamsCPU10 : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 public:
  explicit MPIGossipParamsCPU10(shared_ptr<Solver<Dtype> > root_solver,
          const SolverParameter& param, bool cube, bool rotate);
  virtual ~MPIGossipParamsCPU10();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

 protected:

  void on_start();
  void on_forward(int param_id);
  void on_gradients_ready(int param_id);
  void on_gradients_ready();
  int on_apply(int param_id);
  void on_update();

  void next();
  void next_cube();
  void next_cube_rotate();
  void next_diffuse();
  void next_diffuse_rotate();

  void make_progress();

  bool param_needs_reduce(int param_id);

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
  vector<MPI_Request> requests_;
  double time_comm_;
  double time_comp_;
  stats_t stats_comm_;
  stats_t stats_comp_;
  vector<Dtype*> data_recv_;
  vector<Dtype*> hist_recv_;
  vector<shared_ptr<Layer<Dtype>>> layers_;
  vector<vector<int>> layer_param_ids_;
  shared_ptr<Net<Dtype>> net_;
  bool cube_;
  bool rotate_;
  int state_;
};

}  // namespace caffe

#endif

