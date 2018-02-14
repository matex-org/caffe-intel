#ifndef CAFFE_PARALLEL_GA_SYNC_CPU4_HPP_
#define CAFFE_PARALLEL_GA_SYNC_CPU4_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/stats.h"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

#include "armci.h"

namespace caffe {

// Synchronous data parallelism using Allreduce between remote CPUs.
template<typename Dtype>
class GASyncCPU4 : public Solver<Dtype>::Callback {
 public:
  explicit GASyncCPU4(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~GASyncCPU4();

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

  bool param_needs_reduce(int param_id);

  int comm_rank_;
  int comm_size_;
  shared_ptr<Solver<Dtype> > solver_;
  shared_ptr<SGDSolver<Dtype> > sgdsolver_;
  const vector<Blob<Dtype>*>& params_;
  double time_comm_;
  double time_comp_;
  stats_t stats_comm_;
  stats_t stats_comp_;
  vector<vector<Dtype*> > data_pointers_;
  vector<vector<Dtype*> > hist_pointers_;
  vector<Dtype*> data_recv_;
  vector<Dtype*> hist_recv_;
  vector<armci_hdl_t> data_hdl_;
  vector<armci_hdl_t> hist_hdl_;
  vector<shared_ptr<Layer<Dtype>>> layers_;
  shared_ptr<Net<Dtype>> net_;
  bool first_time_;
};

}  // namespace caffe

#endif
