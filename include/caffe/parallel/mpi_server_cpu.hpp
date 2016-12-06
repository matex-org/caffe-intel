#ifndef CAFFE_PARALLEL_MPI_SERVER_CPU_HPP_
#define CAFFE_PARALLEL_MPI_SERVER_CPU_HPP_

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

namespace caffe {

/**
 * @brief Solver that only applies gradients, gradients supplied by workers
 *        for multi-CPU training.
 */
template <typename Dtype>
class SGDSolverServer : public SGDSolver<Dtype> {
 public:
  explicit SGDSolverServer(const SolverParameter& param);
  virtual ~SGDSolverServer();

  virtual void Step(int iters);

 protected:
  void ApplyUpdate();
  void ApplyUpdate(int param_id);
  void SnapshotSolverState(const string& model_filename);
  void RestoreSolverStateFromBinaryProto(const string& state_file);
  void RestoreSolverStateFromHDF5(const string& state_file);

  using SGDSolver<Dtype>::GetRequestedAction;
  using SGDSolver<Dtype>::ForwardBackward;
  using SGDSolver<Dtype>::Snapshot;
  using SGDSolver<Dtype>::TestAll;
  using SGDSolver<Dtype>::UpdateSmoothedLoss;

  using SGDSolver<Dtype>::param;

  using SGDSolver<Dtype>::callbacks_;
  using SGDSolver<Dtype>::forward_backward_;
  using SGDSolver<Dtype>::iter_;
  using SGDSolver<Dtype>::iteration_timer_;
  using SGDSolver<Dtype>::iterations_last_;
  using SGDSolver<Dtype>::losses_;
  using SGDSolver<Dtype>::net_;
  using SGDSolver<Dtype>::param_;
  using SGDSolver<Dtype>::requested_early_exit_;
  using SGDSolver<Dtype>::smoothed_loss_;

#ifdef USE_MPI
  MPI_Comm comm_;
#endif
  int comm_rank_;
  int comm_size_;
  int current_worker_;
  const size_t size_;
  Dtype *data_;
  Dtype *diff_;
};


}  // namespace caffe

#endif
