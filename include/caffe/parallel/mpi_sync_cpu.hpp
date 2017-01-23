#ifndef CAFFE_PARALLEL_MPI_SYNC_CPU_HPP_
#define CAFFE_PARALLEL_MPI_SYNC_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/parallel/groups.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"

DECLARE_uint64(rgroup_bits);

namespace caffe {

// Synchronous data parallelism using Allreduce between remote CPUs.
template<typename Dtype>
class MPISyncCPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 protected:
#ifdef USE_MPI
   MPI_Comm comm_;
#endif
   int comm_size_;
   int comm_rank_;
   int node_rank_;

   // subgroup storage
   Groups nodegroups;
   std::vector<std::vector<int>> peerlist_;
   const int comm_stages_;
   int current_stage_;
   std::vector<Dtype> mergebuffer_;
   std::vector<Dtype> mergebuffer2_;

  public:
  std::vector<int> my_group_;
  friend class SGDSolver<Dtype>;
  friend class Solver<Dtype>;

  explicit MPISyncCPU(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~MPISyncCPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

  std::vector<MPI_Comm> subcomm_;
  std::vector<MPI_Comm> subcomm2_;
  std::vector<int> subcomm_size_;
  std::vector<int> subcomm_size2_;

  void mpi_avg_2(Dtype * real_buffer, Dtype * temp_buffer, size_t count, int root_node, int remote_node, int tag);


 protected:
  void on_start();
  void on_gradients_ready();
  void on_post_apply();

  shared_ptr<Solver<Dtype> > solver_;

  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  //using Solver<Dtype>::net_;

  //Dtype* history_; // optionally defined if root solver is "SGD" (?)
  const vector<shared_ptr<Blob<Dtype>>> history_;

  private:
   size_t subcount_;
 };

}  // namespace caffe

#endif
