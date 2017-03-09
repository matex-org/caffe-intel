#ifndef CAFFE_PARALLEL_MPI_LAYERWISE_ASYNC_CONST_CPU_HPP_
#define CAFFE_PARALLEL_MPI_LAYERWISE_ASYNC_CONST_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <cstdio>

#include <vector>
#include <atomic>
#include <thread>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/parallel/groups.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"


namespace caffe {

// subgroup, deferred x 1 data parallelism using subgroup exchanges between remote CPUs.
template<typename Dtype>
class MPI_layerwise_async_const_CPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 protected:
//#ifdef USE_MPI
   MPI_Comm comm_;

// #endif
   int comm_size_;
   int comm_rank_;
   const int node_rank_;
   const int rgroup_bits_;
   const vector<Blob<Dtype>*> &params_;

   const size_t num_layers_;

   // subgroup storage
   Groups nodegroups;
   std::vector<std::vector<int>> peerlist_;
   const int comm_stages_;
   int current_stage_;
   std::vector<Dtype> data_send_buffer_;
   std::vector<Dtype> diff_send_buffer_;


   std::vector<Dtype> mergebuffer_;
   std::vector<std::vector<Dtype>> new_data_;

   std::vector<std::atomic<int>> gradient_ready_;
   std::vector<std::atomic<int>> gradient_done_;
   std::vector<std::atomic<int>> apply_done_;

   std::vector<Dtype> prior_data_;
   std::vector<Dtype> prior_diff_;
   MPI_Request prior_data_request[2];
   MPI_Request prior_gradient_request[2];
   size_t gradient_index_count_;
   size_t apply_index_count_;

  public:
  std::vector<int> my_group_;
  friend class SGDSolver<Dtype>;
  friend class Solver<Dtype>;
  explicit MPI_layerwise_async_const_CPU (shared_ptr<Solver<Dtype> > root_solver,
                                          const bool randomize_subgroups,
                                          const uint64_t initial_allreduce_iterations,
                                          const int64_t num_subgroup_iterations_per_allreduce_block,
                                          const int64_t num_allreduce_iterations_per_allreduce_block
  );
  virtual ~MPI_layerwise_async_const_CPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

  void background_task(const int num_learnable_layers);

  std::vector<MPI_Comm> subcomm_;
  std::vector<MPI_Comm> subcomm2_;
  std::vector<int> subcomm_size_;


  void mpi_avg_2(Dtype * real_buffer, Dtype * temp_buffer, size_t count,
                 int root_node, int remote_node, int tag);

 // void mpi_avg_3(Dtype * real_buffer, Dtype * temp_buffer, size_t pcount,
 //                int root_node, int remote_node1, int remote_node2, int tag);



 protected:
  void on_start();
  void on_gradients_ready(int param_id);
  void on_gradients_ready();
  int on_apply(int param_id);
  void on_post_apply();

  shared_ptr<Solver<Dtype> > solver_;

  Timer timer_;
  double time_;


  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
  //using Solver<Dtype>::net_;

  //Dtype* history_; // optionally defined if root solver is "SGD" (?)
  const vector<shared_ptr<Blob<Dtype>>> history_;

  // use for remapping logical rank to physical
  std::vector<std::vector<int>> forward_map_;
  std::vector<std::vector<int>> reverse_map_;
  std::atomic<int> current_map_index_;
  std::mt19937 my_rnd_gen_;

  void shuffle_vector(int *array_ptr, const int num_elements);

 private:
   size_t subcount_;
   const bool randomize_subgroups_;
   const uint64_t initial_allreduce_iterations_;
   const int64_t num_subgroup_iterations_per_allreduce_block_;
   const int64_t num_allreduce_iterations_per_allreduce_block_;

 public:
  std::atomic<int> background_running;
  std::atomic<int> stopping;
  std::thread background_thread_;
 };

}  // namespace caffe

#endif
