#ifndef CAFFE_PARALLEL_MPI_SUBGROUP_NONBLOCKING_CPU_HPP_
#define CAFFE_PARALLEL_MPI_SUBGROUP_NONBLOCKING_CPU_HPP_

#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>

#include "caffe/common.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/parallel/groups.hpp"
#include "caffe/solver.hpp"
#include "caffe/sgd_solvers.hpp"


namespace caffe {

// subgroup, deferred x 1 data parallelism using subgroup exchanges between remote CPUs.
template<typename Dtype>
class MPI_subgroup_nonblocking_CPU : public CPUParams<Dtype>, public Solver<Dtype>::Callback {
 protected:
   int rgroup_bits_;
//#ifdef USE_MPI
   MPI_Comm comm_;

// #endif
   int comm_size_;
   int comm_rank_;
   int node_rank_;

   // subgroup storage
   Groups nodegroups;
   std::vector<std::vector<int>> peerlist_;
   const int comm_stages_;
   int current_stage_;
   std::vector<Dtype> data_send_buffer_;
   std::vector<Dtype> diff_send_buffer_;

   std::vector<Dtype> prior_data_;
   std::vector<Dtype> prior_diff_;
   MPI_Request prior_data_request[2];
   MPI_Request prior_gradient_request[2];


  public:
  std::vector<int> my_group_;
  friend class SGDSolver<Dtype>;
  friend class Solver<Dtype>;

  explicit MPI_subgroup_nonblocking_CPU(shared_ptr<Solver<Dtype> > root_solver, int rgroup_bits);
  virtual ~MPI_subgroup_nonblocking_CPU();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run();
  void Step(int iters);

  std::vector<MPI_Comm> subcomm_;
  std::vector<MPI_Comm> subcomm2_;
  std::vector<int> subcomm_size_;


  void mpi_avg_2(Dtype * real_buffer, Dtype * temp_buffer, size_t count,
                 int root_node, int remote_node, int tag);

  void mpi_avg_3(Dtype * real_buffer, Dtype * temp_buffer, size_t pcount,
                 int root_node, int remote_node1, int remote_node2, int tag);

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

  // use for remapping logical rank to physical
  std::vector<std::vector<int>> forward_map_;
  std::vector<std::vector<int>> reverse_map_;
  int current_map_index_;
  std::mt19937 my_rnd_gen_;

  void shuffle_vector(int *array_ptr, const int num_elements);

 private:
   size_t subcount_;
 };

}  // namespace caffe

#endif
