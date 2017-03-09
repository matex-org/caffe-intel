#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>

#include <sstream>
#include <string>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel/mpi_layerwise_async_const_cpu.hpp"

namespace caffe {


// Constructor
template<typename Dtype>
MPI_layerwise_async_const_CPU<Dtype>::MPI_layerwise_async_const_CPU(shared_ptr<Solver<Dtype> > root_solver,
                                                                  const bool randomize_subgroups,
                                                                  const uint64_t initial_allreduce_iterations,
                                                                  const int64_t num_subgroup_iterations_per_allreduce_block,
                                                                  const int64_t num_allreduce_iterations_per_allreduce_block)
 : CPUParams<Dtype>(root_solver),

  comm_(caffe::mpi::comm_dup()),
  comm_size_(caffe::mpi::comm_size(comm_)),
  comm_rank_(caffe::mpi::comm_rank(comm_)),
  node_rank_(caffe::mpi::node_rank(comm_)),
  rgroup_bits_( 
               [&]()->int {
                 const int num_nodes = (int) caffe::mpi::comm_size(comm_);
                 const int power_of_2 = (int) (!(num_nodes & (num_nodes-1)));
                 if (!power_of_2) { std::cerr << "not handling non-power-of-2-nodes yet" << std::endl; exit(1);}
                 const int highbit =(int)  ( 63 - __builtin_clzll(num_nodes));
                 const int max_sort_bits = (int) (highbit > 0 ? (highbit -1) : 0);
                 return(max_sort_bits);
                }()),

  params_(root_solver->net()->learnable_params()),
  num_layers_(params_.size()),
  nodegroups(static_cast<int>(comm_size_), rgroup_bits_),
  peerlist_(nodegroups.get_stagelist(comm_rank_)),
  comm_stages_(nodegroups.get_num_stages()),
  current_stage_(0),
  data_send_buffer_(std::vector<Dtype>(size_)),
  diff_send_buffer_(std::vector<Dtype>(size_)),
  
  mergebuffer_(std::vector<Dtype>(size_+2)),
  
  new_data_(2, std::vector<Dtype>(size_)),
  gradient_ready_(std::vector<std::atomic<int>>(num_layers_)),
  gradient_done_(std::vector<std::atomic<int>>(num_layers_)),
  apply_done_(std::vector<std::atomic<int>>(num_layers_)),

  prior_data_(std::vector<Dtype>(size_)),
  prior_diff_(std::vector<Dtype>(size_)),
  gradient_index_count_(0), 
  apply_index_count_(0), 
  my_group_( nodegroups.get_assigned_group_per_stage(comm_rank_) ),
  subcomm_(std::vector<MPI_Comm>(nodegroups.get_num_stages())),
  subcomm2_(std::vector<MPI_Comm>(nodegroups.get_num_stages())),

  solver_(),
  timer_(),
  time_(0.0),



  history_(
        [&]()->const vector<shared_ptr<Blob<Dtype>>>& {
           if (!strcmp(root_solver->type(), "SGD")) {
             std::clog << "root solver is SGD" << std::endl;
             SGDSolver<Dtype>* sgd_solver = dynamic_cast<SGDSolver<Dtype>*>(root_solver.get());
             return(sgd_solver->history_);
           } else {
             std::cerr << "mpi_sync_cpu.cpp only configured to handle history for SGD" << std::endl;
             std::cerr << "If you are using an alternative solver, please check and add "
                       << " the appropriate history pointer reference for that solver type."
                       << std::endl;
             std::cerr << "You may need to handle additional vectors in that case." << std::endl;
             exit(99);
           }
         }()),
  forward_map_(std::vector<std::vector<int>> (2, std::vector<int>(comm_size_))),
  reverse_map_(std::vector<std::vector<int>> (2, std::vector<int>(comm_size_))),
  current_map_index_(0),
  my_rnd_gen_(std::mt19937(1492)), // for now, hard-code seed, later take as param
  subcount_(0),
  randomize_subgroups_(randomize_subgroups),
  initial_allreduce_iterations_(initial_allreduce_iterations),
  num_subgroup_iterations_per_allreduce_block_(num_subgroup_iterations_per_allreduce_block),
  num_allreduce_iterations_per_allreduce_block_(num_allreduce_iterations_per_allreduce_block),
  background_running(0),
  stopping(0),
  background_thread_ (std::thread(&MPI_layerwise_async_const_CPU::background_task, this, num_layers_))
{
  std::clog << "Initializing mpi_layerwise_async_const_cpu with effective rgroup bits = " << rgroup_bits_ << std::endl;
  {
    double raw_log=log2(comm_size_);
    if ((raw_log -1) != rgroup_bits_) {
      std::cerr << "Error - number of nodes must be power of 2, and rgroup_bits must be "
                << "log2(# nodes) -1 for this nonblocking subgroup implementation" << std::endl;
      exit(1);
    }
  }
  if (rgroup_bits_ < 1) {
    std::cerr << "Error - layerwise_async_const with effective rgroup_bits of " << rgroup_bits_ << std::endl;
    exit(1);
  }

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);

  for (int j=0; j<2; j++) {
    for (int i = 0; i < comm_size_; i++) {
      forward_map_[j][i]=i;
      reverse_map_[j][i]=i;
    }
  }

  // ?? allocated request_ ??

#ifdef USE_MPI
  std::clog << "Sanity check: Compiled with MPI, I am node " << comm_rank_
            << ", and there are " << comm_size_ << " nodes total." << std::endl;

  std::clog << "rgroup_bits_ is " << rgroup_bits_ << std::endl;

  std::clog << "size_ is " << size_ << std::endl;

  if (comm_size_ > 1) {
    if ((0x1UL << rgroup_bits_) > (comm_size_ / 2)) {
      std::clog << "Error - the number of reduce groups must be a power of two, and must be no more than" << std::endl;
      std::clog << "half the number of nodes." << std::endl;
      exit(1);
    }
  }

  std::clog << "\[" << comm_rank_ << "]-"
              << "There are " << comm_stages_ << " mixing stages " << std::endl;
  for (int i=0; i<nodegroups.get_num_stages(); i++) {
    std::clog << "  stage " << i << ": ";
    for (int j=0; j<peerlist_[i].size(); j++) {
      std::clog << peerlist_[i][j] << " ";
    }
    std::clog << std::endl;
  }


  for (size_t i=0; i<num_layers_; i++) {
    gradient_ready_[i]=0;
    gradient_done_[i]=0;
    apply_done_[i]=0;
  }
 
#else
  NO_MPI;
#endif


  // wait until background_running becomes True
  while (!background_running) {};
  std::clog << "Background thread now running" << std::endl;
  
  caffe::mpi::bcast(data_, size_, 0, comm_);

}


template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::on_start() {
  DLOG(INFO) << "on_start()";
//  LOG(INFO) << "time comm " << time_;
  time_ = 0.0;
/*
  std::clog << "Node \[" << caffe::mpi::comm_rank(comm_) 
          << "] entering on_start(" 
          << ") for iter " << solver_->iter() << std::endl;
*/


}



template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::on_gradients_ready(int param_id) {
  DLOG(INFO) << "on_gradients_ready(param_id)";
/*
  std::clog << "Node [" << caffe::mpi::comm_rank(comm_) 
            << "] entering on_gradients_ready(param_id=" 
            << param_id << ") for iter " << solver_->iter() << ".  "
            << "blobcount is " << blob->count() << "  "
            << " using slots " << gradient_index_count_ << " to " << (gradient_index_count_ + blob->count() -1)
            << std::endl;
*/
  // Would insert trigger for background operation here, and set atomic flag for when part(param_id) is done

  // Do immediate-mode transfer, not background 
  if (solver_->iter() >= initial_allreduce_iterations_) {

    if (gradient_done_[param_id] != 0) {
      size_t waitcount=0;
      while (gradient_done_[param_id] != 0) {waitcount++;}
      std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] in on_gradients_ready("
                << "param_id=" << param_id
                << ") for iter " << solver_->iter() << " had waitcount " << waitcount << " until gradient_done_ finished resetting" << std::endl;
      fflush(stderr);
    }

    // Tell background task to start transferring
    gradient_ready_[param_id]=1;  // atomic variable transition from 1 to 0 enables copy/averaging task
    apply_done_[param_id]=0; 

   /*
    const int virtual_id = reverse_map_[current_map_index_][comm_rank_];

    std::vector<std::vector<int>> peerlist(nodegroups.get_stagelist(virtual_id));
    std::vector<int> virtual_buddies(peerlist[current_stage_]);
    std::vector<int> buddies(virtual_buddies.size());
    for (int i = 0; i < virtual_buddies.size(); i++) {
      buddies[i] = forward_map_[current_map_index_][virtual_buddies[i]];
    }

    if (buddies.size() == 2) {
      // blocking merge diffs between 2 nodes using mergebuffer_ as temp space
      mpi_avg_2(param_diff, (Dtype *) &mergebuffer_[gradient_index_count_], blob->count(),
                buddies[0], buddies[1], (current_stage_ + (0x1 << 6)));

    } else {
      std::cerr << "Error - expecting buddies.size() to be 2 but was " << buddies.size() << std::endl;
      exit(1);
    }   
    */


  }

  
}



template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::background_task(const int num_learnable_layers) {
  size_t current_stage = 0;
  
  MPI_Comm background_comm=caffe::mpi::comm_dup(comm_);
  const int background_comm_rank = caffe::mpi::comm_rank(background_comm);

  // block foreground until we've duplicated our communicators
  background_running = 1;

  size_t local_iter=0;
  int current_layer = (num_learnable_layers - 1); 
  std::clog << "Background task running with " << num_learnable_layers << " layers" << std::endl;
  fflush(stderr);
  while (!stopping) {
    current_layer = (num_learnable_layers - 1); 
    // current layer is "current_layer"


    const int virtual_id = reverse_map_[current_map_index_][background_comm_rank];

    std::vector<std::vector<int>> peerlist(nodegroups.get_stagelist(virtual_id));

    std::vector<int> virtual_buddies(peerlist[current_stage]);
    std::vector<int> buddies(virtual_buddies.size());
    for (int i = 0; i < virtual_buddies.size(); i++) {
      buddies[i] = forward_map_[current_map_index_][virtual_buddies[i]];
    }

    int remote_node;



    // gradients
    while ((!stopping) && (current_layer >= 0)) { 
      // wait for current layer gradient to be ready to average and data to be copied
      while ((!stopping) && (!gradient_ready_[current_layer])) {};   

      // find current layer stuff
      Blob<Dtype> *blob = params_[current_layer];

      const Dtype *param_diff_const = blob->cpu_diff();
     // Dtype *param_diff = blob->mutable_cpu_diff();

      const Dtype *param_data = blob->cpu_data();

      const size_t count = blob->count();
      const size_t offset = (reinterpret_cast<size_t>(param_data) - reinterpret_cast<size_t>(params_[0]->cpu_data())) / sizeof(Dtype);
/*      std::clog << "Node [" << background_comm_rank << "] "
                << " For gradient_ready parameter " 
                << current_layer 
                << " offset is " << offset 
                << " and count is " << count
                << " while size_ is " << size_
                << std::endl;
*/
      if (buddies.size() == 2) {
   
        std::clog << "Node [" << background_comm_rank << "] "
                  << " - exchange of param_id " << current_layer << " is between " 
                  << buddies[0] << " and " << buddies[1] << std::endl;
        fflush(stderr);

        if (buddies[0] == background_comm_rank) {
          // I am root
          remote_node = buddies[1];
        } else {
          remote_node = buddies[0];
        }

        std::clog << "Node [" << background_comm_rank << "] "
                  << " For gradient_ready parameter " 
                  << current_layer 
                  << " offset is " << offset 
                  << " and count is " << count
                  << " while size_ is " << size_
                  << ", exchanging diff with node " << remote_node
                  << std::endl;
        fflush(stderr);


        // blocking merge diffs between 2 nodes using mergebuffer_ as temp space
        //mpi_avg_2(param_diff, (Dtype *) &mergebuffer_[offset], count,
        //          buddies[0], buddies[1], (current_stage + (0x1 << 6)));

        // Exchange diffs
        MPI_Status status_array[2];
        MPI_Request requests[2];
        int error;

        error = MPI_Irecv((&diff_send_buffer_[offset]),
                          count,
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote_node,
                          (current_stage * (0x1 << 6))+current_layer,
                          background_comm,
                          &requests[0]);
        if (error != MPI_SUCCESS) {
          std::clog << "Error doing MPI_Irecv " << std::endl;
          fflush(stderr);
          exit(99);
        }


        error = MPI_Isend((void *)param_diff_const, count, 
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote_node,
                          (current_stage * (0x1 << 6))+current_layer,
                          background_comm,
                          &requests[1]);
        if (error != MPI_SUCCESS) {
          std::clog << "Error doing MPI_Ssend " << std::endl;
          fflush(stderr);
          exit(99);
        }


        error = MPI_Waitall(2, requests, status_array);
        if (error != MPI_SUCCESS) {
          std::clog << "Error doing MPI_Waitall " << std::endl;
          fflush(stderr);
          exit(99);
        }
 

/*
        int error;
        error = MPI_Sendrecv((void *)param_diff_const, count, 
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     remote_node,
                     (current_stage * (0x1 << 6))+current_layer,
                     (void *)(&diff_send_buffer_[offset]),
                     count,
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     remote_node,
                     (current_stage * (0x1 << 6))+current_layer,
                     background_comm,
                     &status);
        if (error != MPI_SUCCESS) {
            std::cerr << "Error doing MPI_SendRecv diffs on send_side" << std::endl;
        }
        if (status.MPI_ERROR != MPI_SUCCESS) {
          std::cerr << "Error on MPI_Sendrecv for diffs" << std::endl;
          exit(1);
        }
*/

        std::clog << "Node [" << background_comm_rank << "] "
                  << " send/receive layer " 
                  << current_layer 
                  << " diff_ completed with node " << remote_node
                  << ". Now exchanging data" 
                  << std::endl;
        fflush(stderr);

        // Exchange data
/*        error = MPI_Sendrecv((void *)param_data, count, 
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     remote_node,
                     (current_stage * (0x1 << 7))+current_layer,
                     (void *)(&data_send_buffer_[offset]),
                     count,
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     remote_node,
                     (current_stage * (0x1 << 7))+current_layer,
                     background_comm,
                     &status);
        if (error != MPI_SUCCESS) {
            std::cerr << "Error doing MPI_SendRecv data on send_side" << std::endl;
        }
        if (status.MPI_ERROR != MPI_SUCCESS) {
          std::cerr << "Error on MPI_Sendrecv for data" << std::endl;
          exit(1);
        }
*/
        error = MPI_Irecv((&data_send_buffer_[offset]),
                          count,
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote_node,
                          (current_stage * (0x1 << 7))+current_layer,
                          background_comm,
                          &requests[0]);
        if (error != MPI_SUCCESS) {
          std::clog << "Error doing MPI_Irecv " << std::endl;
          fflush(stderr);
          exit(99);
        }


        error = MPI_Isend((void *)param_data, count, 
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote_node,
                          (current_stage * (0x1 << 7))+current_layer,
                          background_comm, 
                          &requests[1]);
        if (error != MPI_SUCCESS) {
          std::clog << "Error doing MPI_Ssend " << std::endl;
          fflush(stderr);
          exit(99);
        }


        error = MPI_Waitall(2, requests, status_array);
        if (error != MPI_SUCCESS) {
          std::clog << "Error doing MPI_Waitall " << std::endl;
          fflush(stderr);
          exit(99);
        }



        std::clog << "Node [" << background_comm_rank << "] "
                  << " send/receive layer " 
                  << current_layer 
                  << " data_ exchange completed with node " << remote_node
                  << std::endl;
        fflush(stderr);
        const size_t limit = offset + count;

        // Average data and diffs, but don't move data yet
        for (size_t i = offset; i < limit; i++) {
          const Dtype temp = (data_send_buffer_[i] + data_[i]) * (Dtype)0.5;
          const Dtype temp2 = (diff_send_buffer_[i] + diff_[i]) * (Dtype)0.5;
          data_send_buffer_[i] = temp;
          diff_[i]=temp2;
        }
        std::clog << "Node [" << background_comm_rank << "] "
                  << " merging of layer " 
                  << current_layer << " done. " << std::endl;
        fflush(stderr);

        gradient_done_[current_layer]=1;

      } else {
        std::cerr << "Error - expecting buddies.size() to be 2 but was " << buddies.size() << std::endl;
        fflush(stderr);
        exit(9);
      }   

      current_layer = (current_layer -1);
    }; // while - on_gradients_ready(current_level) 


    // on_apply(param_id) part - let foreground thread do the data apply; we've done background gradient apply already
    current_layer = (num_learnable_layers - 1); 

    while ((!stopping) && (current_layer >= 0)) { 
      // wait for current layer gradient to be ready to average and data to be copied
      while ((!stopping) && (!apply_done_[current_layer])) {};   
      if (!stopping) { 
        gradient_done_[current_layer]=0;
        gradient_ready_[current_layer]=0;
        current_layer = (current_layer -1);
      }
    }; // while for applies 

    if (!stopping) {
      local_iter++;
      // put remapping parts here rather than in the on_post_apply() section
      std::clog << "Node [" << background_comm_rank << "] "
                << "Background iter " << local_iter << " completed." << std::endl;
      fflush(stderr);
      current_stage++;

      if (current_stage == comm_stages_) {
        current_stage = 0;
        const int next_map_index = 1 - current_map_index_;
        for (int i = 0; i < comm_size_; i++) {
          forward_map_[next_map_index][i] = forward_map_[current_map_index_][i];
          reverse_map_[next_map_index][i] = reverse_map_[current_map_index_][i];
        }
        if (randomize_subgroups_) {
          shuffle_vector((int *) &forward_map_[next_map_index][0], comm_size_);
          for (int i = 0; i < comm_size_; i++) {
            reverse_map_[next_map_index][forward_map_[next_map_index][i]] = i;
          }
        }
        current_map_index_ = next_map_index;
      }
    } else { 
      std::clog << "Node [" << background_comm_rank << "] "
                << "Background stopping after " << local_iter << " iterations." << std::endl;
      fflush(stderr);
    } // if not stopping
  } // while not stopping

  std::clog << "Node [" << background_comm_rank << "] "
            << "Background task got stop signal after completing " << local_iter << " iterations " << std::endl;
  fflush(stderr);

  background_running =0;
  return;
}


template<typename Dtype>
int MPI_layerwise_async_const_CPU<Dtype>::on_apply(int param_id) {
/*
  std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] entering on_apply("
            << "param_id=" << param_id
            << ") for iter " << solver_->iter() << std::endl;
*/
  Blob<Dtype> *blob = params_[param_id];
  Dtype *param_data = blob->mutable_cpu_data();
  const size_t count = blob->count();
  const size_t offset = (reinterpret_cast<size_t>(param_data) - reinterpret_cast<size_t>(params_[0]->cpu_data())) / sizeof(Dtype);

  // If (and only if) clip_gradients < 0 then we could put wait( part(param_id) complete) here so that background 
  // transmission done.  ? whether we actually can update _data_ (param_id)  before this point or not ?

  if (solver_->iter() >= initial_allreduce_iterations_) {

     // spin-wait until gradients averaged and data copied
     if (!gradient_done_[param_id]) {
       std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] in on_apply("
                 << " waiting for gradient_done_ for param_id=" << param_id
                 << ") on iter " << solver_->iter() << std::endl;
       fflush(stderr);
     
       size_t waitcount=0;
       while (!gradient_done_[param_id]) {waitcount++;};
        std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] in on_apply("
                  << "param_id=" << param_id
                  << ") for iter " << solver_->iter() << " had waitcount " << waitcount << std::endl;
        fflush(stderr);
    } else {
       std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] in on_apply("
                 << " param_id=" << param_id
                 << ")  had no wait for gradient_done_ on iter " << solver_->iter() << std::endl;
       fflush(stderr);
    }

    // todo: use new_data_ instead and double-buffer pointers rather than bulk copying
    std::memcpy(&data_[offset], (Dtype *)&data_send_buffer_[offset], (count * sizeof(Dtype)));
    apply_done_[param_id]=1; 

  }

  return param_id;
}



template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::shuffle_vector(int *array_ptr, const int num_elements) {
  if (num_elements > 2) {
    for (int i=0; i<(num_elements-2); i++) {
      std::uniform_int_distribution<int> random_element(i,(comm_size_ -1));
      const int j=random_element(my_rnd_gen_);
      const int temp = array_ptr[j];
      array_ptr[j]=array_ptr[i];
      array_ptr[i]=temp;
    }
  }
}


template<typename Dtype>
MPI_layerwise_async_const_CPU<Dtype>::~MPI_layerwise_async_const_CPU() {
  stopping =1;
  std::clog << "Signalling background task." << std::endl;
  while (background_running) {};
  std::clog << "Background task ending." << std::endl;
  background_thread_.join();
  std::clog << "Background task ended." << std::endl;
}





// takes:
//    a pointer to a live buffer (holding the 60 million Alexnet parameters)
//    a pointer to a scratch buffer of the same size(+1 or 2?)  (e.g. ~60 million Alexnet parameters)
//        this should be statically allocated once at beginning of execution and then repeatedly reused
//    the number of items in the live buffer to be averaged
//    the "root node" and the partner node who are averaging their buffers
//    tag for this exchange so that there's no chance of this exchange being confused with any other

template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::mpi_avg_2(Dtype * real_buffer,
                                  Dtype * temp_buffer,
                                  const size_t pcount,
                                  const int root_node,
                                  const int remote_node,
                                  const int tag) {
  int error;
  MPI_Status status_array[2];
  MPI_Request requests[2];
  const size_t midway = pcount >> 1;
  const size_t remainder = (pcount - midway);

  int remote;
  size_t start, distance;
  size_t remote_start, remote_distance;

  if (root_node == comm_rank_) {
    // I am root
    remote = remote_node;
    start=0;
    distance=midway;
    remote_start=midway;
    remote_distance=remainder;
  } else {
    remote = root_node;
    start=midway;
    distance=remainder;
    remote_start=0;
    remote_distance=midway;
  }


  // Give half data
  error = MPI_Irecv(&temp_buffer[0], remote_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote,
                    tag,
                    comm_, &requests[0]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Irecv " << std::endl;
  }


  error = MPI_Isend(&real_buffer[start], distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote,
                    tag,
                    comm_, &requests[1]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Ssend " << std::endl;

  }


  error = MPI_Waitall(2, requests, status_array);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Waitall " << std::endl;
  }


  // average and update half we have
  Dtype *plocal = &real_buffer[remote_start];
  for (size_t i=0; i<remote_distance; i++) {
    const Dtype temp=(temp_buffer[i] + plocal[i]);
    plocal[i] = static_cast<Dtype>(0.5) * temp;
  }


  // Exchange half data computed by partner
  error = MPI_Irecv(&real_buffer[start], distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote,
                    tag,
                    comm_, &requests[0]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Irecv " << std::endl;
  }


  error = MPI_Isend(&real_buffer[remote_start], remote_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote,
                    tag,
                    comm_, &requests[1]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Ssend " << std::endl;

  }


  error = MPI_Waitall(2, requests, status_array);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Waitall " << std::endl;
  }
}






template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
/*

   std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] entering on_gradients_ready() for iter " << solver_->iter() << std::endl;
*/
  // first calculate gradients with the data we already have,
  // _then_ receive prior data and gradients, _then_ average and sum

  // initial warm-up case
  if (solver_->iter() < initial_allreduce_iterations_) {
   std::clog << "Node [" << caffe::mpi::comm_rank(comm_) << "] doing warm_up in on_gradients_ready() for iter " << solver_->iter() << std::endl;
  fflush(stderr);
    if (comm_size_ > 1) {
      MPI_Allreduce((Dtype *) diff_, (Dtype *) &mergebuffer_[0], size_,
                    ((sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_SUM, comm_);

      const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(comm_size_));
      caffe_scal<Dtype>(size_, scalefactor, (Dtype *) &mergebuffer_[0]);
      for (size_t i = 0; i < size_; i++) {
        ((Dtype *) diff_)[i] = ((Dtype *) (&mergebuffer_[0]))[i];
      }
    }
  }

}


template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::on_post_apply() {
/*
  if (solver_->iter() >= initial_allreduce_iterations_) {
    current_stage_++;

    if (current_stage_ == comm_stages_) {
      current_stage_ = 0;
      const int next_map_index = 1 - current_map_index_;
      for (int i = 0; i < comm_size_; i++) {
        forward_map_[next_map_index][i] = forward_map_[current_map_index_][i];
        reverse_map_[next_map_index][i] = reverse_map_[current_map_index_][i];
      }
      if (randomize_subgroups_) {
        shuffle_vector((int *) &forward_map_[next_map_index][0], comm_size_);
        for (int i = 0; i < comm_size_; i++) {
          reverse_map_[next_map_index][forward_map_[next_map_index][i]] = i;
        }
      }
      current_map_index_ = next_map_index;
    }

  }

*/
}




template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization - MPI_layerwise_async_const_CPU";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPI_layerwise_async_const_CPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPI_layerwise_async_const_CPU);

}  // namespace caffe

