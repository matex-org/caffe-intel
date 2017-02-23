#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

#include <sstream>
#include <string>
#include <vector>

#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/mpi.hpp"
#include "caffe/parallel/mpi_sync_cpu.hpp"

namespace caffe {


// Constructor
template<typename Dtype>
MPISyncCPU<Dtype>::MPISyncCPU(shared_ptr<Solver<Dtype> > root_solver, const int rgroup_bits)
    : CPUParams<Dtype>(root_solver),
  rgroup_bits_(rgroup_bits),
//#ifdef USE_MPI
  comm_(caffe::mpi::comm_dup()),
  comm_size_(caffe::mpi::comm_size(comm_)),
  comm_rank_(caffe::mpi::comm_rank(comm_)),
  node_rank_(caffe::mpi::node_rank(comm_)),
  nodegroups(static_cast<int>(comm_size_), rgroup_bits),
  peerlist_(nodegroups.get_stagelist(comm_rank_)),
  comm_stages_(nodegroups.get_num_stages()),
  current_stage_(0),
  mergebuffer_(std::vector<Dtype>(size_+2)),
  mergebuffer2_(std::vector<Dtype>(size_+2)),
  my_group_( nodegroups.get_assigned_group_per_stage(comm_rank_) ),
  subcomm_(std::vector<MPI_Comm>(nodegroups.get_num_stages())),
  subcomm2_(std::vector<MPI_Comm>(nodegroups.get_num_stages())),
//#endif
      solver_(),
      history_(
        [&]()->const vector<shared_ptr<Blob<Dtype>>>& {
           if (!strcmp(root_solver->type(), "SGD")) {
             std::clog << "root solver is SGD" << std::endl;
             SGDSolver<Dtype>* sgd_solver = dynamic_cast<SGDSolver<Dtype>*>(root_solver.get());
             return(sgd_solver->history_);
           }
           std::cerr << "mpi_sync_cpu.cpp only configured to handle history for SGD" << std::endl;
           std::cerr << "If you are using an alternative solver, please check and add "
                     << " the appropriate history pointer reference for that solver type."
                     << std::endl;
           std::cerr << "You may need to handle additional vectors in that case." << std::endl;
           exit(99);
         }()),

      forward_map_(std::vector<std::vector<int>> (2, std::vector<int>(comm_size_))),
      reverse_map_(std::vector<std::vector<int>> (2, std::vector<int>(comm_size_))),
      current_map_index_(0),
      my_rnd_gen_(std::mt19937(1492)), // for now, hard-code seed, later take as param
      subcount_(0)
{
  std::clog << "Initializing with rgroup bits = " << rgroup_bits_ << std::endl;


  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);

  for (int j=0; j<2; j++) {
    for (int i = 0; i < comm_size_; i++) {
      forward_map_[j][i]=i;
      reverse_map_[j][i]=i;
    }
  }

#ifdef USE_MPI
  std::clog << "Sanity check: Compiled with MPI, I am node " << comm_rank_
            << ", and there are " << comm_size_ << " nodes total." << std::endl;

  std::clog << "rgroup_bits is " << rgroup_bits << std::endl;

  std::clog << "size_ is " << size_ << std::endl;

  if (comm_size_ > 1) {
    if ((0x1UL << rgroup_bits) > (comm_size_ / 2)) {
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

  if (rgroup_bits >0) {
    for (int stage=0; stage<nodegroups.get_num_stages(); stage++) {
//      MPI_Comm_split(comm_, my_group_[stage], comm_rank_, &subcomm_[stage]);


      MPI_Comm_dup(comm_, &subcomm2_[stage]);
      MPI_Comm_split(subcomm2_[stage], my_group_[stage], comm_rank_, &subcomm_[stage]);

      int this_size;
      MPI_Comm_size(subcomm_[stage], &this_size);
      subcomm_size_.push_back(this_size);

      std::clog << "[" << comm_rank_ << "] - stage " << stage
                << " is in group " << my_group_[stage] << std::endl;
    }
  }


  caffe::mpi::bcast(data_, size_, 0, comm_);

#else
  NO_MPI;
#endif
}

template<typename Dtype>
void MPISyncCPU<Dtype>::shuffle_vector(int *array_ptr, const int num_elements) {
  if (num_elements > 2) {
    for (int i=0; i<(num_elements-2); i++) {
      //int range= num_elements-i -1; // value between 0 and "range" inclusive
      std::uniform_int_distribution<int> random_element(i,(comm_size_ -1));
      const int j=random_element(my_rnd_gen_);
      const int temp = array_ptr[j];
      array_ptr[j]=array_ptr[i];
      array_ptr[i]=temp;
    }
  }
}

template<typename Dtype>
MPISyncCPU<Dtype>::~MPISyncCPU() {
}


// takes:
//    a pointer to a live buffer (holding the 60 million Alexnet parameters)
//    a pointer to a scratch buffer of the same size(+1 or 2?)  (e.g. ~60 million Alexnet parameters)
//        this should be statically allocated once at beginning of execution and then repeatedly reused
//    the number of items in the live buffer to be averaged
//    the "root node" and the partner node who are averaging their buffers
//    tag for this exchange so that there's no chance of this exchange being confused with any other

template<typename Dtype>
void MPISyncCPU<Dtype>::mpi_avg_3(Dtype * real_buffer,
                                  Dtype * temp_buffer,
                                  const size_t pcount,
                                  const int root_node,
                                  const int remote_node1,
                                  const int remote_node2,
                                  const int tag) {
  int error;
  MPI_Status status_array[3];
  MPI_Request requests[3];
  const size_t first_third_start = 0;
  const size_t first_third_count = pcount / 3;

  const size_t second_third_start = first_third_count;
  const size_t second_third_count = first_third_count;

  const size_t last_third_start = 2*first_third_count;
  const size_t last_third_count = (pcount - last_third_start);

  int remote1, remote2;
  size_t local_start, local_distance;
  size_t remote1_start, remote2_start;
  size_t remote1_distance, remote2_distance;


  if (comm_rank_ == root_node) {
    // I am root, handling first 1/3 of data from all 3 nodes
    remote1 = remote_node1;
    remote2 = remote_node2;
    local_start = first_third_start;
    local_distance = first_third_count;

    remote1_start = second_third_start;
    remote1_distance = second_third_count;

    remote2_start = last_third_start;
    remote2_distance = last_third_count;

  } else if (comm_rank_ == remote_node1) {
    // I am remote_node1 - I handle middle 1/3 of data for all 3 nodes
    remote1 = root_node;
    remote2 = remote_node2;

    local_start = second_third_start;
    local_distance = second_third_count;

    remote1_start = 0;
    remote1_distance = first_third_count;

    remote2_start = last_third_start;
    remote2_distance = last_third_count;

  } else if (comm_rank_ == remote_node2) {
    // I am remote_node2 - I handle last 1/3 of data for all 3 nodes
    remote1 = root_node;
    remote2 = remote_node1;
    local_start = last_third_start;
    local_distance = last_third_count;

    remote1_start = first_third_start;
    remote1_distance = first_third_count;

    remote2_start = second_third_start;
    remote2_distance = second_third_count;

  } else {
      std::cerr << "Error - MPI rank was neither root, remote1, or remote2." << std::endl;
      std::cerr << "MPI Rank is " << comm_rank_ << std::endl;
      std::cerr << "Root node is " << root_node << std::endl;
      std::cerr << "Remote node 1 is " << remote_node1 << std::endl;
      std::cerr << "Remote node 2 is " << remote_node2 << std::endl;
      exit(1);
  }



  // Queue a receive of 1/3 of our data to process from remote_node1 and 1/3 from remote_node2
  error = MPI_Irecv(&temp_buffer[0], local_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote1,
                    tag,
                    comm_, &requests[0]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Irecv " << std::endl;
  }


  error = MPI_Irecv(&temp_buffer[remote2_start], local_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote2,
                    tag,
                    comm_, &requests[1]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Irecv " << std::endl;
  }


  // send a 1/3 of our data to remote_node1 and another 1/3 to remote_node2
  error = MPI_Isend(&real_buffer[remote1_start], remote1_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote1,
                    tag,
                    comm_, &requests[2]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Ssend " << std::endl;

  }

  error = MPI_Isend(&real_buffer[remote2_start], remote2_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote2,
                    tag,
                    comm_, &requests[3]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Ssend " << std::endl;

  }


  error = MPI_Waitall(4, requests, status_array);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Waitall " << std::endl;
  }


  // average and update the third we have with the other 2 thirds we received
  Dtype *plocal = &real_buffer[local_start];
  const Dtype *premote1 = &temp_buffer[0];
  const Dtype *premote2 = &temp_buffer[remote2_start];
  constexpr Dtype scale = static_cast<Dtype>(1.0/3.0);
  for (size_t i=0; i<local_distance; i++) {
    const Dtype temp=(premote1[i] + premote2[i] + plocal[i]);
    plocal[i] = scale * temp;
  }



  // Get our other 2/3rds from the other nodes
  error = MPI_Irecv(&real_buffer[remote1_start], remote1_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote1,
                    tag,
                    comm_, &requests[0]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Irecv " << std::endl;
  }

  error = MPI_Irecv(&real_buffer[remote2_start], remote2_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote2,
                    tag,
                    comm_, &requests[1]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Irecv " << std::endl;
  }


  // send our 1/3 data to each of the other 2 nodes
  error = MPI_Isend(&real_buffer[local_start], local_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote1,
                    tag,
                    comm_, &requests[2]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Isend to remote_node1" << std::endl;

  }


  error = MPI_Isend(&real_buffer[local_start], local_distance,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    remote2,
                    tag,
                    comm_, &requests[3]);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Isend to remote_node2" << std::endl;

  }


  error = MPI_Waitall(4, requests, status_array);
  if (error != MPI_SUCCESS) {
    std::clog << "Error doing MPI_Waitall " << std::endl;
  }
}






// takes:
//    a pointer to a live buffer (holding the 60 million Alexnet parameters)
//    a pointer to a scratch buffer of the same size(+1 or 2?)  (e.g. ~60 million Alexnet parameters)
//        this should be statically allocated once at beginning of execution and then repeatedly reused
//    the number of items in the live buffer to be averaged
//    the "root node" and the partner node who are averaging their buffers
//    tag for this exchange so that there's no chance of this exchange being confused with any other

template<typename Dtype>
void MPISyncCPU<Dtype>::mpi_avg_2(Dtype * real_buffer,
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
 void MPISyncCPU<Dtype>::on_start() {
   const int virtual_id = reverse_map_[current_map_index_][comm_rank_];


   if (rgroup_bits_ > 0) {
     std::vector<std::vector<int>> peerlist(nodegroups.get_stagelist(virtual_id));
     std::vector<int> virtual_buddies(peerlist[current_stage_]);
     std::vector<int> buddies(virtual_buddies.size());
     for (int i=0; i<virtual_buddies.size(); i++) {
       buddies[i] = forward_map_[current_map_index_][virtual_buddies[i]];
     }
/*
     // copy internal momentum history into mergebuffer_ array in preparation
     size_t poffset=0;
     const vector<Blob < Dtype>*>&net_params = solver_->net_->learnable_params();

     for (int param_id = 0;
          param_id < solver_->net_->learnable_params().size();
          ++param_id) {
       const size_t pnum = net_params[param_id]->count();
       Dtype *ptr = (Dtype *)(history_[param_id]->cpu_data());
       for (int j=0; j<pnum; j++) {
         mergebuffer_[j+poffset] = ptr[j];
       }
       poffset = (poffset + pnum);
     }

     if (poffset != size_) {
       std::clog << "*** Error - history is wrong. ***" << std::endl;
       exit(1);
     }
*/

     // do merging
     if (buddies.size() ==2) {
       // merge data between 2 nodes using mergebuffer2_ as temp space
       mpi_avg_2((Dtype *) data_, (Dtype *) &mergebuffer2_[0], size_,
                 buddies[0], buddies[1], (current_stage_ + (0x1 << 6)));
/*
       // merge history (packed into mergebuffer_) between 2 nodes
       // using mergebuffer2_ as temp space
       mpi_avg_2((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0], size_,
                 buddies[0], buddies[1], (current_stage_ + (0x1<<7)));
*/

     } else if (buddies.size()==3) {
       // merge data between 3 nodes
       mpi_avg_3((Dtype *) data_, (Dtype *) &mergebuffer2_[0], size_,
                 buddies[0], buddies[1], buddies[2], (current_stage_ + (0x1 << 6)));
/*
       // merge history (packed into mergebuffer_) between 3 nodes
       // using mergebuffer2_ as temp space
       mpi_avg_3((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0], size_,
                 buddies[0], buddies[1], buddies[2], (current_stage_ + (0x1<<7)));
*/

     } else {
       // merge data between some other number of nodes
       const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(buddies.size()));

       MPI_Allreduce((Dtype *) data_, (Dtype *) &mergebuffer2_[0], size_,
                     ((sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                     MPI_SUM, subcomm_[current_stage_]);

       caffe_scal<Dtype>(size_, scalefactor, (Dtype *) &mergebuffer2_[0]);
       for (size_t i = 0; i < size_; i++) {
         ((Dtype *) data_)[i] = ((Dtype *) (&mergebuffer2_[0]))[i];
       }

/*
       // merge history between some other number of nodes
       MPI_Allreduce((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0],  size_,
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     MPI_SUM, subcomm_[current_stage_]);

       caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer2_[0]);
       for (size_t i=0; i<size_; i++) {
         ((Dtype *)&mergebuffer_[0])[i] = ((Dtype *)(&mergebuffer2_[0]))[i];
       }
*/
     }

////////////////////////////////
/*
     // copy merged momentum history from mergebuffer_ back
     // into the internal data structures
      poffset = 0;
      for (int param_id = 0;
           param_id < solver_->net_->learnable_params().size();
           ++param_id) {
        const size_t pnum = net_params[param_id]->count();
        Dtype *ptr = history_[param_id]->mutable_cpu_data();
        for (int j = 0; j < pnum; j++) {
          ptr[j] = mergebuffer_[j + poffset];
        }
        poffset = (poffset + pnum);
      }
*/
     // do _not_ increment group stage counter here; wait until after on_post_apply()
   } else {

     // may not need or want this reduction
     if (comm_size_ > 1) {

/*
       // do all-reduce of data across all nodes
       const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(comm_size_));
       MPI_Allreduce((Dtype *)data_, (Dtype *)&mergebuffer2_[0],  size_,
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     MPI_SUM, comm_);
       caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer2_[0]);
       for (size_t i=0; i<size_; i++) {
         ((Dtype *)data_)[i] = ((Dtype *)(&mergebuffer2_[0]))[i];
       }


       // copy internal momentum history into mergebuffer_ array in preparation
       size_t poffset=0;
       const vector<Blob < Dtype>*>&net_params = solver_->net_->learnable_params();

       for (int param_id = 0;
            param_id < solver_->net_->learnable_params().size();
            ++param_id) {
         const size_t pnum = net_params[param_id]->count();
         Dtype *ptr = (Dtype *)(history_[param_id]->cpu_data());
         for (int j=0; j<pnum; j++) {
           mergebuffer_[j+poffset] = ptr[j];
         }
         poffset = (poffset + pnum);
       }

       if (poffset != size_) {
         std::clog << "*** Error - history is wrong. ***" << std::endl;
         exit(1);
       }


       // do the history Allreduce
       MPI_Allreduce((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0],  size_,
                     ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                     MPI_SUM, comm_);
       caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer2_[0]);


       // copy merged momentum history from mergebuffer2_ back
       // into the internal data structures
       poffset = 0;
       for (int param_id = 0;
            param_id < solver_->net_->learnable_params().size();
            ++param_id) {
         const size_t pnum = net_params[param_id]->count();
         Dtype *ptr = history_[param_id]->mutable_cpu_data();
         for (int j = 0; j < pnum; j++) {
           ptr[j] = mergebuffer2_[j + poffset];
         }
         poffset = (poffset + pnum);
       }
*/
     } // if comm_size_ > 1
   }
 }


 template<typename Dtype>
 void MPISyncCPU<Dtype>::on_gradients_ready() {
   DLOG(INFO) << "on_gradients_ready()";

#define USE_MPI 1
#ifdef USE_MPI
   // Sum gradients
//  caffe::mpi::allreduce(diff_, size_, MPI_SUM, comm_);

/*
  if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      std::clog << "[" << comm_rank_ << "] pre-merge diffs: ";
      for (int i=0; i<10; i++) {
        std::clog << ((Dtype *)diff_)[i] << " ";
      }
      std::clog << std::endl;
  }
*/

  const int virtual_id = reverse_map_[current_map_index_][comm_rank_];
  if (rgroup_bits_ > 0) {
    std::vector<std::vector<int>> peerlist(nodegroups.get_stagelist(virtual_id));
    std::vector<int> virtual_buddies(peerlist[current_stage_]);
    std::vector<int> buddies(virtual_buddies.size());
    for (int i=0; i<virtual_buddies.size(); i++) {
      buddies[i] = forward_map_[current_map_index_][virtual_buddies[i]];
    }


    if (buddies.size() == 2) {
      // merge data between 2 nodes using mergebuffer2_ as temp space
      mpi_avg_2((Dtype *) diff_, (Dtype *) &mergebuffer_[0], size_,
                buddies[0], buddies[1], (current_stage_ + (0x1 << 8)));

    } else if (buddies.size() == 3) {
      // merge data between 3 nodes
      mpi_avg_3((Dtype *) diff_, (Dtype *) &mergebuffer_[0], size_,
                buddies[0], buddies[1], buddies[2], (current_stage_ + (0x1 << 8)));

    } else {
      // merge data between some other number of nodes
      const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(buddies.size()));

      MPI_Allreduce((Dtype *) diff_, (Dtype *) &mergebuffer_[0], size_,
                    ((sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_SUM, subcomm_[current_stage_]);

      caffe_scal<Dtype>(size_, scalefactor, (Dtype *) &mergebuffer_[0]);
      for (size_t i = 0; i < size_; i++) {
        ((Dtype *) diff_)[i] = ((Dtype *) (&mergebuffer_[0]))[i];
      }

    }

 // Sum gradients
   } else {
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

#else
   NO_MPI;
#endif

 }


template<typename Dtype>
void MPISyncCPU<Dtype>::on_post_apply() {
/*
  const Dtype scalefactor = (static_cast<Dtype>(1.0) /
                             static_cast<Dtype>(subcomm_size_[current_stage_]));
  SGDSolver<Dtype>* sgd_solver = dynamic_cast<SGDSolver<Dtype>*>(solver_.get());

  if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
    std::clog << "[" << comm_rank_ << "] on_post_apply(), stage "
      << current_stage_
      << " iteration " << sgd_solver->iter() << " size: " << size_
      << std::endl;
  }
  */


  if (rgroup_bits_ > 0) {

  /*

      std::vector<int> buddies(peerlist_[current_stage_]);

    // copy internal momentum history into mergebuffer_ array in preparation
    size_t poffset=0;
    const vector<Blob < Dtype>*>&net_params = solver_->net_->learnable_params();

    for (int param_id = 0;
         param_id < solver_->net_->learnable_params().size();
         ++param_id) {
      const size_t pnum = net_params[param_id]->count();
      Dtype *ptr = (Dtype *)(history_[param_id]->cpu_data());
      for (int j=0; j<pnum; j++) {
        mergebuffer_[j+poffset] = ptr[j];
      }
      poffset = (poffset + pnum);
    }

    if (poffset != size_) {
      std::clog << "*** Error - history is wrong. ***" << std::endl;
      exit(1);
    }

    // do merging
    if (buddies.size() ==2) {
      // merge data between 2 nodes using mergebuffer2_ as temp space
      mpi_avg_2((Dtype *) data_, (Dtype *) &mergebuffer2_[0], size_,
                buddies[0], buddies[1], (current_stage_ + (0x1 << 6)));

      // merge history (packed into mergebuffer_) between 2 nodes
      // using mergebuffer2_ as temp space
      mpi_avg_2((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0], size_,
                buddies[0], buddies[1], (current_stage_ + (0x1<<7)));


    } else if (buddies.size()==3) {
      // merge data between 3 nodes
      mpi_avg_3((Dtype *) data_, (Dtype *) &mergebuffer2_[0], size_,
                buddies[0], buddies[1], buddies[2], (current_stage_ + (0x1 << 6)));

      // merge history (packed into mergebuffer_) between 3 nodes
      // using mergebuffer2_ as temp space
      mpi_avg_3((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0], size_,
                buddies[0], buddies[1], buddies[2], (current_stage_ + (0x1<<7)));


    } else {
      // merge data between some other number of nodes
      const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(buddies.size()));

      MPI_Allreduce((Dtype *)data_, (Dtype *)&mergebuffer2_[0],  size_,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_SUM, subcomm_[current_stage_]);

      caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer2_[0]);
      for (size_t i=0; i<size_; i++) {
        ((Dtype *)data_)[i] = ((Dtype *)(&mergebuffer2_[0]))[i];
      }


      // merge history between some other number of nodes
      MPI_Allreduce((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0],  size_,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_SUM, subcomm_[current_stage_]);

      caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer2_[0]);
      for (size_t i=0; i<size_; i++) {
        ((Dtype *)&mergebuffer_[0])[i] = ((Dtype *)(&mergebuffer2_[0]))[i];
      }
    }

////////////////////////////////

    // copy merged momentum history from mergebuffer_ back
    // into the internal data structures
    poffset = 0;
    for (int param_id = 0;
         param_id < solver_->net_->learnable_params().size();
         ++param_id) {
      const size_t pnum = net_params[param_id]->count();
      Dtype *ptr = history_[param_id]->mutable_cpu_data();
      for (int j = 0; j < pnum; j++) {
        ptr[j] = mergebuffer_[j + poffset];
      }
      poffset = (poffset + pnum);
    }

    */


    current_stage_++;
    if (current_stage_ == comm_stages_) current_stage_=0;

    // add forward/reverse mapping stuff here too.
    //


  } else {
    /*
    // may not need or want this reduction
    if (comm_size_ > 1) {
      // do all-reduce of data across all nodes
      const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(comm_size_));
      MPI_Allreduce((Dtype *)data_, (Dtype *)&mergebuffer2_[0],  size_,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_SUM, comm_);
      caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer2_[0]);
      for (size_t i=0; i<size_; i++) {
        ((Dtype *)data_)[i] = ((Dtype *)(&mergebuffer2_[0]))[i];
      }
    }
    */

  }


}




template<typename Dtype>
void MPISyncCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization - mpi_sync_cpu (Allreduce or blocking subgroup)";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPISyncCPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPISyncCPU);

}  // namespace caffe

