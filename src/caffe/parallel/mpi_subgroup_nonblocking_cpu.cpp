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
#include "caffe/parallel/mpi_subgroup_nonblocking_cpu.hpp"

namespace caffe {


// Constructor
template<typename Dtype>
MPI_subgroup_nonblocking_CPU<Dtype>::MPI_subgroup_nonblocking_CPU(shared_ptr<Solver<Dtype> > root_solver, 
                                                                  const int rgroup_bits)
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
  data_send_buffer_(std::vector<Dtype>(size_)),
  diff_send_buffer_(std::vector<Dtype>(size_)),
  prior_data_(std::vector<Dtype>(size_)),
  prior_diff_(std::vector<Dtype>(size_)),
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
      subcount_(0)
{
  std::clog << "Initializing with rgroup bits = " << rgroup_bits_ << std::endl;
  {
    double raw_log=log2(comm_size_);
    if ((raw_log -1) != rgroup_bits_) {
      std::cerr << "Error - number of nodes must be power of 2, and rgroup_bits must be "
                << "log2(# nodes) -1 for this nonblocking subgroup implementation" << std::endl;
      exit(1);
    }
  }

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);


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
MPI_subgroup_nonblocking_CPU<Dtype>::~MPI_subgroup_nonblocking_CPU() {
}


// takes:
//    a pointer to a live buffer (holding the 60 million Alexnet parameters)
//    a pointer to a scratch buffer of the same size(+1 or 2?)  (e.g. ~60 million Alexnet parameters)
//        this should be statically allocated once at beginning of execution and then repeatedly reused
//    the number of items in the live buffer to be averaged
//    the "root node" and the partner node who are averaging their buffers
//    tag for this exchange so that there's no chance of this exchange being confused with any other

template<typename Dtype>
void MPI_subgroup_nonblocking_CPU<Dtype>::mpi_avg_3(Dtype * real_buffer,
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
void MPI_subgroup_nonblocking_CPU<Dtype>::mpi_avg_2(Dtype * real_buffer,
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
 void MPI_subgroup_nonblocking_CPU<Dtype>::on_start() {}


template<typename Dtype>
void MPI_subgroup_nonblocking_CPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
  // first calculate gradients with the data we already have,
  // _then_ receive prior data and gradients, _then_ average and sum


  const int prior_stage_ = (current_stage_ > 0) ? (current_stage_ -1) : (comm_stages_ -1);
  std::vector<int> prior_buddies(peerlist_[prior_stage_]);
  std::vector<int> current_buddies(peerlist_[current_stage_]);
  const size_t prior_stage_size= prior_buddies.size();
  const size_t current_stage_size= current_buddies.size();
  // const int receive_tag = ((0x1 << 13) + prior_stage_);
  const int send_data_tag = ((0x1 << 12) + current_stage_);
  const int send_gradient_tag = ((0x1 << 13) + current_stage_);

  if ((current_stage_size !=2) || (prior_stage_size != 2)) {
    std::cerr << "Error in on_gradients_ready().  current or prior buddies list size != 2" << std::endl;
    exit(1);
  }

#define USE_MPI 1
#ifdef USE_MPI
  MPI_Status present_status[2];


// if not 1st iteration, wait until we've received prior buddy's data and
  // finished sending our data to prior buddy
  if (solver_->iter() > 0) {
    //LOG(INFO) << "Reading prior buddies' data" << std::endl;
    int error = MPI_Waitall(prior_stage_size, prior_data_request, present_status);
    if (error != MPI_SUCCESS) {
      std::clog << "Error doing MPI_Waitall in on_start()" << std::endl;
    }

    // merge current data with prior_buddies' earlier data
    //Dtype *plocal = &data_[0];
    Dtype *premote= &prior_data_[0];
    Dtype *pfinal = &data_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &data_send_buffer_[0];

    for (Dtype *plocal = &data_[0]; plocal < pfinal;) {
      const Dtype temp=(*plocal + *premote++) * 0.5;
      *plocal++ = temp;
     // *pbuffer++ = temp;
    }
  } /*else {
    Dtype *pfinal = &data_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &data_send_buffer_[0];
    for (Dtype *plocal = &data_[0]; plocal < pfinal;) {
      *pbuffer++ = *plocal++;
    }

  }*/

  // if we're not the last iteration, start the next send/receive of Data with current buddy...
  if (solver_->iter() < (solver_->max_iter()-1)) {
    int remote = (current_buddies[0] == comm_rank_)
                 ? current_buddies[1] : current_buddies[0];

    // pre-post receive for next iteration's data
    int error = MPI_Irecv((Dtype *)&prior_data_[0], size_,
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote,
                          send_data_tag,
                          comm_, &prior_data_request[0]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Irecv for data" << std::endl;
    }

    // send merged data now to current buddy before doing current gradient calculations
//    error = MPI_Isend((Dtype *)&data_send_buffer_[0], size_,
    error = MPI_Isend((Dtype *)&data_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_data_tag,
                      comm_, &prior_data_request[1]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Isend for data" << std::endl;
    }

  }


  // if not 1st iteration, wait until we've received prior buddy's diffs
  // finished sending our diffs to our prior buddy
  if (solver_->iter() > 0) {
    int error = MPI_Waitall(prior_stage_size, prior_gradient_request, present_status);
    if (error != MPI_SUCCESS) {
      std::clog << "Error doing MPI_Waitall in on_gradients_ready()" << std::endl;
    }

    // merge current diffs with prior_buddies' earlier diffs
    //Dtype *plocal = &data_[0];
    Dtype *premote= &prior_diff_[0];
    Dtype *pfinal = &diff_[(size_ + 1)];
    //Dtype *pbuffer = (Dtype *) &diff_send_buffer_[0];

    for (Dtype *plocal = &diff_[0]; plocal < pfinal;) {
      const Dtype temp=(*plocal + *premote++) * 0.5;
      *plocal++ = temp;
    //  *pbuffer++ = temp;
    }
  } /*else {
    Dtype *pfinal = &diff_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &diff_send_buffer_[0];
    for (Dtype *plocal = &diff_[0]; plocal < pfinal;) {
      *pbuffer++ = *plocal++;
    }
  }*/

  if (solver_->iter() < (solver_->max_iter()-1)) {
    // if we're not the last iteration, start MPI_Isend of merged data to current buddy...
    int remote = (current_buddies[0] == comm_rank_)
                 ? current_buddies[1] : current_buddies[0];

    // pre-post receive for next iteration's diff
    int error = MPI_Irecv((Dtype *)&prior_diff_[0], size_,
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote,
                          send_gradient_tag,
                          comm_, &prior_gradient_request[0]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Irecv for diff" << std::endl;
    }

    // send merged diffs now before doing current gradient calculations
//    error = MPI_Isend((Dtype *)&diff_send_buffer_[0], size_,
    error = MPI_Isend((Dtype *)&diff_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_gradient_tag,
                      comm_, &prior_gradient_request[1]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Isend for diff" << std::endl;
    }

  }


#else
  NO_MPI;
#endif

}

/* second version nonblocking

template<typename Dtype>
void MPI_subgroup_nonblocking_CPU<Dtype>::on_gradients_ready() {
  DLOG(INFO) << "on_gradients_ready()";
  // first calculate gradients with the data we already have,
  // _then_ receive prior data and gradients, _then_ average and sum


  const int prior_stage_ = (current_stage_ > 0) ? (current_stage_ -1) : (comm_stages_ -1);
  std::vector<int> prior_buddies(peerlist_[prior_stage_]);
  std::vector<int> current_buddies(peerlist_[current_stage_]);
  const size_t prior_stage_size= prior_buddies.size();
  const size_t current_stage_size= current_buddies.size();
  // const int receive_tag = ((0x1 << 13) + prior_stage_);
  const int send_data_tag = ((0x1 << 12) + current_stage_);
  const int send_gradient_tag = ((0x1 << 13) + current_stage_);

  if ((current_stage_size !=2) || (prior_stage_size != 2)) {
    std::cerr << "Error in on_gradients_ready().  current or prior buddies list size != 2" << std::endl;
    exit(1);
  }

#define USE_MPI 1
#ifdef USE_MPI
  MPI_Status present_status[2];


// if not 1st iteration, wait until we've received prior buddy's data and
  // finished sending our data to prior buddy
  if (solver_->iter() > 0) {
    //LOG(INFO) << "Reading prior buddies' data" << std::endl;
    int error = MPI_Waitall(prior_stage_size, prior_data_request, present_status);
    if (error != MPI_SUCCESS) {
      std::clog << "Error doing MPI_Waitall in on_start()" << std::endl;
    }

    // merge current data with prior_buddies' earlier data
    //Dtype *plocal = &data_[0];
    Dtype *premote= &prior_data_[0];
    Dtype *pfinal = &data_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &data_send_buffer_[0];

    for (Dtype *plocal = &data_[0]; plocal < pfinal;) {
      const Dtype temp=(*plocal + *premote++) * 0.5;
      *plocal++ = temp;
      *pbuffer++ = temp;
    }
  } else {
    Dtype *pfinal = &data_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &data_send_buffer_[0];
    for (Dtype *plocal = &data_[0]; plocal < pfinal;) {
      *pbuffer++ = *plocal++;
    }

  }

  // if we're not the last iteration, start the next send/receive of Data with current buddy...
  if (solver_->iter() < (solver_->max_iter()-1)) {
    int remote = (current_buddies[0] == comm_rank_)
                 ? current_buddies[1] : current_buddies[0];

    // pre-post receive for next iteration's data
    int error = MPI_Irecv((Dtype *)&prior_data_[0], size_,
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote,
                          send_data_tag,
                          comm_, &prior_data_request[0]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Irecv for data" << std::endl;
    }

    // send merged data now to current buddy before doing current gradient calculations
    error = MPI_Isend((Dtype *)&data_send_buffer_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_data_tag,
                      comm_, &prior_data_request[1]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Isend for data" << std::endl;
    }

  }


  // if not 1st iteration, wait until we've received prior buddy's diffs
  // finished sending our diffs to our prior buddy
  if (solver_->iter() > 0) {
    int error = MPI_Waitall(prior_stage_size, prior_gradient_request, present_status);
    if (error != MPI_SUCCESS) {
      std::clog << "Error doing MPI_Waitall in on_gradients_ready()" << std::endl;
    }

    // merge current diffs with prior_buddies' earlier diffs
    //Dtype *plocal = &data_[0];
    Dtype *premote= &prior_diff_[0];
    Dtype *pfinal = &diff_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &diff_send_buffer_[0];

    for (Dtype *plocal = &diff_[0]; plocal < pfinal;) {
      const Dtype temp=(*plocal + *premote++) * 0.5;
      *plocal++ = temp;
      *pbuffer++ = temp;
    }
  } else {
    Dtype *pfinal = &diff_[(size_ + 1)];
    Dtype *pbuffer = (Dtype *) &diff_send_buffer_[0];
    for (Dtype *plocal = &diff_[0]; plocal < pfinal;) {
      *pbuffer++ = *plocal++;
    }
  }

  if (solver_->iter() < (solver_->max_iter()-1)) {
    // if we're not the last iteration, start MPI_Isend of merged data to current buddy...
    int remote = (current_buddies[0] == comm_rank_)
                 ? current_buddies[1] : current_buddies[0];

    // pre-post receive for next iteration's diff
    int error = MPI_Irecv((Dtype *)&prior_diff_[0], size_,
                          ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                          remote,
                          send_gradient_tag,
                          comm_, &prior_gradient_request[0]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Irecv for diff" << std::endl;
    }

    // send merged diffs now before doing current gradient calculations
    error = MPI_Isend((Dtype *)&diff_send_buffer_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_gradient_tag,
                      comm_, &prior_gradient_request[1]);

    if (error != MPI_SUCCESS) {
      std::clog << "Error queueing MPI_Isend for diff" << std::endl;
    }

  }


#else
  NO_MPI;
#endif

}



 * /


/*  original nonblocking


template<typename Dtype>
 void MPI_subgroup_nonblocking_CPU<Dtype>::on_start() {
   // we've already verified we're safe to run with # nodes / # rgroup bits
   const int prior_stage_ = (current_stage_ > 0) ? (current_stage_ -1) : (comm_stages_ -1);
   std::vector<int> prior_buddies(peerlist_[prior_stage_]);
   std::vector<int> current_buddies(peerlist_[current_stage_]);
   const size_t prior_stage_size= prior_buddies.size();
   const size_t current_stage_size= current_buddies.size();
   //const int receive_tag = ((0x1 << 12) + prior_stage_);
   const int send_tag = ((0x1 << 12) + current_stage_);

   MPI_Status present_status[2];

   if ((current_stage_size !=2) || (prior_stage_size != 2)) {
     std::cerr << "Error in on_start.  current or prior buddies list size != 2" << std::endl;
     exit(1);
   }

   // LOG(INFO) << "on_start(), iter=" << solver_->iter() << std::endl;

   // if not 1st iteration, wait until we've received prior buddy's data and
   // finished sending our data to prior buddy
   if (solver_->iter() > 0) {
     //LOG(INFO) << "Reading prior buddies' data" << std::endl;
     int error = MPI_Waitall(prior_stage_size, prior_data_request, present_status);
     if (error != MPI_SUCCESS) {
       std::clog << "Error doing MPI_Waitall in on_start()" << std::endl;
     }

     // merge current data with prior_buddies' earlier data
     //Dtype *plocal = &data_[0];
     Dtype *premote= &prior_data_[0];
     Dtype *pfinal = &data_[(size_ + 1)];
     Dtype *pbuffer = (Dtype *) &data_send_buffer_[0];

     for (Dtype *plocal = &data_[0]; plocal < pfinal;) {
      const Dtype temp=(*plocal + *premote++) * 0.5;
      *plocal++ = temp;
      *pbuffer++ = temp;
     }
   } else {
     Dtype *pfinal = &data_[(size_ + 1)];
     Dtype *pbuffer = (Dtype *) &data_send_buffer_[0];
     for (Dtype *plocal = &data_[0]; plocal < pfinal;) {
      *pbuffer++ = *plocal++;
     }

   }


   // if we're not the last iteration, start the next send/receive with current buddy...
   if (solver_->iter() < (solver_->max_iter()-1)) {
     int remote = (current_buddies[0] == comm_rank_)
                  ? current_buddies[1] : current_buddies[0];

     // pre-post receive for next iteration's data
     int error = MPI_Irecv((Dtype *)&prior_data_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_tag,
                      comm_, &prior_data_request[0]);

     if (error != MPI_SUCCESS) {
       std::clog << "Error queueing MPI_Irecv for data" << std::endl;
     }

     // send merged data now to current buddy before doing current gradient calculations
     error = MPI_Isend((Dtype *)&data_send_buffer_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_tag,
                      comm_, &prior_data_request[1]);

     if (error != MPI_SUCCESS) {
       std::clog << "Error queueing MPI_Isend for data" << std::endl;
     }

   }


   // do _not_ increment group stage counter here; wait until after on_post_apply()

 }


 template<typename Dtype>
 void MPI_subgroup_nonblocking_CPU<Dtype>::on_gradients_ready() {
   DLOG(INFO) << "on_gradients_ready()";
   const int prior_stage_ = (current_stage_ > 0) ? (current_stage_ -1) : (comm_stages_ -1);
   std::vector<int> prior_buddies(peerlist_[prior_stage_]);
   std::vector<int> current_buddies(peerlist_[current_stage_]);
   const size_t prior_stage_size= prior_buddies.size();
   const size_t current_stage_size= current_buddies.size();
  // const int receive_tag = ((0x1 << 13) + prior_stage_);
   const int send_tag = ((0x1 << 13) + current_stage_);

#define USE_MPI 1
#ifdef USE_MPI
   MPI_Status present_status[2];
   // if not 1st iteration, wait until we've received prior buddy's data and
   // finished sending our data to prior buddy
   if (solver_->iter() > 0) {
     int error = MPI_Waitall(prior_stage_size, prior_gradient_request, present_status);
     if (error != MPI_SUCCESS) {
       std::clog << "Error doing MPI_Waitall in on_gradients_ready()" << std::endl;
     }

     // merge current diffs with prior_buddies' earlier diffs
     //Dtype *plocal = &data_[0];
     Dtype *premote= &prior_diff_[0];
     Dtype *pfinal = &diff_[(size_ + 1)];
     Dtype *pbuffer = (Dtype *) &diff_send_buffer_[0];

     for (Dtype *plocal = &diff_[0]; plocal < pfinal;) {
      const Dtype temp=(*plocal + *premote++) * 0.5;
      *plocal++ = temp;
      *pbuffer++ = temp;
     }
   } else {
     Dtype *pfinal = &diff_[(size_ + 1)];
     Dtype *pbuffer = (Dtype *) &diff_send_buffer_[0];
     for (Dtype *plocal = &diff_[0]; plocal < pfinal;) {
      *pbuffer++ = *plocal++;
     }
   }

   if (solver_->iter() < (solver_->max_iter()-1)) {
     // if we're not the last iteration, start MPI_Isend of merged data to current buddy...
     int remote = (current_buddies[0] == comm_rank_)
                  ? current_buddies[1] : current_buddies[0];

     // pre-post receive for next iteration's diff
     int error = MPI_Irecv((Dtype *)&prior_diff_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_tag,
                      comm_, &prior_gradient_request[0]);

     if (error != MPI_SUCCESS) {
       std::clog << "Error queueing MPI_Irecv for diff" << std::endl;
     }

     // send merged diffs now before doing current gradient calculations
     error = MPI_Isend((Dtype *)&diff_send_buffer_[0], size_,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote,
                      send_tag,
                      comm_, &prior_gradient_request[1]);

     if (error != MPI_SUCCESS) {
       std::clog << "Error queueing MPI_Isend for diff" << std::endl;
     }

   }


#else
   NO_MPI;
#endif

 }


 */


template<typename Dtype>
void MPI_subgroup_nonblocking_CPU<Dtype>::on_post_apply() {
  current_stage_++;
  if (current_stage_ == comm_stages_) current_stage_=0;
}




template<typename Dtype>
void MPI_subgroup_nonblocking_CPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization - MPI_subgroup_nonblocking_CPU";

  // Run root solver on current thread
  solver_->Solve();
}

template<typename Dtype>
void MPI_subgroup_nonblocking_CPU<Dtype>::Step(int iters) {
  //LOG(INFO)<< "Stepping Optimization";

  // Run root solver on current thread
  solver_->Step(iters);
}

INSTANTIATE_CLASS(MPI_subgroup_nonblocking_CPU);

}  // namespace caffe

