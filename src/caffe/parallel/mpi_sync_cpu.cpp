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

template<typename Dtype>
MPISyncCPU<Dtype>::MPISyncCPU(shared_ptr<Solver<Dtype> > root_solver)
    : CPUParams<Dtype>(root_solver),
#ifdef USE_MPI
  comm_(caffe::mpi::comm_dup()),
  comm_size_(caffe::mpi::comm_size(comm_)),
  comm_rank_(caffe::mpi::comm_rank(comm_)),
  node_rank_(caffe::mpi::node_rank(comm_)),
  nodegroups(comm_size_, FLAGS_rgroup_bits),
  peerlist_(nodegroups.get_stagelist(comm_rank_)),
  comm_stages_(nodegroups.get_num_stages()),
  current_stage_(0),
  mergebuffer_(std::vector<Dtype>(size_)),
  mergebuffer2_(std::vector<Dtype>(size_)),
  my_group_( nodegroups.get_assigned_group_per_stage(comm_rank_) ),
  subcomm_(std::vector<MPI_Comm>(nodegroups.get_num_stages())),
  subcomm2_(std::vector<MPI_Comm>(nodegroups.get_num_stages())),
#endif
      solver_(),
      history_(
        [&]()->const vector<shared_ptr<Blob<Dtype>>>& {
           if (!strcmp(root_solver->type(), "SGD")) {
             std::clog << "root solver is SGD" << std::endl;
             SGDSolver<Dtype>* sgd_solver = dynamic_cast<SGDSolver<Dtype>*>(root_solver.get());
             return(sgd_solver->history_);
           }
           return(vector<shared_ptr<Blob<Dtype>>>());
         }()),
      subcount_(0)
{

  solver_ = root_solver;
  this->configure(solver_.get());
  solver_->add_callback(this);

  // if the root solver is type SGD, get the
  // history_ pointer.
/*  if (!strcmp(root_solver->type(), "SGD")) {
    std::clog << "root solver is SGD" << std::endl;
    SGDSolver<Dtype>* sgd_solver = dynamic_cast<SGDSolver<Dtype>*>(root_solver.get());
    phistory_ = &sgd_solver->history_;
  } else {
    phistory_ = (std::vector<<Blob<Dtype> *>> *)nullptr;
  }
*/

#ifdef USE_MPI
  std::clog << "Sanity check: Compiled with MPI, I am node " << comm_rank_
            << ", and there are " << comm_size_ << " nodes total." << std::endl;

  std::clog << "FLAGS_rgroup_bits is " << FLAGS_rgroup_bits << std::endl;

  std::clog << "size_ is " << size_ << std::endl;

  if ((0x1UL << FLAGS_rgroup_bits) > (comm_size_ / 2)) {
    std::clog << "Error - the number of reduce groups must be a power of two, and must be no more than" << std::endl;
    std::clog << "half the number of nodes." << std::endl;
    exit(1);
  }

  sleep (2*comm_rank_);
  std::clog << "\[" << comm_rank_ << "]-"
              << "There are " << comm_stages_ << " mixing stages " << std::endl;
  for (int i=0; i<nodegroups.get_num_stages(); i++) {
    std::clog << "  stage " << i << ": ";
    for (int j=0; j<peerlist_[i].size(); j++) {
      std::clog << peerlist_[i][j] << " ";
    }
    std::clog << std::endl;
  }

  if (FLAGS_rgroup_bits >0) {
    for (int stage=0; stage<nodegroups.get_num_stages(); stage++) {
//      MPI_Comm_split(comm_, my_group_[stage], comm_rank_, &subcomm_[stage]);
      MPI_Comm_split(comm_, my_group_[stage], comm_rank_, &subcomm_[stage]);
      MPI_Comm_dup(subcomm_[stage], &subcomm2_[stage]);
      int this_size;
      MPI_Comm_size(subcomm_[stage], &this_size);
      subcomm_size_.push_back(this_size);
      subcomm_size2_.push_back(this_size);
      std::clog << "[" << comm_rank_ << "] - stage " << stage
                << " is in group " << my_group_[stage] << std::endl;
    }
  }


  caffe::mpi::bcast(data_, size_, 0, comm_);



//? unsure yet
//  solver_->set_scale_on_apply(Dtype(1.0 / comm_size_));

#else
  NO_MPI;
#endif



}

template<typename Dtype>
MPISyncCPU<Dtype>::~MPISyncCPU() {
}

 template<typename Dtype>
 void MPISyncCPU<Dtype>::mpi_avg_2(Dtype * real_buffer,
                                   Dtype * temp_buffer,
                                   size_t pcount,
                                   int root_node,
                                   int remote_node,
                                   int tag) {
   int error;
   MPI_Status status;
   if (root_node == comm_rank_) {
     // I am root
     error = MPI_Ssend(real_buffer, pcount,
                       ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                       remote_node,
                       tag,
                       comm_);
     if (error != MPI_SUCCESS) {
       std::clog << "Error doing MPI_Ssend " << std::endl;

     }

     error = MPI_Recv(temp_buffer, pcount,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      remote_node,
                      tag,
                      comm_, &status);
     if (error != MPI_SUCCESS) {
       std::clog << "Error doing MPI_Recv " << std::endl;
     }

   } else {
     // I am remote

     error = MPI_Recv(temp_buffer, pcount,
                      ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                      root_node,
                      tag,
                      comm_, &status);
     if (error != MPI_SUCCESS) {
       std::clog << "Error doing MPI_Recv " << std::endl;
     }

     error = MPI_Ssend(real_buffer, pcount,
                       ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                       root_node,
                       tag,
                       comm_);
     if (error != MPI_SUCCESS) {
       std::clog << "Error doing MPI_Ssend " << std::endl;
     }
   }

   for (size_t i=0; i<pcount; i++) {
     const Dtype temp=(temp_buffer[i] + real_buffer[i]);
     real_buffer[i] = static_cast<Dtype>(0.5) * temp;
   }

 }





 template<typename Dtype>
 void MPISyncCPU<Dtype>::on_start() {
   const Dtype scalefactor = (static_cast<Dtype>(1.0) /
     static_cast<Dtype>(subcomm_size_[current_stage_]));
   SGDSolver<Dtype>* sgd_solver = dynamic_cast<SGDSolver<Dtype>*>(solver_.get());




   if (FLAGS_rgroup_bits > 0) {

     if (subcomm_size2_[current_stage_] != 2 ) {
       std::cerr << "Error - subcomm size not 2" << std::endl;
       exit(1);
     }
/*
     if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
       std::clog << "[" << comm_rank_ << "] on_start(), stage "
                 << current_stage_
                 << " iteration " << sgd_solver->iter()
                 << std::endl;
       std::clog << "[" << comm_rank_ << "] scalefactor: "
                 << scalefactor << "  size: " << size_ << std::endl;
     }


     if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
       std::clog << "[" << comm_rank_ << "]  pre-merge data: ";
       for (int i=0; i<10; i++) {
         std::clog << ((Dtype *)data_)[i] << " ";
       }
       std::clog << ",  size=" << size_;
       std::clog << std::endl;
     }
*/

     std::vector<int> buddies(peerlist_[current_stage_]);
     mpi_avg_2((Dtype *)data_, (Dtype *)&mergebuffer2_[0], size_,
               buddies[0], buddies[1], (current_stage_ + (0x1<<6)));
/*
     if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
       std::clog << "[" << comm_rank_ << "] post-merge data: ";
       for (int i=0; i<10; i++) {
         std::clog << ((Dtype *)data_)[i] << " ";
       }
       std::clog << std::endl;
     }
*/

////////////////////////////////

   // merge histories
/*
    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      for (int param_id = 0;
           param_id < solver_->net_->learnable_params().size();
           ++param_id) {
        Dtype *ptr = (Dtype *) (history_[param_id]->cpu_data());
        std::clog << "[" << comm_rank_ << "] pre merge history[" << param_id << "]: ";
        for (int i = 0; i < 10; i++) {
          std::clog << ptr[i] << " ";
        }
        std::clog << std::endl;
      }
    }
*/

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

     //std::vector<int> buddies(peerlist_[current_stage_]);
     mpi_avg_2((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0], size_,
               buddies[0], buddies[1], (current_stage_ + (0x1<<7)));



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

/*
    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      for (int param_id = 0;
           param_id < solver_->net_->learnable_params().size();
           ++param_id) {
        Dtype *ptr = (Dtype *) (history_[param_id]->cpu_data());
        std::clog << "[" << comm_rank_ << "] post-merge history[" << param_id << "]: ";
        for (int i = 0; i < 10; i++) {
          std::clog << ptr[i] << " ";
        }
        std::clog << std::endl;
      }
    }
*/

     // do _not_ increment group stage counter here; wait until after on_post_apply()
   } else {
     /*
           const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(comm_size_));
           MPI_Allreduce((Dtype *)data_, (Dtype *)&mergebuffer_[0],  size_,
                         ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                         MPI_SUM, comm_);
           caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer_[0]);
           for (size_t i=0; i<size_; i++) {
             ((Dtype *)data_)[i] = ((Dtype *)(&mergebuffer_[0]))[i];
           }
      */

   }
 }


 template<typename Dtype>
 void MPISyncCPU<Dtype>::on_gradients_ready() {
   DLOG(INFO) << "on_gradients_ready()";
   /*
   if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
     std::clog << "[" << comm_rank_ << "] on_gradients_ready(), stage "
               << current_stage_
               << " iteration " << sgd_solver->iter()
               << std::endl;
     std::clog << "[" << comm_rank_ << "] scalefactor: " << scalefactor
               << "  size:" << size_ << std::endl;
   }
   */

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

  if (FLAGS_rgroup_bits > 0) {
    //const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(subcomm_size_[current_stage_]));

    //caffe::mpi::allreduce(diff_, size_, MPI_SUM, subcomm_[current_stage_]);
    // just sum over our subgroup for this stage

/*
       std::clog << "[" << comm_rank_ << "] peerlist for stage " << current_stage_ << "  ";
      int num_peers = peerlist_[current_stage_].size();
      for (int j=0; j<num_peers; j++) {
        std::clog << peerlist_[current_stage_][j] << " ";
      }
      std::clog <<std::endl;
*/


     std::vector<int> buddies(peerlist_[current_stage_]);
     if (buddies.size() != 2) {
       std::cerr << "Error - only works for group size 2" << std::endl;
       exit(1);
     }
     mpi_avg_2((Dtype *)diff_, (Dtype *)&mergebuffer_[0], size_,
               buddies[0], buddies[1], (current_stage_ + (0x1<<8)));

/*
    MPI_Allreduce((Dtype *)diff_, (Dtype *)&mergebuffer_[0], size_,
                 ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                 MPI_SUM, subcomm2_[current_stage_]);

    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
        std::clog << "[" << comm_rank_ << "] scalefactor: " << scalefactor << std::endl;
      std::clog << "[" << comm_rank_ << "] summed diffs: ";
      for (int i=0; i<10; i++) {
        std::clog << ((Dtype *)&mergebuffer_[0])[i] << " ";
      }
      std::clog << std::endl;
    }

     caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer_[0]);

    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
        std::clog << "[" << comm_rank_ << "] scalefactor: " << scalefactor << std::endl;
      std::clog << "[" << comm_rank_ << "] averaged diffs: ";
      for (int i=0; i<10; i++) {
        std::clog << ((Dtype *)&mergebuffer_[0])[i] << " ";
      }
      std::clog << std::endl;
    }

    for (size_t i=0; i<size_; i++) {
      ((Dtype *)diff_)[i] = ((Dtype *)(&mergebuffer_[0]))[i];
    }

/*
    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
        std::clog << "[" << comm_rank_ << "] scalefactor: " << scalefactor << std::endl;
      std::clog << "[" << comm_rank_ << "] post-merge diffs: ";
      for (int i=0; i<10; i++) {
        std::clog << ((Dtype *)diff_)[i] << " ";
      }
      std::clog << std::endl;
    }
*/


 // Sum gradients
   } else {
    //    caffe::mpi::allreduce(diff_, size_, MPI_SUM, comm_);
    MPI_Allreduce((Dtype *)diff_, (Dtype *)&mergebuffer_[0],  size_,
                    ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                    MPI_SUM, comm_);

    const Dtype scalefactor = (static_cast<Dtype>(1.0) / static_cast<Dtype>(comm_size_));
    caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer_[0]);
    for (size_t i=0; i<size_; i++) {
      ((Dtype *)diff_)[i] = ((Dtype *)(&mergebuffer_[0]))[i];
    }

/*
    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
        std::clog << "[" << comm_rank_ << "] post-merge diffs: ";
        for (int i=0; i<10; i++) {
          std::clog << ((Dtype *)diff_)[i] << " ";
        }
        std::clog << std::endl;
    }
*/


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
  if (FLAGS_rgroup_bits > 0) {

/*
    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      std::clog << "[" << comm_rank_ << "] post-apply  pre merge data: ";
      for (int i=0; i<10; i++) {
        std::clog << ((Dtype *)data_)[i] << " ";
      }
      std::clog << std::endl;
    }
*/


    std::vector<int> buddies(peerlist_[current_stage_]);
    if (buddies.size() != 2) {
      std::cerr << "Error - only works for group size 2" << std::endl;
      exit(1);
    }
    mpi_avg_2((Dtype *)data_, (Dtype *)&mergebuffer2_[0], size_,
              buddies[0], buddies[1], (current_stage_ + (0x1<<9)));

/*
    //const Dtype scalefactor = (static_cast<Dtype>(1.0) /
    //  static_cast<Dtype>(subcomm_size_[current_stage_]));
    MPI_Allreduce((Dtype *)data_, (Dtype *)&mergebuffer_[0],  size_,
                  ((sizeof(Dtype) ==4) ? MPI_FLOAT : MPI_DOUBLE),
                  MPI_SUM, subcomm_[current_stage_]);

    int commsize;
    MPI_Comm_size(subcomm_[current_stage_], &commsize);
    std::clog << "[" << comm_rank_
              << "] " << " subcomm size=" << commsize << " nodes." << std::endl;

    caffe_scal<Dtype>(size_, scalefactor, (Dtype *)&mergebuffer_[0]);

    for (size_t i=0; i<size_; i++) {
      data_[i] = mergebuffer_[i];
    }
*/

/*
    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      std::clog << "[" << comm_rank_ << "] post-apply post-merge data: ";
      for (int i=0; i<10; i++) {
        std::clog << ((Dtype *)data_)[i] << " ";
      }
      std::clog << std::endl;
    }
*/

    size_t poffset=0;
    const vector<Blob < Dtype>*>&net_params = solver_->net_->learnable_params();
    for (int param_id = 0;
         param_id < solver_->net_->learnable_params().size();
         ++param_id) {
      const size_t pcount = net_params[param_id]->count();
      Dtype *ptr = (Dtype *)(history_[param_id]->cpu_data());
      for (int j=0; j<pcount; j++) {
        mergebuffer_[j+poffset] = ptr[j];
      }
      poffset = (poffset + pcount);
    }

    if (poffset != size_) {
      std::clog << "*** Error - history is wrong. ***" << std::endl;
      exit(1);
    }

    //std::vector<int> buddies(peerlist_[current_stage_]);
    mpi_avg_2((Dtype *)&mergebuffer_[0], (Dtype *)&mergebuffer2_[0], size_,
              buddies[0], buddies[1], (current_stage_ + (0x1<<10)));

    poffset = 0;
    for (int param_id = 0;
         param_id < solver_->net_->learnable_params().size();
         ++param_id) {
      const size_t pcount = net_params[param_id]->count();
      Dtype *ptr = history_[param_id]->mutable_cpu_data();
      for (int j = 0; j < pcount; j++) {
        ptr[j] = mergebuffer_[j + poffset];
      }
      poffset = (poffset + pcount);
    }


/*

    // merge histories
    size_t offset = 0;
    for (int param_id = 0;
         param_id < solver_->net_->learnable_params().size();
         ++param_id) {
      const size_t num = net_params[param_id]->count();
      Dtype *ptr = (Dtype *) (history_[param_id]->cpu_data());
      for (int j = 0; j < num; j++) {
        mergebuffer_[j + offset] = ptr[j];
      }
      offset += num;
    }

    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      for (int param_id = 0;
           param_id < solver_->net_->learnable_params().size();
           ++param_id) {
        Dtype *ptr = (Dtype *) (history_[param_id]->cpu_data());
        std::clog << "[" << comm_rank_ << "] pre merge history[" << param_id << "]: ";
        for (int i = 0; i < 10; i++) {
          std::clog << ptr[i] << " ";
        }
        std::clog << std::endl;
      }
    }

    if (offset != size_) {
      std::clog << "*** Error - history is wrong. ***" << std::endl;
      exit(1);
    }


    MPI_Allreduce((Dtype *) &mergebuffer_[0], (Dtype *) &mergebuffer2_[0],
                  size_,
                  ((sizeof(Dtype) == 4) ? MPI_FLOAT : MPI_DOUBLE),
                  MPI_SUM, subcomm_[current_stage_]);

    caffe_scal<Dtype>(size_, scalefactor, ((Dtype *) (&mergebuffer2_[0])));


    offset = 0;
    for (int param_id = 0;
         param_id < solver_->net_->learnable_params().size();
         ++param_id) {
      const size_t num = net_params[param_id]->count();
      Dtype *ptr = history_[param_id]->mutable_cpu_data();
      for (int j = 0; j < num; j++) {
        ptr[j] = mergebuffer2_[j + offset];
      }
      offset += num;
    }

    if ((comm_rank_ == 0) || (comm_rank_== 8) || (comm_rank_== 4)) {
      for (int param_id = 0;
           param_id < solver_->net_->learnable_params().size();
           ++param_id) {

        Dtype *ptr = (Dtype *) (history_[param_id]->mutable_cpu_data());
        std::clog << "[" << comm_rank_ << "] post-merge history[" << param_id << "]: ";
        for (int i = 0; i < 10; i++) {
          std::clog << ptr[i] << " ";
        }
        std::clog << std::endl;
      }
    }
*/

    current_stage_++;
    if (current_stage_ == comm_stages_) current_stage_=0;

  }


}




template<typename Dtype>
void MPISyncCPU<Dtype>::Run() {
  LOG(INFO)<< "Starting Optimization";

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

