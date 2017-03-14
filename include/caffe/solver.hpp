/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver_factory.hpp"

#include "caffe/util/benchmark.hpp"

#ifdef ADAPTIVE_BATCH
#include "caffe/mpi.hpp"
#include <assert.h>
#include <queue>
#include <type_traits>
#include <random>
#endif

namespace caffe {

/**
  * @brief Enumeration of actions that a client of the Solver may request by
  * implementing the Solver's action request function, which a
  * a client may optionally provide in order to request early termination
  * or saving a snapshot without exiting. In the executable caffe, this
  * mechanism is used to allow the snapshot to be saved when stopping
  * execution with a SIGINT (Ctrl-C).
  */
  namespace SolverAction {
    enum Enum {
      UNKNOWN = -1,
      NONE = 0,  // Take no special action.
      STOP = 1,  // Stop training. snapshot_after_train controls whether a
                 // snapshot is created.
      SNAPSHOT = 2  // Take a snapshot, and keep training.
    };
  }

#ifdef ADAPTIVE_BATCH
namespace AdaptiveBatchOption {  // Should be friend class to Solver Class. 
  struct Random {}; // Random value between 1 and max_iter size.
  struct RatioCToC {}; // Ratio of communication to computation.
  struct LossRate {}; // Rate of change of Loss > threshhold, increase batch size. 
}

template <typename Option>
struct NewBatchSize {
  // Random Itersize. Input max value; min value is 1. 
  template<typename U = Option, 
    typename = typename std::enable_if<std::is_same<
                        U, AdaptiveBatchOption::Random>::value, U>::type>
  static int get(int upperLimit, std::mt19937& gen) {
    int random = 0;
    std::uniform_int_distribution<int> dist(1, upperLimit);

    random = dist(gen);// dist(rSeed);
    DLOG(INFO) << "RANDOM_UPPER_LIMIT" << upperLimit;
    DLOG(INFO) << "RANDOM_NUMBER GENERATED" << random; 
    return random;
  }

  // Ratio Communication/Computation. 
  // Data-History approach has Mininum 2 communication if computation is 1.
  template<typename U = Option, 
    typename = 
      typename std::enable_if<std::is_same<
                  U, AdaptiveBatchOption::RatioCToC>::value, U>::type,
      typename Dtype>
  static int get( std::deque<Dtype>& deltaLosses
                , std::deque<double>& commTimes
                , std::deque<double>& commCompTimes
                , float lossThres
                , float CToCThres
                // , float currentCToC
                , int& batchApplyIter
                 ) {
    // std::assert(commTimes.size() == commCompTimes.size());
    // int new_batchsize = 0; 
    //if()
    // return new_batchsize;

    // Communication to (communication + computation) ratio
    // Sum of most recent of the last 20 recorded values;
    double sumComm = 0, sumCommComp = 0;
    for (auto c : commTimes)
      sumComm += c;

    for (auto c2 : commCompTimes)
      sumCommComp += c2;

    double CToCRatio = sumComm / sumCommComp;
    
    // Second half
    Dtype deltaAvg1 = 0;
      for (int i = 0; i < 10; ++i)
        deltaAvg1 += deltaLosses[i];
      deltaAvg1 = deltaAvg1/(0.5 * deltaLosses.size());
    // First half
    Dtype deltaAvg2 = 0;
    for (int i = 10; i < 20; ++i)
      deltaAvg2 += deltaLosses[i];
    deltaAvg2 = deltaAvg2/(0.5 * deltaLosses.size());

    // Dtype trendAvg = (deltaAvg1 + deltaAvg2)/ 2;
    Dtype trendDiff = deltaAvg2 - deltaAvg1; 
    Dtype trendAcc = (deltaAvg2 - deltaAvg1)/deltaLosses.size();

    // if( (trendDiff > 0) && trendDiff > lossThres) {
    if( (trendAcc > 0) 
      && (trendAcc > lossThres)
      && (CToCRatio > CToCThres)
      ) {
      return batchApplyIter + 4; // fixed increment size;  
    }
    //else if ((trendDiff > 0) && trendDiff <= lossThres ){
    else if ((trendAcc > 0) 
            // && (trendAcc <= lossThres 
              && (trendAcc > (0.9 * lossThres))
              && (CToCRatio > CToCThres)
              ){
      return batchApplyIter + 1; // continue with same batch size;
    }
    else if ((trendAcc > 0) 
            // && (trendAcc <= lossThres 
              && (trendAcc > (0.9 * lossThres))
              && (CToCRatio < CToCThres)
              ){
      return batchApplyIter; // continue with same batch size;
    }
    else {
      // if(batchApplyIter > 1) {
        //return (batchApplyIter - 1); // decrease batch size; 
      //}
      return 1;
    }
  }

  template<typename U = Option, 
    typename = typename std::enable_if<std::is_same<
                        U, AdaptiveBatchOption::LossRate>::value, U>::type,
    typename Dtype>
  static int get(std::deque<Dtype>& deltaLosses, float lossThres, int& batchApplyIter) {
    // if(deltaLosses.size() > 1)
    // {
      Dtype deltaAvg1 = 0;
      for (int i = 0; i < 10; ++i)
        deltaAvg1 += deltaLosses[i];
      deltaAvg1 = deltaAvg1/(0.5 * deltaLosses.size());
      
      Dtype deltaAvg2 = 0;
      for (int i = 10; i < 20; ++i)
        deltaAvg2 += deltaLosses[i];
      deltaAvg2 = deltaAvg2/(0.5 * deltaLosses.size());

      // Dtype trendAvg = (deltaAvg1 + deltaAvg2)/ 2;
      Dtype trendDiff = deltaAvg2 - deltaAvg1; 
      Dtype trendAcc = (deltaAvg2 - deltaAvg1)/deltaLosses.size();

      // if( (trendDiff > 0) && trendDiff > lossThres) {
      if( (trendAcc > 0) && trendAcc > lossThres) {
        return (batchApplyIter + 1); // fixed increment size;  
      }
      //else if ((trendDiff > 0) && trendDiff <= lossThres ){
      else if ((trendAcc > 0) 
              // && (trendAcc <= lossThres 
                  && trendAcc > (0.9 * lossThres)){
        return batchApplyIter; // continue with same batch size;
      }
      else {
        // if(batchApplyIter > 1) {
          //return (batchApplyIter - 1); // decrease batch size; 
        //}
        return 1;
      }
  }
};
#endif

/**
 * @brief Type of a function that returns a Solver Action enumeration.
 */
typedef boost::function<SolverAction::Enum()> ActionCallback;

/**
 * @brief An interface for classes that perform optimization on Net%s.
 *
 * Requires implementation of ApplyUpdate to compute a parameter update
 * given the current state of the Net parameters.
 */
template <typename Dtype>
class Solver {
 public:
  explicit Solver(const SolverParameter& param,
      const Solver* root_solver = NULL);
  explicit Solver(const string& param_file, const Solver* root_solver = NULL);
  void Init(const SolverParameter& param);
  void InitTrainNet();
  void InitTestNets();

  // Client of the Solver optionally may call this in order to set the function
  // that the solver uses to see what action it should take (e.g. snapshot or
  // exit training early).
  void SetActionFunction(ActionCallback func);
  SolverAction::Enum GetRequestedAction();
  // The main entry of the solver function. In default, iter will be zero. Pass
  // in a non-zero iter number to resume training for a pre-trained net.
  virtual void Solve(const char* resume_file = NULL);
  inline void Solve(const string resume_file) { Solve(resume_file.c_str()); }
  void Step(int iters);

#ifdef ADAPTIVE_BATCH
  virtual void AssignItersize(std::size_t itersize);
  virtual Dtype ForwardBackward(int iter_size);
#else
  virtual Dtype ForwardBackward();
#endif

  // The Restore method simply dispatches to one of the
  // RestoreSolverStateFrom___ protected methods. You should implement these
  // methods to restore the state from the appropriate snapshot type.
  void Restore(const char* resume_file);
  virtual ~Solver() {}
  inline const SolverParameter& param() const { return param_; }
  inline SolverParameter& param() { return param_; }
  inline shared_ptr<Net<Dtype> > net() { return net_; }
  inline const vector<shared_ptr<Net<Dtype> > >& test_nets() {
    return test_nets_;
  }
  int iter() { return iter_; }
  void set_iter(int value) { iter_ = value; }
  float scale_on_apply() { return scale_on_apply_; }
  void set_scale_on_apply(float value) { scale_on_apply_ = value; }

  // Invoked at specific points during an iteration
  class Callback {
   protected:
#ifdef ADAPTIVE_BATCH
    virtual void on_start(int i) {}
#else
    virtual void on_start() = 0;
#endif    
    virtual void on_gradients_ready() = 0;

    template <typename T>
    friend class Solver;
  };
  const vector<Callback*>& callbacks() const { return callbacks_; }
  void add_callback(Callback* value) {
    callbacks_.push_back(value);
  }

  typedef boost::function<Dtype()> ForwardBackwardFunc;
#ifdef ADAPTIVE_BATCH
  typedef boost::function<Dtype(int)> ForwardBackwardFuncArg;
  void set_forward_backward(ForwardBackwardFuncArg func) {
    forward_backward_ = func;
  }
#else
  // typedef boost::function<Dtype()> ForwardBackwardFunc;
  void set_forward_backward(ForwardBackwardFunc func) {
    forward_backward_ = func;
  }
#endif

  void CheckSnapshotWritePermissions();
  /**
   * @brief Returns the solver type.
   */
  virtual inline const char* type() const { return ""; }

  // The Solver::Snapshot function implements the basic snapshotting utility
  // that stores the learned net. You should implement the SnapshotSolverState()
  // function that produces a SolverState protocol buffer that needs to be
  // written to disk together with the learned net.
  void Snapshot();

  // Make and apply the update value for the current iteration.
  #ifdef ADAPTIVE_BATCH
  virtual void ApplyUpdate() {}//= 0;
  virtual void ApplyUpdate(int param_id) = 0;
// #ifdef ADAPTIVE_BATCH
  virtual void ApplyUpdate(bool batch_h_update) = 0;
#else
  virtual void ApplyUpdate() = 0;
  virtual void ApplyUpdate(int param_id) = 0;
#endif

  void TestAll();


#ifdef CAFFE_PER_LAYER_TIMINGS
  /* Timers for performance measurements */
  Timer timer;
  std::vector<double> forward_time_per_layer;
  std::vector<double> backward_time_per_layer;
  std::vector<double> update_time_per_layer;

  std::vector<double> forward_time_per_layer_total;
  std::vector<double> backward_time_per_layer_total;
  std::vector<double> update_time_per_layer_total;

  void InitTimers();
  void ResetTimers();
  void PrintTimers(bool printTotal);
#endif /* CAFFE_PER_LAYER_TIMINGS */

 protected:
  string SnapshotFilename(const string extension);
  string SnapshotToBinaryProto();
  string SnapshotToHDF5();
  // The test routine
  void Test(const int test_net_id = 0);
  virtual void SnapshotSolverState(const string& model_filename) = 0;
  virtual void RestoreSolverStateFromHDF5(const string& state_file) = 0;
  virtual void RestoreSolverStateFromBinaryProto(const string& state_file) = 0;
  void DisplayOutputBlobs(const int net_id);
  void UpdateSmoothedLoss(Dtype loss, int start_iter, int average_loss);

  SolverParameter param_;
  int iter_;
  int current_step_;
  shared_ptr<Net<Dtype> > net_;
  vector<shared_ptr<Net<Dtype> > > test_nets_;
  vector<Callback*> callbacks_;
  vector<Dtype> losses_;
  Dtype smoothed_loss_;
#ifdef ADAPTIVE_BATCH
  int newitersize_;
  std::deque<Dtype> deltaLosses_;
  std::deque<double> commTimes_;
  std::deque<double> commCompTimes_;
#endif 

  // The root solver that holds root nets (actually containing shared layers)
  // in data parallelism
  const Solver* const root_solver_;

  // A function that can be set by a client of the Solver to provide indication
  // that it wants a snapshot saved and/or to exit early.
  ActionCallback action_request_function_;

  // True iff a request to stop early was received.
  bool requested_early_exit_;

  // Scale gradients during apply
  float scale_on_apply_;

#ifdef ADAPTIVE_BATCH
  // Timing information
  // Timer iteration_timer_;
  // float iterations_last_;
  ForwardBackwardFuncArg forward_backward_;
#else
  ForwardBackwardFunc forward_backward_;
#endif 

  DISABLE_COPY_AND_ASSIGN(Solver);
};

/**
 * @brief Solver that only computes gradients, used as worker
 *        for multi-GPU training.
 */
template <typename Dtype>
class WorkerSolver : public Solver<Dtype> {
 public:
  explicit WorkerSolver(const SolverParameter& param,
      const Solver<Dtype>* root_solver = NULL)
      : Solver<Dtype>(param, root_solver) {}

 protected:
  void ApplyUpdate() { }
  void ApplyUpdate(int param_id) { }
  void SnapshotSolverState(const string& model_filename) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromBinaryProto(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
  void RestoreSolverStateFromHDF5(const string& state_file) {
    LOG(FATAL) << "Should not be called on worker solver.";
  }
};

}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
