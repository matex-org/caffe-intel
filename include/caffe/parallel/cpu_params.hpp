#ifndef CAFFE_PARALLEL_CPU_PARAMS_HPP_
#define CAFFE_PARALLEL_CPU_PARAMS_HPP_

#include "caffe/common.hpp"
#include "caffe/parallel.hpp"
#include "caffe/solver.hpp"

namespace caffe {

// Params stored in CPU memory.
template<typename Dtype>
class CPUParams : public Params<Dtype> {
 public:
  CPUParams(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~CPUParams();

  void configure(Solver<Dtype>* solver) const;

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

#endif

