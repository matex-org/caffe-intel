#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/parallel/cpu_params.hpp"
#include "caffe/solver.hpp"

namespace caffe {

template<typename Dtype>
CPUParams<Dtype>::CPUParams(shared_ptr<Solver<Dtype> > root_solver)
    : Params<Dtype>(root_solver) {
  data_ = new Dtype[size_];

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);

  diff_ = new Dtype[size_];
  caffe_set(size_, Dtype(0), diff_);
}

template<typename Dtype>
CPUParams<Dtype>::~CPUParams() {
  delete [] data_;
  delete [] diff_;
}

template<typename Dtype>
void CPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
  apply_buffers(net, data_, size_, replace_cpu);
  apply_buffers(net, diff_, size_, replace_cpu_diff);
}

INSTANTIATE_CLASS(CPUParams);

}  // namespace caffe
