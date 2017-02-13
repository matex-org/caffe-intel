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

#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

#ifdef USE_MLSL

#include "caffe/internode/mlsl_util.hpp"
using namespace MLSL;

#ifdef DISTR_WEIGHT_UPDATE
#define DISTRIBUTED_WEIGHT_UPDATE true
#else
#define DISTRIBUTED_WEIGHT_UPDATE false
#endif

#endif /* USE_MLSL */

#define MAX_ELEMS_TO_LOG 16
#define LOG_LAYER(layer) DLOG(INFO) << layer->type() << ": "
#define LOG_BLOB(layer, blob, part, blob_id, description)              \
  do                                                                   \
  {                                                                    \
      int elems_to_log = std::min(MAX_ELEMS_TO_LOG, blob->count());    \
      for (int idx = 0; idx < elems_to_log; idx++)                     \
      {                                                                \
          LOG_LAYER(layer) << description                              \
                           << ", blob_id " << blob_id                  \
                           << ", idx "     << idx                      \
                           << ", value "   << blob->cpu_##part()[idx]; \
      }                                                                \
  } while (0)

#define LOG_PARAM_BLOB(blob, part, blob_id, description)               \
  do                                                                   \
  {                                                                    \
      int elems_to_log = std::min(MAX_ELEMS_TO_LOG, blob->count());    \
      for (int idx = 0; idx < elems_to_log; idx++)                     \
      {                                                                \
          DLOG(INFO) << description                                    \
                     << ", blob_id " << blob_id                        \
                     << ", idx "     << idx                            \
                     << ", value "   << blob->cpu_##part()[idx];       \
      }                                                                \
  } while (0)

#define LOG_BUFFER(layer, buffer, buffer_id, description)    \
  do                                                         \
  {                                                          \
      if (!buffer) {                                         \
          /*LOG(WARNING) << "skip NULL buffer";*/                \
          break;                                             \
      }                                                      \
      for (int idx = 0; idx < MAX_ELEMS_TO_LOG; idx++)       \
      {                                                      \
          LOG_LAYER(layer) << description                    \
                           << ", buffer_id " << buffer_id    \
                           << ", idx "       << idx          \
                           << ", value "     << buffer[idx]; \
      }                                                      \
  } while (0)

#define CHECK_NUM_WEIGHTS(layer, params_ids)                    \
  do                                                            \
  {                                                             \
      DCHECK_EQ(param_ids.size(), layer->layerOp->NumWeights()) \
        << "check failed for layer " << layer->type()           \
        << ", param_ids.size() " << param_ids.size()            \
        << ", NumWeights " << layer->layerOp->NumWeights();     \
  } while (0)

/**
 Forward declare boost::thread instead of including boost/thread.hpp
 to avoid a boost/NVCC issues (#1009, #1010) on OSX.
 */
namespace boost { class mutex; }

namespace caffe {

/**
 * @brief An interface for the units of computation which can be composed into a
 *        Net.
 *
 * Layer%s must implement a Forward function, in which they take their input
 * (bottom) Blob%s (if any) and compute their output Blob%s (if any).
 * They may also implement a Backward function, in which they compute the error
 * gradients with respect to their input Blob%s, given the error gradients with
 * their output Blob%s.
 */
template <typename Dtype>
class Layer {

#ifdef USE_MLSL

public:

  /*************** MLSL ***************/

	MLSL::ComputeOp *layerOp;
	vector<MLSL::ComputeOp*> prevLayerOps;
	vector<Layer<Dtype>*> prevLayers;

	vector<int> ifm2ofm_map;
  std::vector<uint32_t> bottom_sizes;

  vector<Blob<Dtype>* > bottom_vec;
  vector<Blob<Dtype>* > top_vec;

  void on_delinp_ready(const vector<bool>& propagate_down) {

      LOG_LAYER(this) << "bprop: on_delinp_ready: enter";
      if (!this->layerOp->NumInputFeatureMaps()) {
          LOG_LAYER(this) << "bprop: on_delinp_ready: there is no any input FMs, exit";
          return;
      }

      int bottom_size = this->layer_param().bottom_size();
      LOG_LAYER(this) << "bprop: on_delinp_ready: bottom size " << bottom_size;

      for (int bottom_id = 0; bottom_id < bottom_size; ++bottom_id) {

          if (!propagate_down[bottom_id]/* || !fm->NumPackBlocks()*/) {
              LOG_LAYER(this) << "bprop: on_delinp_ready: skip CommsStart for bottom_id " << bottom_id;
              continue;
          }

          FeatureMap* fm = this->layerOp->InputFeatureMap(bottom_id);
          Dtype* comms_buf = (Dtype *)fm->CBuf()->GetPtr();

          if (comms_buf) {
              this->pack_buffer(fm, comms_buf, this->bottom_vec[bottom_id]->cpu_diff());
              LOG_BLOB(this, this->bottom_vec[bottom_id], diff, bottom_id, "bprop: on_delinp_ready: bottom_diff:");
              LOG_BUFFER(this, comms_buf, bottom_id, "bprop: on_delinp_ready: comms_buf:");
              LOG_LAYER(this) << "bprop: on_delinp_ready: send delinp for bottom # " << bottom_id;
              fm->CommsStart(comms_buf);
          }
      }
  }

  virtual void pack_buffer(MLSL::FeatureMap *fm, Dtype *to, const Dtype *from) {
    	for (int i = 0; i < fm->NumPackBlocks(); i++) {
      		BlockInfo * bi = fm->GetPackBlock(i);
      		int bMBLen = bi->MBLen();
      		int bMBStart = bi->MBStart();
      		int bFMLen = bi->FMLen();
      		int bFMStart = bi->FMStart();
      		Dtype *src = (Dtype*) from;
      		Dtype *dst = (Dtype*) (to + bi->BufOffset());
      		for (int mb = 0; mb < bMBLen; mb++) {
        			for (int fm = 0; fm < bFMLen; fm++) {
          				for (int s = 0 ; s < bi->FMSize(); s++) {
                          dst[(mb*bFMLen + fm)*bi->FMSize() + s] = src[((bMBStart+mb)*bFMLen + bFMStart+fm)*bi->FMSize() + s];
          				}
        			}
      		}
    	}
  }

  virtual void unpack_buffer(MLSL::FeatureMap *fm, const Dtype *from, Dtype *to) {
    for (int i = 0; i < fm->NumUnpackBlocks(); i++) {
        BlockInfo * bi = fm->GetUnpackBlock(i);
        int bMBLen = bi->MBLen();
        int bMBStart = bi->MBStart();
        int bFMLen = bi->FMLen();
        int bFMStart = bi->FMStart();
        Dtype *dst = (Dtype*) to;
        Dtype *src = (Dtype*) (from + bi->BufOffset());
        for (int mb = 0; mb < bMBLen; mb++) {
            for (int fm = 0; fm < bFMLen; fm++) {
                for (int s = 0 ; s < bi->FMSize(); s++) {
                  dst[((bMBStart+mb)*bFMLen + bFMStart+fm)*bi->FMSize() + s] = src[(mb*bFMLen + fm)*bi->FMSize() + s];
                }
            }
        }
    }
  }

  void SetPrevLayer(int index, Layer<Dtype> *prevLayer) {
      this->prevLayers[index] = prevLayer;
      this->prevLayerOps[index] = prevLayer->layerOp;
  }

  void ConfigureMLSL() {

      uint32_t in_size;
      this->layerOp->Finalize();
      this->layerOp->AllocCommsBufs();
      this->bottom_sizes.resize(this->prevLayerOps.size());

      for (int i = 0; i < this->prevLayerOps.size(); i++)
      {
          in_size = this->layerOp->InputFeatureMap(i)->LocalLen() * this->layerOp->LocalMinibatchLen() * this->layerOp->InputFeatureMap(i)->FMSize() * sizeof(Dtype);

          this->bottom_sizes[i] = in_size;

          LOG_LAYER(this) << "ConfigureMLSL: bottom_id " << i << ", in_size " << in_size
          << ", ifm ll " << this->layerOp->InputFeatureMap(i)->LocalLen() 
          << ", local mblen " << this->layerOp->LocalMinibatchLen()
          << ", ifm fmsize " << this->layerOp->InputFeatureMap(i)->FMSize()
          << ", sizeof Dtype " << sizeof(Dtype);
      }
  }

  /*************** MLSL ***************/
#endif /* USE_MLSL */

 public:
  /**
   * You should not implement your own constructor. Any set up code should go
   * to SetUp(), where the dimensions of the bottom blobs are provided to the
   * layer.
   */
  explicit Layer(const LayerParameter& param)
    : layer_param_(param), is_shared_(false) {
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase();
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());
          blobs_[i]->FromProto(layer_param_.blobs(i));
        }
      }

#ifdef USE_MLSL
      this->layerOp = 0;
#endif
    }

#ifdef USE_MLSL
  virtual ~Layer() {
      if (this->layerOp) {
          this->layerOp->FreeCommsBufs();
          //DLOG(INFO) << "~Layer: delete layerOp " << this->layerOp;
          delete this->layerOp;
          this->layerOp = 0;
      }
  }
#else
  virtual ~Layer() {}
#endif

  /**
   * @brief Implements common layer setup functionality.
   *
   * @param bottom the preshaped input blobs
   * @param top
   *     the allocated but unshaped output blobs, to be shaped by Reshape
   *
   * Checks that the number of bottom and top blobs is correct.
   * Calls LayerSetUp to do special layer setup for individual layer types,
   * followed by Reshape to set up sizes of top blobs and internal buffers.
   * Sets up the loss weight multiplier blobs for any non-zero loss weights.
   * This method may not be overridden.
   */
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    InitMutex();
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);

#ifdef USE_MLSL
    this->prevLayers.resize(bottom.size());
    this->prevLayerOps.resize(bottom.size());
#endif /* USE_MLSL */

  }

  /**
   * @brief Does layer-specific setup: your layer should implement this function
   *        as well as Reshape.
   *
   * @param bottom
   *     the preshaped input blobs, whose data fields store the input data for
   *     this layer
   * @param top
   *     the allocated but unshaped output blobs
   *
   * This method should do one-time layer specific setup. This includes reading
   * and processing relevent parameters from the <code>layer_param_</code>.
   * Setting up the shapes of top blobs and internal buffers should be done in
   * <code>Reshape</code>, which will be called before the forward pass to
   * adjust the top blob sizes.
   */
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

#ifdef USE_MLSL

  virtual MLSL::OpType getLayerTypeId(std::string const& layerType) {
	  if(layerType == "Convolution") return COMP_OP_TYPE_CC;
	  if(layerType == "InnerProduct") return COMP_OP_TYPE_CC;
	  if(layerType == "Data") return COMP_OP_TYPE_DATA;
	  if(layerType == "ReLU") return COMP_OP_TYPE_ACT;
	  if(layerType == "Dropout") return COMP_OP_TYPE_ACT;
	  if(layerType == "Pooling") return COMP_OP_TYPE_POOL;
	  if(layerType == "LRN") return COMP_OP_TYPE_POOL;
	  if(layerType == "Accuracy") return COMP_OP_TYPE_EVAL;
	  if(layerType == "SoftmaxWithLoss") return COMP_OP_TYPE_EVAL;
	  if(layerType == "Split") return COMP_OP_TYPE_BCAST;
	  if(layerType == "Concat") return COMP_OP_TYPE_CONCAT;
	  if(layerType == "Flatten") return COMP_OP_TYPE_ACT;
	  return COMP_OP_TYPE_CC;
  }

  virtual void SetUpMLSL(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	  DataType dt = (sizeof(Dtype) == 4)? DT_FLOAT : DT_DOUBLE;
	  ComputeOpRegInfo *myRegInfo;
	  myRegInfo = new ComputeOpRegInfo(getLayerTypeId(this->layer_param_.type()));
	  //myRegInfo = new ComputeOpRegInfo(COMP_OP_TYPE_BCAST);
	  myRegInfo->SetName(this->layer_param_.name().c_str());
	  for(int i=0; i<bottom.size(); i++)
	  {
		int ic = bottom[i]->channels();
		int iw = bottom[i]->width();
		int ih = bottom[i]->height();
		myRegInfo->AddInputFeatureMap(ic, iw*ih, dt);
	  }
	  for(int i=0; i<top.size(); i++)
	  {
		int oc = bottom[0]->channels();
		int ow = bottom[0]->width();
		int oh = bottom[0]->height();
		myRegInfo->AddOutputFeatureMap(oc, ow*oh, dt);
	  }

	  myRegInfo->Validate();
	  this->layerOp = new ComputeOp(myRegInfo, caffe::internode::data_parallelism);
	  delete myRegInfo;
  }
#endif

  /**
   * @brief Whether a layer should be shared by multiple nets during data
   *        parallelism. By default, all layers except for data layers should
   *        not be shared. data layers should be shared to ensure each worker
   *        solver access data sequentially during data parallelism.
   */
  virtual inline bool ShareInParallel() const { return false; }

  /** @brief Return whether this layer is actually shared by other nets.
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then this function is expected return true.
   */
  inline bool IsShared() const { return is_shared_; }

  /** @brief Set whether this layer is actually shared by other nets
   *         If ShareInParallel() is true and using more than one GPU and the
   *         net has TRAIN phase, then is_shared should be set true.
   */
  inline void SetShared(bool is_shared) {
    CHECK(ShareInParallel() || !is_shared)
        << type() << "Layer does not support sharing.";
    is_shared_ = is_shared;
  }

  /**
   * @brief Adjust the shapes of top blobs and internal buffers to accommodate
   *        the shapes of the bottom blobs.
   *
   * @param bottom the input blobs, with the requested input shapes
   * @param top the top blobs, which should be reshaped as needed
   *
   * This method should reshape top blobs as needed according to the shapes
   * of the bottom (input) blobs, as well as reshaping any internal buffers
   * and making any other necessary adjustments so that the layer can
   * accommodate the bottom blobs.
   */
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;

  /**
   * @brief Given the bottom blobs, compute the top blobs and the loss.
   *
   * @param bottom
   *     the input blobs, whose data fields store the input data for this layer
   * @param top
   *     the preshaped output blobs, whose data fields will store this layers'
   *     outputs
   * \return The total loss from the layer.
   *
   * The Forward wrapper calls the relevant device wrapper function
   * (Forward_cpu or Forward_gpu) to compute the top blob values given the
   * bottom blobs.  If the layer has any non-zero loss_weights, the wrapper
   * then computes and returns the loss.
   *
   * Your layer should implement Forward_cpu and (optionally) Forward_gpu.
   */
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Given the top blob error gradients, compute the bottom blob error
   *        gradients.
   *
   * @param top
   *     the output blobs, whose diff fields store the gradient of the error
   *     with respect to themselves
   * @param propagate_down
   *     a vector with equal length to bottom, with each index indicating
   *     whether to propagate the error gradients down to the bottom blob at
   *     the corresponding index
   * @param bottom
   *     the input blobs, whose diff fields will store the gradient of the error
   *     with respect to themselves after Backward is run
   *
   * The Backward wrapper calls the relevant device wrapper function
   * (Backward_cpu or Backward_gpu) to compute the bottom blob diffs given the
   * top blob diffs.
   *
   * Your layer should implement Backward_cpu and (optionally) Backward_gpu.
   */
  inline void Backward(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom);

  /**
   * @brief Returns the vector of learnable parameter blobs.
   */
  vector<shared_ptr<Blob<Dtype> > >& blobs() {
    return blobs_;
  }

  /**
   * @brief Returns the layer parameter.
   */
  const LayerParameter& layer_param() const { return layer_param_; }

  /**
   * @brief Writes the layer parameter to a protocol buffer
   */
  virtual void ToProto(LayerParameter* param, bool write_diff = false);

  /**
   * @brief Returns the scalar loss associated with a top blob at a given index.
   */
  inline Dtype loss(const int top_index) const {
    return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
  }

  /**
   * @brief Sets the loss associated with a top blob at a given index.
   */
  inline void set_loss(const int top_index, const Dtype value) {
    if (loss_.size() <= top_index) {
      loss_.resize(top_index + 1, Dtype(0));
    }
    loss_[top_index] = value;
  }

  /**
   * @brief Returns the layer type.
   */
  virtual inline const char* type() const { return ""; }

  /**
   * @brief Returns the exact number of bottom blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of bottom blobs.
   */
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of bottom blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of bottom blobs.
   */
  virtual inline int MinBottomBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of bottom blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of bottom blobs.
   */
  virtual inline int MaxBottomBlobs() const { return -1; }
  /**
   * @brief Returns the exact number of top blobs required by the layer,
   *        or -1 if no exact number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some exact number of top blobs.
   */
  virtual inline int ExactNumTopBlobs() const { return -1; }
  /**
   * @brief Returns the minimum number of top blobs required by the layer,
   *        or -1 if no minimum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some minimum number of top blobs.
   */
  virtual inline int MinTopBlobs() const { return -1; }
  /**
   * @brief Returns the maximum number of top blobs required by the layer,
   *        or -1 if no maximum number is required.
   *
   * This method should be overridden to return a non-negative value if your
   * layer expects some maximum number of top blobs.
   */
  virtual inline int MaxTopBlobs() const { return -1; }
  /**
   * @brief Returns true if the layer requires an equal number of bottom and
   *        top blobs.
   *
   * This method should be overridden to return true if your layer expects an
   * equal number of bottom and top blobs.
   */
  virtual inline bool EqualNumBottomTopBlobs() const { return false; }

  /**
   * @brief Return whether "anonymous" top blobs are created automatically
   *        by the layer.
   *
   * If this method returns true, Net::Init will create enough "anonymous" top
   * blobs to fulfill the requirement specified by ExactNumTopBlobs() or
   * MinTopBlobs().
   */
  virtual inline bool AutoTopBlobs() const { return false; }

  /**
   * @brief Return whether to allow force_backward for a given bottom blob
   *        index.
   *
   * If AllowForceBackward(i) == false, we will ignore the force_backward
   * setting and backpropagate to blob i only if it needs gradient information
   * (as is done when force_backward == false).
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  /**
   * @brief Specifies whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   *
   * You can safely ignore false values and always compute gradients
   * for all parameters, but possibly with wasteful computation.
   */
  inline bool param_propagate_down(const int param_id) {
    return (param_propagate_down_.size() > param_id) ?
        param_propagate_down_[param_id] : false;
  }
  /**
   * @brief Sets whether the layer should compute gradients w.r.t. a
   *        parameter at a particular index given by param_id.
   */
  inline void set_param_propagate_down(const int param_id, const bool value) {
    if (param_propagate_down_.size() <= param_id) {
      param_propagate_down_.resize(param_id + 1, true);
    }
    param_propagate_down_[param_id] = value;
  }


 protected:
  /** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;

  /** @brief Using the CPU device, compute the layer output. */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
  /**
   * @brief Using the GPU device, compute the layer output.
   *        Fall back to Forward_cpu() if unavailable.
   */
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // LOG(WARNING) << "Using CPU code as backup.";
    return Forward_cpu(bottom, top);
  }

  /**
   * @brief Using the CPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) = 0;
  /**
   * @brief Using the GPU device, compute the gradients for any parameters and
   *        for the bottom blobs if propagate_down is true.
   *        Fall back to Backward_cpu() if unavailable.
   */
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
    // LOG(WARNING) << "Using CPU code as backup.";
    Backward_cpu(top, propagate_down, bottom);
  }

  /**
   * Called by the parent Layer's SetUp to check that the number of bottom
   * and top Blobs provided as input match the expected numbers specified by
   * the {ExactNum,Min,Max}{Bottom,Top}Blobs() functions.
   */
  virtual void CheckBlobCounts(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top) {
    if (ExactNumBottomBlobs() >= 0) {
      CHECK_EQ(ExactNumBottomBlobs(), bottom.size())
          << type() << " Layer takes " << ExactNumBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MinBottomBlobs() >= 0) {
      CHECK_LE(MinBottomBlobs(), bottom.size())
          << type() << " Layer takes at least " << MinBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (MaxBottomBlobs() >= 0) {
      CHECK_GE(MaxBottomBlobs(), bottom.size())
          << type() << " Layer takes at most " << MaxBottomBlobs()
          << " bottom blob(s) as input.";
    }
    if (ExactNumTopBlobs() >= 0) {
      CHECK_EQ(ExactNumTopBlobs(), top.size())
          << type() << " Layer produces " << ExactNumTopBlobs()
          << " top blob(s) as output.";
    }
    if (MinTopBlobs() >= 0) {
      CHECK_LE(MinTopBlobs(), top.size())
          << type() << " Layer produces at least " << MinTopBlobs()
          << " top blob(s) as output.";
    }
    if (MaxTopBlobs() >= 0) {
      CHECK_GE(MaxTopBlobs(), top.size())
          << type() << " Layer produces at most " << MaxTopBlobs()
          << " top blob(s) as output.";
    }
    if (EqualNumBottomTopBlobs()) {
      CHECK_EQ(bottom.size(), top.size())
          << type() << " Layer produces one top blob as output for each "
          << "bottom blob input.";
    }
  }

  /**
   * Called by SetUp to initialize the weights associated with any top blobs in
   * the loss function. Store non-zero loss weights in the diff blob.
   */
  inline void SetLossWeights(const vector<Blob<Dtype>*>& top) {
    const int num_loss_weights = layer_param_.loss_weight_size();
    if (num_loss_weights) {
      CHECK_EQ(top.size(), num_loss_weights) << "loss_weight must be "
          "unspecified or specified once per top blob.";
      for (int top_id = 0; top_id < top.size(); ++top_id) {
        const Dtype loss_weight = layer_param_.loss_weight(top_id);
        if (loss_weight == Dtype(0)) { continue; }
        this->set_loss(top_id, loss_weight);
        const int count = top[top_id]->count();
        Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
        caffe_set(count, loss_weight, loss_multiplier);
      }
    }
  }

 private:
  /** Whether this layer is actually shared by other nets*/
  bool is_shared_;

  /** The mutex for sequential forward if this layer is shared */
  shared_ptr<boost::mutex> forward_mutex_;

  /** Initialize forward_mutex_ */
  void InitMutex();
  /** Lock forward_mutex_ if this layer is shared */
  void Lock();
  /** Unlock forward_mutex_ if this layer is shared */
  void Unlock();

  DISABLE_COPY_AND_ASSIGN(Layer);
};  // class Layer

// Forward and backward wrappers. You should implement the cpu and
// gpu specific implementations instead, and should not change these
// functions.
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // Lock during forward to ensure sequential forward
  Lock();
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
  case Caffe::GPU:
    Forward_gpu(bottom, top);
#ifndef CPU_ONLY
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->gpu_data();
      const Dtype* loss_weights = top[top_id]->gpu_diff();
      Dtype blob_loss = 0;
      caffe_gpu_dot(count, data, loss_weights, &blob_loss);
      loss += blob_loss;
    }
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
  Unlock();
  return loss;
}

template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Backward_cpu(top, propagate_down, bottom);
    break;
  case Caffe::GPU:
    Backward_gpu(top, propagate_down, bottom);
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode.";
  }
}

// Serialize LayerParameter to protocol buffer
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) {
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) {
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  }
}

}  // namespace caffe

#endif  // CAFFE_LAYER_H_
