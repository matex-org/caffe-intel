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

#ifdef USE_MLSL

#include "caffe/multinode/MlslSync.hpp"

namespace caffe {

template<typename Dtype>
MlslSync<Dtype>::MlslSync(shared_ptr<Solver<Dtype> > root_solver)
        : solver(boost::make_shared<MlslSolver<Dtype> >(root_solver))
        , initialized(false)
        , solver_thread_id(boost::this_thread::get_id())
        , snapshot_per_iters(root_solver->param().snapshot())
        , layers(root_solver->net()->layers())
        , net(root_solver->net())
        , net_params(root_solver->net()->learnable_params())
        , is_root(MLSL::GetNodeId() == 0)
    {
        root_solver->param().set_disabled_update(true);
        if (!is_root) root_solver->param().clear_snapshot();
        if (!is_root) root_solver->param().set_snapshot_after_train(false);
        //if (!is_root) root_solver->param().clear_test_interval();

        if (root_solver->iter() == 0)
            root_solver->set_iter(1);

        for (int idx = 0; idx < layers.size(); idx++)
        {
            layers[idx]->bottom_vec = root_solver->net()->bottom_vecs()[idx];
            layers[idx]->top_vec = root_solver->net()->top_vecs()[idx];
        }

        layer_param_ids.resize(layers.size());

        bottom_pack_block_nums.resize(layers.size());
        bottom_unpack_block_nums.resize(layers.size());
        top_pack_block_nums.resize(layers.size());
        top_unpack_block_nums.resize(layers.size());

        for (int layer_id = 0; layer_id < layers.size(); layer_id++) {
            shared_ptr<Layer<Dtype> > layer = layers[layer_id];

            /* cache param ids */
            layer_param_ids[layer_id] = net->get_layer_learnable_param_ids(layer_id);

            /* cache bottom/top pack/unpack blocks nums */
            int bottom_size = layer->layer_param().bottom_size();
            int top_size = layer->layer_param().top_size();

            bottom_pack_block_nums[layer_id].resize(bottom_size, 0);
            bottom_unpack_block_nums[layer_id].resize(bottom_size, 0);

            top_pack_block_nums[layer_id].resize(top_size, 0);
            top_unpack_block_nums[layer_id].resize(top_size, 0);

            if (layer->layerOp->NumInputFeatureMaps())
            {
                for (int bottom_id = 0; bottom_id < bottom_size; ++bottom_id) {
                    FeatureMap *fm = layer->layerOp->InputFeatureMap(bottom_id);
                    bottom_pack_block_nums[layer_id][bottom_id] = fm->NumPackBlocks();
                    bottom_unpack_block_nums[layer_id][bottom_id] = fm->NumUnpackBlocks();
                }
            }

            if (layer->layerOp->NumOutputFeatureMaps())
            {
                for (int top_id = 0; top_id < top_size; ++top_id) {
                    FeatureMap *fm = layer->layerOp->OutputFeatureMap(top_id);
                    top_pack_block_nums[layer_id][top_id] = fm->NumPackBlocks();
                    top_unpack_block_nums[layer_id][top_id] = fm->NumUnpackBlocks();
                }
            }

            /* set owned_count and owned_offset for distributed weight update */
#ifdef DISTR_WEIGHT_UPDATE
            if (layer->layerOp->HasWeights()) {

                vector<int>& param_ids = layer_param_ids[layer_id];
                CHECK_NUM_WEIGHTS(layer, param_ids);

                for (int i = 0; i < param_ids.size(); ++i) {
                    int owned_count = layer->layerOp->Weights(i)->OwnedLen() * layer->layerOp->Weights(i)->WTSize();
                    int owned_offset = layer->layerOp->Weights(i)->OwnedStart() * layer->layerOp->Weights(i)->WTSize();

                    if (layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() != net_params[param_ids[i]]->count()) {
                        LOG(FATAL) << "different local weigth count in CAFFE (" << net_params[param_ids[i]]->count() << ") "
                                   << "and MLSL (" << layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() << ")";
                    }
                    /*if (layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() > net_params[param_ids[i]]->count()) {
                        owned_count -= (layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize() - net_params[param_ids[i]]->count());
                    }*/

                    net_params[param_ids[i]]->set_owned_count(owned_count);
                    net_params[param_ids[i]]->set_owned_offset(owned_offset);

                    LOG(INFO) << "layer " << layer->type()
                                 << ", weigth_idx " << i
                                 << ", MLSL owned_cout " << layer->layerOp->Weights(i)->OwnedLen() * layer->layerOp->Weights(i)->WTSize()
                                 << ", CAFFE owned_count " << owned_count
                                 << ", CAFFE owned_offset " << owned_offset
                                 << ", MLSL local_count " << layer->layerOp->Weights(i)->LocalLen() * layer->layerOp->Weights(i)->WTSize()
                                 << ", CAFFE local_count " << net_params[param_ids[i]]->count();
                }
            }
#endif /* DISTR_WEIGHT_UPDATE */
        }
    }

template<typename Dtype>
MlslSync<Dtype>::~MlslSync()
{
}

  INSTANTIATE_CLASS(MlslSync);
} // namespace caffe

#endif /* USE_MLSL */
