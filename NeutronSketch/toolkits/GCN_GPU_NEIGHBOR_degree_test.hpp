#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include "core/graph.hpp"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <algorithm>
class GCN_GPU_NEIGHBOR_Degree_impl {
 public:
  int iterations;
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType drop_rate;
  ValueType alpha;
  ValueType beta1;
  ValueType beta2;
  ValueType epsilon;
  ValueType decay_rate;
  ValueType decay_epoch;
  ValueType best_val_acc;
  double start_time;
  int layers;
  double used_gpu_mem, total_gpu_mem;
  // graph
  VertexSubset* active;
  // graph with no edge data
  Graph<Empty>* graph;
  // std::vector<CSC_segment_pinned *> subgraphs;
  // NN
  GNNDatum* gnndatum;
  NtsVar L_GT_C;
  NtsVar L_GT_G;
  NtsVar MASK;
  NtsVar MASK_gpu;
  // GraphOperation *gt;
  PartitionedGraph* partitioned_graph;
  // Variables
  std::vector<Parameter*> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  FullyRepGraph* fully_rep_graph;
  double train_compute_time = 0;
  double mpi_comm_time = 0;
  double rpc_comm_time = 0;
  double rpc_wait_time = 0;
  float loss_epoch = 0;
  float f1_epoch = 0;
  Sampler* train_sampler = nullptr;
  Sampler* eval_sampler = nullptr;
  Sampler* test_sampler = nullptr;
  // double gcn_start_time = 0;
  double gcn_run_time;


  double gcn_gather_time;
  double gcn_sample_time;
  double gcn_trans_time;
  double gcn_train_time;
  double gcn_cache_hit_rate;
  double gcn_trans_memory;

  double epoch_sample_time = 0;
  double epoch_gather_label_time = 0;
  double epoch_gather_feat_time = 0;
  double epoch_transfer_graph_time = 0;
  double epoch_transfer_feat_time = 0;
  double epoch_transfer_label_time = 0;
  double epoch_train_time = 0;
  int epoch_cache_hit = 0;
  int epoch_all_node = 0;
  double debug_time = 0;
  vector<float> explicit_rate;

  int threads;
  float* dev_cache_feature;

  VertexId *local_idx, *local_idx_cache, *dev_local_idx, *dev_local_idx_cache;
  Cuda_Stream* cuda_stream;

  std::mutex sample_mutex;
  std::mutex transfer_mutex;
  std::mutex train_mutex;
  int pipelines;
  Cuda_Stream* cuda_stream_list;
  std::vector<at::cuda::CUDAStream> torch_stream;

  // int batch_size_switch_idx = 0;

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  long train_nodes;
  int max_batch_num;
  int min_batch_num;
  std::string dataset_name;
  torch::nn::Dropout drpmodel;
  // double sample_cost = 0;
  std::vector<torch::nn::BatchNorm1d> bn1d;

  ntsPeerRPC<ValueType, VertexId>* rpc;
  int hosts;
  // std::unordered_map<std::string, std::vector<int>> batch_size_mp;
  // std::vector<int> batch_size_vec;

  std::vector<int> cache_node_idx_seq;
  // std::unordered_set<int> cache_node_hashmap;
  // std::vector<int> cache_node_hashmap;
  VertexId* cache_node_hashmap;
  VertexId* dev_cache_node_hashmap;
  int cache_node_num = 0;

  // std::vector<int>is_sketch;
  int *is_sketch;
  // float *mean_feat;
  std::map<int,std::vector<int>>class_nodes;
  std::vector<std::vector<std::pair<int,float>>>nid_nbr;
  // std::map<int,float>score;

  GCN_GPU_NEIGHBOR_Degree_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
                        bool process_overlap = false) {
    graph = graph_;
    iterations = iterations_;

    active = graph->alloc_vertex_subset();
    active->fill();

    graph->init_gnnctx(graph->config->layer_string);
    // graph->init_gnnctx_fanout(graph->config->fanout_string);
    graph->init_gnnctx_fanout(graph->gnnctx->fanout, graph->config->fanout_string);
    graph->init_gnnctx_fanout(graph->gnnctx->val_fanout, graph->config->val_fanout_string);
    assert(graph->gnnctx->fanout.size() == graph->gnnctx->val_fanout.size());
    reverse(graph->gnnctx->fanout.begin(), graph->gnnctx->fanout.end());
    reverse(graph->gnnctx->val_fanout.begin(), graph->gnnctx->val_fanout.end());
    graph->init_rtminfo();
    graph->rtminfo->process_local = graph->config->process_local;
    graph->rtminfo->reduce_comm = graph->config->process_local;
    graph->rtminfo->copy_data = false;
    graph->rtminfo->process_overlap = graph->config->overlap;
    graph->rtminfo->with_weight = true;
    graph->rtminfo->with_cuda = false;
    graph->rtminfo->lock_free = graph->config->lock_free;
    hosts = graph->partitions;
    if (hosts > 1) {
      rpc = new ntsPeerRPC<ValueType, VertexId>();
    } else {
      rpc = nullptr;
    }
    best_val_acc = 0;
    cuda_stream = new Cuda_Stream();

    pipelines = graph->config->pipelines;
    pipelines = std::max(1, pipelines);
    torch_stream.clear();

    gcn_run_time = 0;
    gcn_gather_time = 0;
    gcn_sample_time = 0;
    gcn_trans_time = 0;
    gcn_train_time = 0;
    gcn_cache_hit_rate = 0;
    gcn_trans_memory = 0;
    threads = std::max(1, numa_num_configured_cpus() / 2 - 1);

    // batch_size_mp["ppi"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 2048, 4096, 9716};
    // batch_size_mp["ppi-large"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44906};
    // batch_size_mp["flickr"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 44625};
    // batch_size_mp["AmazonCoBuy_computers"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8250};
    // batch_size_mp["ogbn-arxiv"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    // 65536, 90941}; batch_size_mp["AmazonCoBuy_photo"] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4590};

    // batch_size_switch_idx = 0;
    // batch_size_vec = graph->config->batch_size_vec;

    is_sketch = graph->alloc_vertex_array<int>();

  }

  void init_active() {
    active = graph->alloc_vertex_subset();
    active->fill();
  }

  void init_graph() {
    fully_rep_graph = new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");

    // cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx = new nts::ctx::NtsContext();
  }

  void get_batch_num() {
    VertexId max_vertex = 0;
    VertexId min_vertex = std::numeric_limits<VertexId>::max();
    for (int i = 0; i < graph->partitions; i++) {
      max_vertex = std::max(graph->partition_offset[i + 1] - graph->partition_offset[i], max_vertex);
      min_vertex = std::min(graph->partition_offset[i + 1] - graph->partition_offset[i], min_vertex);
    }
    max_batch_num = max_vertex / graph->config->batch_size;
    min_batch_num = min_vertex / graph->config->batch_size;
    if (max_vertex % graph->config->batch_size != 0) {
      max_batch_num++;
    }
    if (min_vertex % graph->config->batch_size != 0) {
      min_batch_num++;
    }
  }

  void init_nn() {
    // const uint64_t seed = 2000;
    // torch::manual_seed(seed);
    // torch::cuda::manual_seed_all(seed);

    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    layers = graph->gnnctx->layer_size.size() - 1;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    gnndatum = new GNNDatum(graph->gnnctx, graph);
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
      // printf("test before sim");

      // for(int i=0;i<graph->vertices;i++) {
      //   std::vector<std::pair<int,float>>tmp_vec;
      //   for(int j=fully_rep_graph->column_offset[i] ,index=0; j<fully_rep_graph->column_offset[i+1]; j++,index++) {
      //     tmp_vec.push_back(std::make_pair(index,fully_rep_graph->sim_value[j]));
      //   }
      //   sort(tmp_vec.begin(), tmp_vec.end(), [&](auto& x, auto& y) { return x.second > y.second; });
      //   nid_nbr.push_back(tmp_vec);
      // }
      printf("init nn end4!");
    }
    // else if (graph->config->degree_switch==-1){
    //   gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
    //                                    graph->config->mask_file);
    // } else{
    //   gnndatum->readFeature_Label_Mask_Degree(graph->config->feature_file, graph->config->label_file,
    //                                    graph->config->mask_file,graph->config->target_degree);
    // }
    // creating tensor to save Label and Mask
    if (graph->config->classes > 1) {
      gnndatum->registLabel(L_GT_C, gnndatum->local_label, gnndatum->gnnctx->l_v_num, graph->config->classes);
    } else {
      gnndatum->registLabel(L_GT_C);
    }
    gnndatum->registMask(MASK);
    MASK_gpu = MASK.cuda();
    gnndatum->generate_gpu_data();

    torch::Device GPU(torch::kCUDA, 0);

    for (int i = 0; i < layers; i++) {
      P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
                                epsilon, weight_decay));
      // P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], learn_rate, weight_decay));
      if (graph->config->batch_norm && i < layers - 1) {
        bn1d.push_back(torch::nn::BatchNorm1d(graph->gnnctx->layer_size[i]));
        // bn1d.back().to(GPU);
        // bn1d.back().cuda();
      }
    }
    printf("init nn end3!");
    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      P[i]->to(GPU);
      P[i]->Adam_to_GPU();
    }
    printf("init nn end2!");
    // drpmodel = torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);
    // std::cout<<F<<endl;
    // X[i] is vertex representation at layer i
    for (int i = 0; i < layers + 1; i++) {
      NtsVar d;
      X.push_back(d);
    }

    X[0] = F.set_requires_grad(true);
    printf("init nn end1!");
    if (hosts > 1) {
      rpc->set_comm_num(graph->partitions - 1);
      rpc->register_function("get_feature", [&](std::vector<VertexId> vertexs) {
        int start = graph->partition_offset[graph->partition_id];
        int feature_size = F.size(1);
        ValueType* ntsVarBuffer = graph->Nts->getWritableBuffer(F, torch::DeviceType::CPU);
        std::vector<std::vector<ValueType>> result_vector;
        result_vector.resize(vertexs.size());

// omp_set_num_threads(threads);
#pragma omp parallel for
        for (int i = 0; i < vertexs.size(); i++) {
          result_vector[i].resize(feature_size);
          memcpy(result_vector[i].data(), ntsVarBuffer + (vertexs[i] - start) * feature_size,
                 feature_size * sizeof(ValueType));
        }
        return result_vector;
      });
    }
    printf("init nn end!");
  }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      if (ctx->is_train() && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      if (graph->gnnctx->l_v_num == 0) {
        P[i]->all_reduce_to_gradient(torch::zeros_like(P[i]->W));
      } else {
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      }
      if (ctx->is_train() && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
      // update parameters with Adam optimizer
      // printf("adam!!");
      // P[i]->learnC2G(learn_rate);
      // P[i]->learnC2C_with_Adam();
      P[i]->learnC2G_with_decay_Adam();
      P[i]->next();
      // P[i]->learnC2G_with_decay_SGD();
      // P[i]->learnC2G_with_SGD();
    }
  }

  void UpdateZero() {
    for (int l = 0; l < layers; l++) {
      P[l]->all_reduce_to_gradient(torch::zeros({P[l]->row, P[l]->col}, torch::kFloat));
    }
  }

  void old_version(Sampler* sampler) {

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }

    sampler->metis_batch_id = 0;
    int batch_id = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      if (graph->config->run_time > 0 && gcn_run_time >= graph->config->run_time) {
        break;
      }
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute

      epoch_sample_time -= get_time();
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());
      epoch_sample_time += get_time();

      epoch_transfer_graph_time -= get_time();
      ssg->trans_graph_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      // ssg->trans_graph_to_gpu_async(cuda_stream->stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      epoch_transfer_graph_time += get_time();

      epoch_transfer_feat_time -= get_time();
      // sampler->load_feature_gpu(cuda_stream, ssg, X[0], gnndatum->dev_local_feature);
      sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      epoch_transfer_feat_time += get_time();

      epoch_transfer_label_time -= get_time();
      // sampler->load_label_gpu(cuda_stream, ssg, target_lab, gnndatum->dev_local_label);
      sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      epoch_transfer_label_time += get_time();

      epoch_train_time -= get_time();
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], cuda_stream);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->is_train()) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }
      epoch_train_time += get_time();

      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }

      sampler->reverse_sgs();
      batch_id++;
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
}

  void pipeline_version(Sampler* sampler) {
    NtsVar tmp_X0[pipelines];
    NtsVar tmp_target_lab[pipelines];
    LOG_DEBUG("pipeline %d", pipelines);
    for (int i = 0; i < pipelines; i++) {
      tmp_X0[i] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
      if (graph->config->classes > 1) {
        tmp_target_lab[i] =
            graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
      } else {
        tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
      }
    }

    std::thread threads[pipelines];
    for (int tid = 0; tid < pipelines; ++tid) {
      threads[tid] = std::thread(
          [&](int thread_id) {
            ////////////////////////////////// sample //////////////////////////////////
            std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
            std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
            std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
            sample_lock.lock();
            while (sampler->work_offset < sampler->work_range[1]) {
              auto ssg = sampler->subgraph_list[thread_id];
              epoch_sample_time -= get_time();
              sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
              epoch_sample_time += get_time();
              cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
              sample_lock.unlock();

              ////////////////////////////////// transfer //////////////////////////////////
              transfer_lock.lock();
              epoch_transfer_graph_time -= get_time();
              ssg->trans_graph_to_gpu_async(cuda_stream_list[thread_id].stream, graph->config->mini_pull > 0);
              epoch_transfer_graph_time += get_time();
              if (graph->config->cache_type == "none") {  // trans feature use zero copy (omit gather feature)
                epoch_transfer_feat_time -= get_time();
                sampler->load_feature_gpu(&cuda_stream_list[thread_id], ssg, tmp_X0[thread_id],
                                          gnndatum->dev_local_feature);
                epoch_transfer_feat_time += get_time();
                // get_gpu_mem(used_gpu_mem, total_gpu_mem);
              } else if (graph->config->cache_type == "gpu_memory" ||
                         graph->config->cache_type == "rate") {  // trans freature which is not cache in gpu
                // epoch_transfer_feat_time -= get_time();
                auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(
                    &cuda_stream_list[thread_id], ssg, tmp_X0[thread_id], gnndatum->dev_local_feature,
                    dev_cache_feature, local_idx, local_idx_cache, cache_node_hashmap, dev_local_idx,
                    dev_local_idx_cache, dev_cache_node_hashmap);
                // epoch_transfer_feat_time += get_time();
                epoch_transfer_feat_time += trans_feature_tmp;
                epoch_gather_feat_time += gather_gpu_cache_tmp;

                debug_time -= get_time();
                epoch_all_node += ssg->sampled_sgs[0]->src().size();
                for (auto& it : ssg->sampled_sgs[0]->src()) {
                  if (cache_node_hashmap[it] != -1) {
                    epoch_cache_hit++;
                  }
                }
                debug_time += get_time();
              } else {
                std::cout << "cache_type: " << graph->config->cache_type << " is not support!" << std::endl;
                assert(false);
              }
              epoch_transfer_label_time -= get_time();
              sampler->load_label_gpu(&cuda_stream_list[thread_id], ssg, tmp_target_lab[thread_id],
                                      gnndatum->dev_local_label);

              epoch_transfer_label_time += get_time();
              cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
              transfer_lock.unlock();

              ////////////////////////////////// train //////////////////////////////////
              train_lock.lock();
              at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
              if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
              epoch_train_time -= get_time();
              for (int l = 0; l < layers; l++) {  // forward
                graph->rtminfo->curr_layer = l;
                if (l == 0) {
                  NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, tmp_X0[thread_id],
                                                                                &cuda_stream_list[thread_id]);
                  X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                } else {
                  NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l],
                                                                                &cuda_stream_list[thread_id]);
                  X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                }
              }

              auto loss_ = Loss(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
              loss_epoch += loss_.item<float>();

              if (ctx->training == true) {
                ctx->appendNNOp(X[layers], loss_);
                ctx->self_backward(false);
                Update();
              }

              if (graph->config->classes == 1) {
                correct += get_correct(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                train_nodes += tmp_target_lab[thread_id].size(0);
              } else {
                f1_epoch += f1_score(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
              }
              epoch_train_time += get_time();

              cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
              train_lock.unlock();

              sample_lock.lock();
              sampler->reverse_sgs(ssg);
            }
            sample_lock.unlock();
            /////// disable thread
            return;
          },
          tid);
    }
    for (int tid = 0; tid < pipelines; ++tid) {
      threads[tid].join();
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }

  void explicit_version(Sampler* sampler) {
    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }
    sampler->metis_batch_id = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
      auto ssg = sampler->subgraph;
      epoch_sample_time -= get_time();
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      epoch_sample_time += get_time();

      epoch_transfer_graph_time -= get_time();
      ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      // ssg->trans_graph_to_gpu_async(cuda_stream->stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      epoch_transfer_graph_time += get_time();

      ///////////start gather and trans feature (explicit version) /////////////
      epoch_gather_feat_time -= get_time();
      // if (hosts > 1) {
      //   X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src().data(),
      //   ssg->sampled_sgs[0]->src_size,
      //                                           F, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= graph->config->time_skip) rpc_comm_time += tmp_time;
      // } else {
      X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
      // X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src(), F, graph);
      // }
      epoch_gather_feat_time += get_time();

      epoch_transfer_feat_time -= get_time();
      X[0] = X[0].cuda().set_requires_grad(true);
      epoch_transfer_feat_time += get_time();

      ////////start trans target_lab (explicit)//////////
      epoch_gather_label_time -= get_time();
      // if (hosts > 1) {
      //   target_lab = nts::op::get_label_from_global(ssg->sampled_sgs.back()->dst().data(),
      //                                               ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
      //   // if (type == 0 && graph->rtminfo->epoch >= graph->config->time_skip) rpc_comm_time += tmp_time;
      // } else {
      target_lab =
          nts::op::get_label(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);

      // target_lab = nts::op::get_label(ssg->sampled_sgs.back()->dst(), L_GT_C, graph);
      // }
      epoch_gather_label_time += get_time();

      // double trans_label_gpu_cost = -get_time();
      epoch_transfer_label_time -= get_time();
      target_lab = target_lab.cuda();
      epoch_transfer_label_time += get_time();
      ///////end trans target_lab (explicit)///////

      epoch_train_time -= get_time();
      at::cuda::setCurrentCUDAStream(torch_stream[0]);
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        // NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l]);
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->training == true) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }
      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      epoch_train_time += get_time();

      sampler->reverse_sgs();
    }
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }

  void zerocopy_version(Sampler* sampler) {
    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }
    sampler->metis_batch_id = 0;

    std::vector<VertexId>epoch_nodes;
    VertexId node_size = 0;
    VertexId edge_size = 0;
    VertexId tmp = 0;
    int i =0;
    std::map<int,std::vector<float>>class_similarity;
    
    int class_size = graph->gnnctx->label_num;
    int *result_1 = new int[graph->gnnctx->label_num*graph->gnnctx->label_num];
    int *result_2 = new int[graph->gnnctx->label_num*graph->gnnctx->label_num];
    memset(result_1,0,sizeof(int)*(graph->gnnctx->label_num*graph->gnnctx->label_num));
    memset(result_2,0,sizeof(int)*(graph->gnnctx->label_num*graph->gnnctx->label_num));

    std::vector<int>nid_vec;
    NtsVar aggre_feat = torch::empty({0,128},torch::DeviceType::CUDA);

    // LOG_INFO("work_range%d",sampler->work_range[1]);
    while (sampler->work_offset < sampler->work_range[1]) {
      if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
      auto ssg = sampler->subgraph;
      // class_similarity.clear();
      epoch_sample_time -= get_time();
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_by_label(ssg, graph->config->batch_type, ctx->is_train(),is_sketch,gnndatum);
      // sampler->sample_one_by_label(ssg, graph->config->batch_type, ctx->is_train(),nid_nbr);

      // node_size = node_size + ssg->sampled_sgs[1]->v_size + ssg->sampled_sgs[1]->src_size + ssg->sampled_sgs[0]->src_size;
      // edge_size = edge_size + ssg->sampled_sgs[0]->e_size + ssg->sampled_sgs[1]->e_size;
      // auto d_0 = ssg->sampled_sgs[0]->dst();
      // auto s_0 = ssg->sampled_sgs[0]->src();
      // auto d_1 = ssg->sampled_sgs[1]->dst();
      // auto s_1 = ssg->sampled_sgs[1]->src();
      // printf("batch: %d col_off_1 %d row_ind_1 %d  col_off_0 %d row_ind_0 %d d1 %d s1 %d d0 %d s0 %d\n",i,ssg->sampled_sgs[1]->column_offset.size(),ssg->sampled_sgs[1]->row_indices.size(),ssg->sampled_sgs[0]->column_offset.size(),ssg->sampled_sgs[0]->row_indices.size(),d_1.size(),s_1.size(),d_0.size(),s_0.size());
      // for(int index = 0;index<d_1.size();index++) {
      //   int nid = d_1[index];
      //   float one_nbr_sim = 0.0;
      //   float two_nbr_sim = 0.0;
      //   int cur_class = gnndatum->local_label[nid];
      //   int one_same_num = 0;
      //   int two_same_num = 0;
      //   int one_nbr_num = ssg->sampled_sgs[1]->column_offset[index+1] - ssg->sampled_sgs[1]->column_offset[index];
      //   int two_nbr_num = 0;
      //   for(int j=ssg->sampled_sgs[1]->column_offset[index];j<ssg->sampled_sgs[1]->column_offset[index+1];j++) {
      //     int one_nbr_indice = ssg->sampled_sgs[1]->row_indices[j];
      //     int one_nbr = s_1[one_nbr_indice];
      //     result_1[class_size*cur_class+gnndatum->local_label[one_nbr]]+=1;
      //     std::vector<VertexId>::iterator it = find(d_0.begin(),d_0.end(),one_nbr);
      //     int two_index = &*it-&d_0[0];
      //     // printf("one nbr index: %d,nid: %d\n",two_index,one_nbr);
      //     two_nbr_num += ssg->sampled_sgs[0]->column_offset[two_index+1] - ssg->sampled_sgs[0]->column_offset[two_index];
      //     if(gnndatum->local_label[one_nbr] == cur_class) {
      //       one_same_num ++;
      //     }
      //     for(int n=ssg->sampled_sgs[0]->column_offset[two_index];n<ssg->sampled_sgs[0]->column_offset[two_index+1];n++) {
      //       result_2[class_size*cur_class+gnndatum->local_label[ssg->sampled_sgs[0]->row_indices[n]]]+=1;
      //       if(gnndatum->local_label[ssg->sampled_sgs[0]->row_indices[n]] == cur_class) {
      //       two_same_num ++;
      //       }
      //     }
      //   }
      //   one_nbr_sim = float(one_same_num)/one_nbr_num;
      //   two_nbr_sim = float(two_same_num)/two_nbr_num;
      //   if(class_similarity.count(cur_class)==0){
      //     std::vector<float>tmp_vec;
      //     tmp_vec.push_back(two_nbr_sim);
      //     class_similarity.insert(std::make_pair(cur_class,tmp_vec));
      //   }else{
      //     class_similarity[cur_class].push_back(two_nbr_sim);
      //   }
      //   // LOG_INFO("batch %d nid %d class %d 1-hop-sim: %f 2-hop-sim: %f",i,nid,cur_class,one_nbr_sim,two_nbr_sim);
      // }

      // std::vector<int>result;
      // result.clear();
      // for(const auto&v:d_1){
      //   result.push_back(v);
      //   epoch_nodes.push_back(v);
      // }
      // for(const auto&v:s_1){
      //   result.push_back(v);
      //   epoch_nodes.push_back(v);
      // }
      // for(const auto&v:s_0){
      //   result.push_back(v);
      //   epoch_nodes.push_back(v);
      // }
      // printf("batch: %d nodes num: %d\n",i,result.size());
      // std::set<int>s(result.begin(),result.end());
      // result.assign(s.begin(),s.end());
      // printf("after batch: %d nodes num: %d\n",i,result.size());
      // // for(auto it:class_similarity) {
      // //   double sum = std::accumulate(std::begin(it.second), std::end(it.second), 0.0);  
      // //   double mean =  sum / it.second.size(); //均值
      // //   double accum  = 0.0; 
      // //   std::for_each (std::begin(it.second), std::end(it.second), [&](const double d) {  
      // //     accum  += (d-mean)*(d-mean);  
      // //   }); 
      // //   double var = sqrt(accum/(it.second.size()-1)); //方差 
      // // LOG_INFO("class: %d two-sim-mean: %lf two-sim-var: %lf",it.first,mean,var);
      // // }
      // i = i+1;

      // LOG_INFO("sample one!!!");
      epoch_sample_time += get_time();
      
      // std::cout << "sample done" << std::endl;

      epoch_transfer_graph_time -= get_time();
      ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      // ssg->trans_graph_to_gpu_async(cuda_stream->stream, graph->config->mini_pull > 0);  // trans subgraph to gpu
      epoch_transfer_graph_time += get_time();
      // std::cout << "train graph done" << std::endl;

      ///////// trans feature (zero copy or cache  version)//////////////////
      // if (graph->config->cache_rate <= 0) { // trans feature use zero
      // graph->config->cache_type = "none";
      if (graph->config->cache_type == "none") {  // trans feature use zero copy (omit gather feature)
        // sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
        // sampler->load_feature_gpu(cuda_stream, ssg, X[0], gnndatum->dev_local_feature);
        epoch_transfer_feat_time -= get_time();
        sampler->load_feature_gpu(&cuda_stream_list[0], ssg, X[0], gnndatum->dev_local_feature);
        // cuda_stream_list[0]->zero_copy_feature_move_gpu()
        epoch_transfer_feat_time += get_time();
        // trans freature which is not cache in gpu
        // } else if (graph->config->cache_type == "gpu_memory" && graph->rtminfo->epoch >= 5){
        if (graph->config->threshold_trans > 0) explicit_rate.push_back(cnt_suit_explicit_block(ssg));
      } else if (graph->config->cache_type == "gpu_memory" ||
                 graph->config->cache_type == "rate") {  // trans freature which is not cache in gpu
                                                         // trans_feature_cost -= get_time();
        // auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(
        //     X[0], gnndatum->dev_local_feature, dev_cache_feature, local_idx, local_idx_cache, cache_node_hashmap,
        //     dev_local_idx, dev_local_idx_cache, dev_cache_node_hashmap);

        epoch_transfer_feat_time -= get_time();
        auto [trans_feature_tmp, gather_gpu_cache_tmp] = sampler->load_feature_gpu_cache(
            &cuda_stream_list[0], ssg, X[0], gnndatum->dev_local_feature, dev_cache_feature, local_idx, local_idx_cache,
            cache_node_hashmap, dev_local_idx, dev_local_idx_cache, dev_cache_node_hashmap);
        epoch_transfer_feat_time += get_time();

        epoch_all_node += ssg->sampled_sgs[0]->src().size();
        for (auto& it : ssg->sampled_sgs[0]->src()) {
          if (cache_node_hashmap[it] != -1) {
            epoch_cache_hit++;
          }
        }
        if (graph->config->threshold_trans > 0)
          explicit_rate.push_back(cnt_suit_explicit_block(ssg, cache_node_hashmap));
      } else {
        std::cout << "cache_type: " << graph->config->cache_type << " is not support!" << std::endl;
        assert(false);
      }
      // std::cout << "load feature done" << std::endl;
      // /####/end trans feature (zero copy or cache version) ############/

      ///////start trans target_lab (zero copy) //////
      // sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      epoch_transfer_label_time -= get_time();
      sampler->load_label_gpu(&cuda_stream_list[0], ssg, target_lab, gnndatum->dev_local_label);
      epoch_transfer_label_time += get_time();
      // std::cout << "load label done" << std::endl;
      // /end trans target_lab (zero  copy)////////

      epoch_train_time -= get_time();
      at::cuda::setCurrentCUDAStream(torch_stream[0]);

      
      // auto tmp_vec = class_nodes[10];
      // // int *tmp = new int[tmp_vec.size()];
      // printf("nodes: %d\n",tmp_vec.size());
      // NtsVar M = graph->Nts->NewLeafTensor({tmp_vec.size(), F.size(1)}, torch::DeviceType::CUDA);
      // sampler->load_target_feature_gpu(&cuda_stream_list[0],M,tmp_vec,gnndatum->dev_local_feature);
      // std::cout<<"M sizes: "<<M.sizes()<<endl;
      // NtsVar M_W = P[0]->forward(M);
      // std::cout<<"M*W sizes: "<<M_W.sizes()<<endl;
      // std::cout<<"t sizes: "<<torch::sum(M_W,0).sizes()<<endl;
      // auto t = torch::div(torch::sum(M_W,0),static_cast<int>(tmp_vec.size()));
      // auto t1 = M_W[1];
      // double cossim = (torch::dot(t1,t).item<double>()) / ((torch::norm(t1).item<double>()) * (torch::norm(t).item<double>()));
      // // auto t = M_W.sum().div(tmp_vec.size());
      // std::cout<<"t sizes: "<<t.sizes()<<endl;
      // std::cout<<"cossim: "<<cossim<<endl;
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        // NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l]);
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
        
        // std::cout<<l<<"Y sizes: "<<Y_i.sizes()<<endl;

        // if(l == 1) {
        //   int index = 0;
        //   for(auto i:ssg->sampled_sgs[1]->destination) {
        //     if(gnndatum->local_label[i]==4) {
        //       nid_vec.push_back(i);
        //       // std::cout<<"agg "<<aggre_feat.sizes()<<"Y "<<Y_i[index].sizes()<<Y_i.dim()<<Y_i[index].dim()<<endl;
        //       aggre_feat = torch::cat({aggre_feat,Y_i[index].view({1,Y_i.size(1)})});
        //     }
        //     index++;
        //   }
        // }

        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
        // std::cout<<l<<"X sizes: "<<X[l+1].sizes()<<endl;
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (ctx->training == true) {
        ctx->appendNNOp(X[layers], loss_);
        ctx->self_backward(false);
        Update();
      }
      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      epoch_train_time += get_time();
      sampler->reverse_sgs();

    }

      // NtsVar M = graph->Nts->NewLeafTensor({nid_vec.size(), F.size(1)}, torch::DeviceType::CUDA);
      // sampler->load_target_feature_gpu(&cuda_stream_list[0],M,nid_vec,gnndatum->dev_local_feature);
      // NtsVar M_W = P[0]->forward(M);
      // auto t = torch::div(torch::sum(M_W,0),static_cast<int>(nid_vec.size()));
      // auto mean_feat = torch::div(torch::sum(aggre_feat,0),static_cast<int>(nid_vec.size()));
      // std::vector<double>before_cosSim;
      // std::vector<double>after_cosSim;
      // std::vector<double>diff_cosSim;
      // double cossim = 0.0;
      // // double test = 0.0; 
      // for(int i=0;i<nid_vec.size();i++) {
      //   auto x1 = M_W[i];
      //   if(torch::norm(x1).item<double>()==0) {
      //     cossim = 0;
      //   } else {
      //     cossim = (torch::dot(x1,t).item<double>()) / ((torch::norm(x1).item<double>()) * (torch::norm(t).item<double>()));
      //   }
        
      //   double cossim_aggre =  (torch::dot(aggre_feat[i],mean_feat).item<double>()) / ((torch::norm(aggre_feat[i]).item<double>()) * (torch::norm(mean_feat).item<double>()));
      //   // std::cout<<"nid: "<<nid_vec[i]<<"before cosSim: "<<cossim<<"after cosSim: "<<cossim_aggre<<endl;
      //   before_cosSim.push_back(cossim);
      //   // test+=cossim;
      //   // printf("%lf\n",test);
      //   after_cosSim.push_back(cossim_aggre);
      //   diff_cosSim.push_back(cossim_aggre-cossim);
      // }
      // double before_mean = 0.0;
      // double before_var = 0.0;
      // double after_mean = 0.0;
      // double after_var = 0.0;
      // double diff_mean = 0.0;
      // double diff_var = 0.0;
      // double tmp1 = 0.0;
      // double tmp2 = 0.0;
      // double tmp3 = 0.0;
      
      // before_mean = std::accumulate(before_cosSim.begin(),before_cosSim.end(),0.0)/before_cosSim.size();
      // after_mean = std::accumulate(after_cosSim.begin(),after_cosSim.end(),0.0)/after_cosSim.size();
      // diff_mean = std::accumulate(diff_cosSim.begin(),diff_cosSim.end(),0.0)/diff_cosSim.size();
      // for (int i=0;i<before_cosSim.size();i++) {
      //   tmp1 += (before_cosSim[i] - before_mean) * (before_cosSim[i] - before_mean);
      //   tmp2 += (after_cosSim[i] - after_mean) * (after_cosSim[i] - after_mean);
      //   tmp3 += (diff_cosSim[i] - diff_mean) * (diff_cosSim[i] - diff_mean);
      // }
      // before_var = tmp1/before_cosSim.size();
      // after_var = tmp2/after_cosSim.size();
      // diff_var = tmp3/diff_cosSim.size();
      // std::cout<<"before cos mean: "<<before_mean<<"before var: "<<before_var<<endl;
      // std::cout<<"after cos mean: "<<after_mean<<"after var: "<<after_var<<endl;
      // std::cout<<"diff cos mean: "<<diff_mean<<"diff var: "<<diff_var<<endl;

      // auto t1 = M_W[1];
      // double cossim = (torch::dot(t1,t).item<double>()) / ((torch::norm(t1).item<double>()) * (torch::norm(t).item<double>()));
      // std::cout<<"cossim: "<<cossim<<endl;

    // auto mean_feat = torch::div(torch::sum(aggre_feat,0),static_cast<int>(nid_vec.size()));
    // double cossim_aggre =  (torch::dot(aggre_feat[1],mean_feat).item<double>()) / ((torch::norm(aggre_feat[1]).item<double>()) * (torch::norm(mean_feat).item<double>()));
    // cout<<"aggre feat: "<<aggre_feat.sizes()<<"cos sim: "<<cossim_aggre<<endl;


    // for(int m=0;m<class_size;m++) {
    //   printf("class: %d\n",m);
    //   for(int n=0; n<class_size;n++) {
    //     printf("1-hop nbrs class: %d has %d nodes\n",n,result_1[m*class_size+n]);
    //   }
    //   for(int n=0; n<class_size;n++) {
    //     printf("2-hop nbrs class: %d has %d nodes\n",n,result_2[m*class_size+n]);
    //   }
    // }

    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
    if (graph->config->threshold_trans > 0) {
      LOG_DEBUG("epoch suit explicit trans block rate %.3f(%.3f)", get_mean(explicit_rate), get_var(explicit_rate));
    }
  }


  NtsVar vertexForward(NtsVar& n_i) {
    int l = graph->rtminfo->curr_layer;
    if (l == layers - 1) {  // last layer
      return P[l]->forward(n_i);
    } else {
      if (graph->config->batch_norm) {
        // std::cout << n_i.device() << std::endl;
        // std::cout << this->bn1d[l].device() << std::endl;
        n_i = this->bn1d[l](n_i);  // for arxiv dataset
        // n_i = torch::batch_norm(n_i);
      }
      return torch::dropout(torch::relu(P[l]->forward(n_i)), drop_rate, ctx->is_train());
    }
  }

  // float Forward(Sampler* sampler, int type = 0) {
  std::pair<float, double> Forward(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = true;
    correct = 0;
    train_nodes = 0;
    float f1_epoch = 0;
    batch = 0;
    loss_epoch = 0;
    double nn_cost = 0;
    double backward_cost = 0;

    // enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS};
    if (graph->config->batch_type == SHUFFLE || graph->config->batch_type == RANDOM ||
        graph->config->batch_type == DELLOW || graph->config->batch_type == DELHIGH) {
      shuffle_vec_seed(sampler->sample_nids);
    }

    // node sampling
    // sampler->zero_debug_time();

    epoch_sample_time = 0;
    epoch_gather_label_time = 0;
    epoch_gather_feat_time = 0;
    epoch_transfer_graph_time = 0;
    epoch_transfer_feat_time = 0;
    epoch_transfer_label_time = 0;
    epoch_train_time = 0;
    epoch_cache_hit = 0;
    epoch_all_node = 0;
    debug_time = 0;


    int batch_num = sampler->batch_nums;

    // old_version(sampler);

    if (graph->config->mode == "pipeline") {
      LOG_DEBUG("pipeline version");
      pipeline_version(sampler);
    } else if (graph->config->mode == "explicit") {
      LOG_DEBUG("explicit version");
      explicit_version(sampler);
    } else if (graph->config->mode == "zerocopy") {
      LOG_DEBUG("zerocopy version");
      zerocopy_version(sampler);
    } else {
      LOG_DEBUG("not support");
      assert(false);
    }

    loss_epoch /= sampler->batch_nums;

    if (graph->config->classes > 1) {
      f1_epoch /= sampler->batch_nums;
      MPI_Allreduce(MPI_IN_PLACE, &f1_epoch, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      acc = f1_epoch / hosts;
    } else {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
      acc = 1.0 * correct / train_nodes;
    }

    double epoch_trans_time = epoch_transfer_graph_time + epoch_transfer_feat_time + epoch_transfer_label_time;
    double epoch_all_train_time = epoch_sample_time + epoch_train_time + epoch_trans_time;
    gcn_run_time += epoch_all_train_time;

    return {acc, epoch_all_train_time};
  }

  float EvalForward(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = false;
    correct = 0;
    train_nodes = 0;
    float f1_epoch = 0;
    batch = 0;
    loss_epoch = 0;

    // enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS};
    if (graph->config->batch_type == SHUFFLE || graph->config->batch_type == RANDOM ||
        graph->config->batch_type == DELLOW || graph->config->batch_type == DELHIGH) {
      shuffle_vec_seed(sampler->sample_nids);
    }

    // node sampling
    sampler->zero_debug_time();
    int batch_num = sampler->batch_nums;

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }

    std::vector<int>num_class_tag;
    std::vector<int>num_class_pre;
    num_class_tag.resize(graph->gnnctx->label_num);
    num_class_pre.resize(graph->gnnctx->label_num);    
    int class_size = graph->gnnctx->label_num;
    int *result = new int[class_size*class_size];
    memset(result,0,sizeof(int)*(class_size*class_size));



    sampler->metis_batch_id = 0;

    std::vector<VertexId>epoch_nodes;
    VertexId node_size = 0;
    VertexId edge_size = 0;
    VertexId tmp = 0;
    int i =0;
    std::map<int,std::vector<float>>class_similarity;
    
    // int class_size = graph->gnnctx->label_num;
    int *result_1 = new int[graph->gnnctx->label_num*graph->gnnctx->label_num];
    int *result_2 = new int[graph->gnnctx->label_num*graph->gnnctx->label_num];
    memset(result_1,0,sizeof(int)*(graph->gnnctx->label_num*graph->gnnctx->label_num));
    memset(result_2,0,sizeof(int)*(graph->gnnctx->label_num*graph->gnnctx->label_num));

    while (sampler->work_offset < sampler->work_range[1]) {
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_by_label(ssg, graph->config->batch_type, ctx->is_train(),nid_nbr);
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());

      // node_size = node_size + ssg->sampled_sgs[1]->v_size + ssg->sampled_sgs[1]->src_size + ssg->sampled_sgs[0]->src_size;
      // edge_size = edge_size + ssg->sampled_sgs[0]->e_size + ssg->sampled_sgs[1]->e_size;
      // auto d_0 = ssg->sampled_sgs[0]->dst();
      // auto s_0 = ssg->sampled_sgs[0]->src();
      // auto d_1 = ssg->sampled_sgs[1]->dst();
      // auto s_1 = ssg->sampled_sgs[1]->src();
      // printf("batch: %d col_off_1 %d row_ind_1 %d  col_off_0 %d row_ind_0 %d d1 %d s1 %d d0 %d s0 %d\n",i,ssg->sampled_sgs[1]->column_offset.size(),ssg->sampled_sgs[1]->row_indices.size(),ssg->sampled_sgs[0]->column_offset.size(),ssg->sampled_sgs[0]->row_indices.size(),d_1.size(),s_1.size(),d_0.size(),s_0.size());
      // for(int index = 0;index<d_1.size();index++) {
      //   int nid = d_1[index];
      //   float one_nbr_sim = 0.0;
      //   float two_nbr_sim = 0.0;
      //   int cur_class = gnndatum->local_label[nid];
      //   int one_same_num = 0;
      //   int two_same_num = 0;
      //   int one_nbr_num = ssg->sampled_sgs[1]->column_offset[index+1] - ssg->sampled_sgs[1]->column_offset[index];
      //   int two_nbr_num = 0;
      //   for(int j=ssg->sampled_sgs[1]->column_offset[index];j<ssg->sampled_sgs[1]->column_offset[index+1];j++) {
      //     int one_nbr_indice = ssg->sampled_sgs[1]->row_indices[j];
      //     int one_nbr = s_1[one_nbr_indice];
      //     result_1[class_size*cur_class+gnndatum->local_label[one_nbr]]+=1;
      //     std::vector<VertexId>::iterator it = find(d_0.begin(),d_0.end(),one_nbr);
      //     int two_index = &*it-&d_0[0];
      //     // printf("one nbr index: %d,nid: %d\n",two_index,one_nbr);
      //     two_nbr_num += ssg->sampled_sgs[0]->column_offset[two_index+1] - ssg->sampled_sgs[0]->column_offset[two_index];
      //     if(gnndatum->local_label[one_nbr] == cur_class) {
      //       one_same_num ++;
      //     }
      //     for(int n=ssg->sampled_sgs[0]->column_offset[two_index];n<ssg->sampled_sgs[0]->column_offset[two_index+1];n++) {
      //       result_2[class_size*cur_class+gnndatum->local_label[ssg->sampled_sgs[0]->row_indices[n]]]+=1;
      //       if(gnndatum->local_label[ssg->sampled_sgs[0]->row_indices[n]] == cur_class) {
      //       two_same_num ++;
      //       }
      //     }
      //   }
      //   one_nbr_sim = float(one_same_num)/one_nbr_num;
      //   two_nbr_sim = float(two_same_num)/two_nbr_num;
      //   if(class_similarity.count(cur_class)==0){
      //     std::vector<float>tmp_vec;
      //     tmp_vec.push_back(two_nbr_sim);
      //     class_similarity.insert(std::make_pair(cur_class,tmp_vec));
      //   }else{
      //     class_similarity[cur_class].push_back(two_nbr_sim);
      //   }
      //   // LOG_INFO("batch %d nid %d class %d 1-hop-sim: %f 2-hop-sim: %f",i,nid,cur_class,one_nbr_sim,two_nbr_sim);
      // }

      // // std::vector<int>result;
      // // result.clear();
      // // for(const auto&v:d_1){
      // //   result.push_back(v);
      // //   epoch_nodes.push_back(v);
      // // }
      // // for(const auto&v:s_1){
      // //   result.push_back(v);
      // //   epoch_nodes.push_back(v);
      // // }
      // // for(const auto&v:s_0){
      // //   result.push_back(v);
      // //   epoch_nodes.push_back(v);
      // // }
      // // printf("batch: %d nodes num: %d\n",i,result.size());
      // // std::set<int>s(result.begin(),result.end());
      // // result.assign(s.begin(),s.end());
      // // printf("after batch: %d nodes num: %d\n",i,result.size());
      // // for(auto it:class_similarity) {
      // //   double sum = std::accumulate(std::begin(it.second), std::end(it.second), 0.0);  
      // //   double mean =  sum / it.second.size(); //均值
      // //   double accum  = 0.0; 
      // //   std::for_each (std::begin(it.second), std::end(it.second), [&](const double d) {  
      // //     accum  += (d-mean)*(d-mean);  
      // //   }); 
      // //   double var = sqrt(accum/(it.second.size()-1)); //方差 
      // // LOG_INFO("class: %d two-sim-mean: %lf two-sim-var: %lf",it.first,mean,var);
      // // }
      // i = i+1;


      // ssg->trans_graph_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);  // trans subgraph to gpu

      // sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      sampler->load_feature_gpu(&cuda_stream_list[0], ssg, X[0], gnndatum->dev_local_feature);
      
      // sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      sampler->load_label_gpu(&cuda_stream_list[0], ssg, target_lab, gnndatum->dev_local_label);

      if (ctx->is_train()) zero_grad();   // should zero grad after every mini batch compute
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
        // NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], cuda_stream);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        NtsVar tmp_output = X[layers].argmax(1).to(torch::kLong).to(torch::kLong);
        NtsVar output = X[layers].argmax(1).to(torch::kLong).eq(target_lab).to(torch::kLong);
        // int* result = output.to(torch::kInt32).to(torch::kCPU).data_ptr<int>();
        // int* target = target_lab.to(torch::kInt32).to(torch::kCPU).data_ptr<int>();
        // std::cout<<tmp_output<<endl;
        // std::cout<<target_lab<<endl;
        // LOG_INFO("work_offset: %d",sampler->work_offset);
        // printf("%d %d\n",target[0],target[1]);
        // std::cout<<output[0]<<endl;
        // std::cout<<output.dtype()<<endl;
        // std::cout<<target_lab[0]<<endl;
        // std::cout<<target_lab.dtype()<<endl;

        for(int i=0;i<sampler->work_offset;i++) {
          
          num_class_tag[target_lab[i].item<int>()]++;
          if(output[i].item<int>() == 1) {
            num_class_pre[target_lab[i].item<int>()]++;
            result[class_size*target_lab[i].item<int>()+target_lab[i].item<int>()]+=1;
          } else {
             result[class_size*target_lab[i].item<int>()+tmp_output[i].item<int>()]+=1;
            // for(int j=0;j<tmp_output.size(0);j++) {
            //   if(tmp_output[j].item<int>() == target_lab[j].item<int>()) {
            //     result[class_size*target_lab[i].item<int>()+j]+=1;
            //   }
            // }
          }

        }


        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      sampler->reverse_sgs();
    }

    // for(int m=0;m<class_size;m++) {
    //   printf("class: %d\n",m);
    //   for(int n=0; n<class_size;n++) {
    //     printf("1-hop nbrs class: %d has %d nodes\n",n,result_1[m*class_size+n]);
    //   }
    //   for(int n=0; n<class_size;n++) {
    //     printf("2-hop nbrs class: %d has %d nodes\n",n,result_2[m*class_size+n]);
    //   }
    // }

    assert(sampler->work_offset == sampler->work_range[1]);
    // LOG_INFO("test!!");

    // for(int i=0;i<class_size;i++) {
    //   printf("val class: %d\n",i);
    //   for(int j=0;j<class_size;j++) {
    //     printf("val result class %d has %d nodes\n",j,result[i*class_size+j]);
    //   }
    // }
    // for(int index=0;index<graph->gnnctx->label_num;index++)
    // {
    //   LOG_INFO("label %d nodes %d right nodes %d Val Acc %.2f",index,num_class_tag[index],num_class_pre[index],float(num_class_pre[index])/float(num_class_tag[index]));
    // }

    sampler->restart();

    loss_epoch /= sampler->batch_nums;
    if (graph->config->classes > 1) {
      f1_epoch /= sampler->batch_nums;
      MPI_Allreduce(MPI_IN_PLACE, &f1_epoch, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      acc = f1_epoch / hosts;
    } else {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
      acc = 1.0 * correct / train_nodes;
    }
    return acc;
  }

float EvalForward_2(Sampler* sampler, int type = 0) {
    graph->rtminfo->forward = false;
    correct = 0;
    train_nodes = 0;
    float f1_epoch = 0;
    batch = 0;
    loss_epoch = 0;

    // enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS};
    if (graph->config->batch_type == SHUFFLE || graph->config->batch_type == RANDOM ||
        graph->config->batch_type == DELLOW || graph->config->batch_type == DELHIGH) {
      shuffle_vec_seed(sampler->sample_nids);
    }

    // node sampling
    sampler->zero_debug_time();
    int batch_num = sampler->batch_nums;

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }

    std::vector<int>num_class_tag;
    std::vector<int>num_class_pre;
    num_class_tag.resize(graph->gnnctx->label_num);
    num_class_pre.resize(graph->gnnctx->label_num);    
    int class_size = graph->gnnctx->label_num;
    int *result = new int[class_size*class_size];
    memset(result,0,sizeof(int)*(class_size*class_size));



    sampler->metis_batch_id = 0;

    std::vector<VertexId>epoch_nodes;
    VertexId node_size = 0;
    VertexId edge_size = 0;
    VertexId tmp = 0;
    int i =0;
    std::map<int,std::vector<float>>class_similarity;
    
    // int class_size = graph->gnnctx->label_num;
    int *result_1 = new int[graph->gnnctx->label_num*graph->gnnctx->label_num];
    int *result_2 = new int[graph->gnnctx->label_num*graph->gnnctx->label_num];
    memset(result_1,0,sizeof(int)*(graph->gnnctx->label_num*graph->gnnctx->label_num));
    memset(result_2,0,sizeof(int)*(graph->gnnctx->label_num*graph->gnnctx->label_num));

    while (sampler->work_offset < sampler->work_range[1]) {
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      // sampler->sample_one_by_label(ssg, graph->config->batch_type, ctx->is_train(),nid_nbr);
      // sampler->sample_one_by_label(ssg, graph->config->batch_type, ctx->is_train(),is_sketch,gnndatum);
      // sampler->sample_one_with_dst(ssg, graph->config->batch_type, ctx->is_train());

      node_size = node_size + ssg->sampled_sgs[1]->v_size + ssg->sampled_sgs[1]->src_size + ssg->sampled_sgs[0]->src_size;
      edge_size = edge_size + ssg->sampled_sgs[0]->e_size + ssg->sampled_sgs[1]->e_size;
      auto d_0 = ssg->sampled_sgs[0]->dst();
      auto s_0 = ssg->sampled_sgs[0]->src();
      auto d_1 = ssg->sampled_sgs[1]->dst();
      auto s_1 = ssg->sampled_sgs[1]->src();
      printf("batch: %d col_off_1 %d row_ind_1 %d  col_off_0 %d row_ind_0 %d d1 %d s1 %d d0 %d s0 %d\n",i,ssg->sampled_sgs[1]->column_offset.size(),ssg->sampled_sgs[1]->row_indices.size(),ssg->sampled_sgs[0]->column_offset.size(),ssg->sampled_sgs[0]->row_indices.size(),d_1.size(),s_1.size(),d_0.size(),s_0.size());
      for(int index = 0;index<d_1.size();index++) {
        int nid = d_1[index];
        float one_nbr_sim = 0.0;
        float two_nbr_sim = 0.0;
        int cur_class = gnndatum->local_label[nid];
        int one_same_num = 0;
        int two_same_num = 0;
        int one_nbr_num = ssg->sampled_sgs[1]->column_offset[index+1] - ssg->sampled_sgs[1]->column_offset[index];
        int two_nbr_num = 0;
        for(int j=ssg->sampled_sgs[1]->column_offset[index];j<ssg->sampled_sgs[1]->column_offset[index+1];j++) {
          int one_nbr_indice = ssg->sampled_sgs[1]->row_indices[j];
          int one_nbr = s_1[one_nbr_indice];
          result_1[class_size*cur_class+gnndatum->local_label[one_nbr]]+=1;
          std::vector<VertexId>::iterator it = find(d_0.begin(),d_0.end(),one_nbr);
          int two_index = &*it-&d_0[0];
          // printf("one nbr index: %d,nid: %d\n",two_index,one_nbr);
          two_nbr_num += ssg->sampled_sgs[0]->column_offset[two_index+1] - ssg->sampled_sgs[0]->column_offset[two_index];
          if(gnndatum->local_label[one_nbr] == cur_class) {
            one_same_num ++;
          }
          for(int n=ssg->sampled_sgs[0]->column_offset[two_index];n<ssg->sampled_sgs[0]->column_offset[two_index+1];n++) {
            result_2[class_size*cur_class+gnndatum->local_label[ssg->sampled_sgs[0]->row_indices[n]]]+=1;
            if(gnndatum->local_label[ssg->sampled_sgs[0]->row_indices[n]] == cur_class) {
            two_same_num ++;
            }
          }
        }
        one_nbr_sim = float(one_same_num)/one_nbr_num;
        two_nbr_sim = float(two_same_num)/two_nbr_num;
        if(class_similarity.count(cur_class)==0){
          std::vector<float>tmp_vec;
          tmp_vec.push_back(two_nbr_sim);
          class_similarity.insert(std::make_pair(cur_class,tmp_vec));
        }else{
          class_similarity[cur_class].push_back(two_nbr_sim);
        }
        // LOG_INFO("batch %d nid %d class %d 1-hop-sim: %f 2-hop-sim: %f",i,nid,cur_class,one_nbr_sim,two_nbr_sim);
      }

      // // std::vector<int>result;
      // // result.clear();
      // // for(const auto&v:d_1){
      // //   result.push_back(v);
      // //   epoch_nodes.push_back(v);
      // // }
      // // for(const auto&v:s_1){
      // //   result.push_back(v);
      // //   epoch_nodes.push_back(v);
      // // }
      // // for(const auto&v:s_0){
      // //   result.push_back(v);
      // //   epoch_nodes.push_back(v);
      // // }
      // // printf("batch: %d nodes num: %d\n",i,result.size());
      // // std::set<int>s(result.begin(),result.end());
      // // result.assign(s.begin(),s.end());
      // // printf("after batch: %d nodes num: %d\n",i,result.size());
      // // for(auto it:class_similarity) {
      // //   double sum = std::accumulate(std::begin(it.second), std::end(it.second), 0.0);  
      // //   double mean =  sum / it.second.size(); //均值
      // //   double accum  = 0.0; 
      // //   std::for_each (std::begin(it.second), std::end(it.second), [&](const double d) {  
      // //     accum  += (d-mean)*(d-mean);  
      // //   }); 
      // //   double var = sqrt(accum/(it.second.size()-1)); //方差 
      // // LOG_INFO("class: %d two-sim-mean: %lf two-sim-var: %lf",it.first,mean,var);
      // // }
      // i = i+1;


      // ssg->trans_graph_to_gpu(graph->config->mini_pull > 0);  // wheather trans csr data to gpu
      ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);  // trans subgraph to gpu

      // sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      sampler->load_feature_gpu(&cuda_stream_list[0], ssg, X[0], gnndatum->dev_local_feature);
      
      // sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      sampler->load_label_gpu(&cuda_stream_list[0], ssg, target_lab, gnndatum->dev_local_label);

      if (ctx->is_train()) zero_grad();   // should zero grad after every mini batch compute
      for (int l = 0; l < layers; l++) {  // forward
        graph->rtminfo->curr_layer = l;
        NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
        // NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], cuda_stream);
        X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
      }

      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();

      if (graph->config->classes == 1) {
        correct += get_correct(X[layers], target_lab, graph->config->classes == 1);
        NtsVar tmp_output = X[layers].argmax(1).to(torch::kLong).to(torch::kLong);
        NtsVar output = X[layers].argmax(1).to(torch::kLong).eq(target_lab).to(torch::kLong);
        // int* result = output.to(torch::kInt32).to(torch::kCPU).data_ptr<int>();
        // int* target = target_lab.to(torch::kInt32).to(torch::kCPU).data_ptr<int>();
        // std::cout<<tmp_output<<endl;
        // std::cout<<target_lab<<endl;
        // LOG_INFO("work_offset: %d",sampler->work_offset);
        // printf("%d %d\n",target[0],target[1]);
        // std::cout<<output[0]<<endl;
        // std::cout<<output.dtype()<<endl;
        // std::cout<<target_lab[0]<<endl;
        // std::cout<<target_lab.dtype()<<endl;

        for(int i=0;i<sampler->work_offset;i++) {
          
          num_class_tag[target_lab[i].item<int>()]++;
          if(output[i].item<int>() == 1) {
            num_class_pre[target_lab[i].item<int>()]++;
            result[class_size*target_lab[i].item<int>()+target_lab[i].item<int>()]+=1;
          } else {
             result[class_size*target_lab[i].item<int>()+tmp_output[i].item<int>()]+=1;
            // for(int j=0;j<tmp_output.size(0);j++) {
            //   if(tmp_output[j].item<int>() == target_lab[j].item<int>()) {
            //     result[class_size*target_lab[i].item<int>()+j]+=1;
            //   }
            // }
          }

        }


        train_nodes += target_lab.size(0);
      } else {
        f1_epoch += f1_score(X[layers], target_lab, graph->config->classes == 1);
      }
      sampler->reverse_sgs();
    }

    for(int m=0;m<class_size;m++) {
      printf("class: %d\n",m);
      for(int n=0; n<class_size;n++) {
        printf("1-hop nbrs class: %d has %d nodes\n",n,result_1[m*class_size+n]);
      }
      for(int n=0; n<class_size;n++) {
        printf("2-hop nbrs class: %d has %d nodes\n",n,result_2[m*class_size+n]);
      }
    }

    assert(sampler->work_offset == sampler->work_range[1]);
    // LOG_INFO("test!!");

    for(int i=0;i<class_size;i++) {
      printf("val class: %d\n",i);
      for(int j=0;j<class_size;j++) {
        printf("val result class %d has %d nodes\n",j,result[i*class_size+j]);
      }
    }
    for(int index=0;index<graph->gnnctx->label_num;index++)
    {
      LOG_INFO("label %d nodes %d right nodes %d Val Acc %.2f",index,num_class_tag[index],num_class_pre[index],float(num_class_pre[index])/float(num_class_tag[index]));
    }

    sampler->restart();

    loss_epoch /= sampler->batch_nums;
    if (graph->config->classes > 1) {
      f1_epoch /= sampler->batch_nums;
      MPI_Allreduce(MPI_IN_PLACE, &f1_epoch, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      acc = f1_epoch / hosts;
    } else {
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      MPI_Allreduce(MPI_IN_PLACE, &correct, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &train_nodes, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      if (type == 0 && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
      acc = 1.0 * correct / train_nodes;
    }
    return acc;
  }

  void zero_grad() {
    for (int i = 0; i < P.size(); i++) {
      P[i]->zero_grad();
    }
  }

  void saveW(std::string suffix="") {
    for (int i = 0; i < layers; ++i) {
      P[i]->save_W("/home/yuanh/neutron-sanzo/saved_modules", graph->config->dataset_name + suffix, i);
    }
  }

  void loadW(std::string suffix = "") {
    for (int i = 0; i < layers; ++i) {
      P[i]->load_W("/home/yuanh/neutron-sanzo/saved_modules", graph->config->dataset_name + suffix, i);
    }
  }


  void gater_cpu_cache_feature_and_trans_to_gpu() {
    long feat_dim = graph->gnnctx->layer_size[0];
    // LOG_DEBUG("feat_dim %d", feat_dim);
    dev_cache_feature = (float*)cudaMallocGPU(cache_node_num * sizeof(float) * feat_dim);
    // gather_cache_feature, prepare trans to gpu
    // LOG_DEBUG("start gather_cpu_cache_feature");
    float* local_cache_feature_gather = new float[cache_node_num * feat_dim];
// #pragma omp parallel for
// omp_set_num_threads(threads);
#pragma omp parallel for num_threads(threads)
    for (int i = 0; i < cache_node_num; ++i) {
      int node_id = cache_node_idx_seq[i];
      // assert(node_id < graph->vertices);
      // assert(node_id < graph->gnnctx->l_v_num);
      // LOG_DEBUG("copy node_id %d to", node_id);
      // LOG_DEBUG("local_id %d", cache_node_hashmap[node_id]);

      for (int j = 0; j < feat_dim; ++j) {
        assert(cache_node_hashmap[node_id] < cache_node_num);
        local_cache_feature_gather[cache_node_hashmap[node_id] * feat_dim + j] =
            gnndatum->local_feature[node_id * feat_dim + j];
      }
    }
    LOG_DEBUG("start trans to gpu");
    move_data_in(dev_cache_feature, local_cache_feature_gather, 0, cache_node_num, feat_dim);
    local_idx = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    local_idx_cache = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    dev_local_idx = (VertexId*)getDevicePointer(local_idx);
    dev_local_idx_cache = (VertexId*)getDevicePointer(local_idx_cache);
  }

  void mark_cache_node(std::vector<int>& cache_nodes) {
    // init mask
    // #pragma omp parallel for
    // #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < graph->vertices; ++i) {
      cache_node_hashmap[i] = -1;
      // assert(cache_node_hashmap[i] == -1);
    }

    // mark cache nodes
    int tmp_idx = 0;
    for (int i = 0; i < cache_node_num; ++i) {
      // LOG_DEBUG("cache_nodes[%d] = %d", i, cache_nodes[i]);
      cache_node_hashmap[cache_nodes[i]] = tmp_idx++;
    }
    LOG_DEBUG("cache_node_num %d tmp_idx %d", cache_node_num, tmp_idx);
    assert(cache_node_num == tmp_idx);

    // // debug
    // int cache_node_hashmap_num = 0;
    // for (int i = 0; i < graph->vertices; ++i) {
    //   // LOG_DEBUG("cache_node_hashmap[%d] = %d", i, cache_node_hashmap[i]);
    //   cache_node_hashmap_num += cache_node_hashmap[i] != -1; // unsigned
    // }
    // LOG_DEBUG("cache_node_hashmap_num %d", cache_node_hashmap_num);
    // assert(cache_node_hashmap_num == cache_node_num);
  }

  void cache_high_degree(std::vector<int>& node_idx) {
    std::sort(node_idx.begin(), node_idx.end(), [&](const int x, const int y) {
      return graph->out_degree_for_backward[x] > graph->out_degree_for_backward[y];
    });
    // #pragma omp parallel for num_threads(threads)
    for (int i = 1; i < graph->vertices; ++i) {
      assert(graph->out_degree_for_backward[node_idx[i]] <= graph->out_degree_for_backward[node_idx[i - 1]]);
    }
    mark_cache_node(node_idx);
  }

  void cache_random_node(std::vector<int>& node_idx) {
    shuffle_vec_seed(node_idx);
    mark_cache_node(node_idx);
  }

  
  void cache_sample(std::vector<int>& node_idx) {
    std::vector<int> node_sample_cnt(node_idx.size(), 0);
    int epochs = 1;
    auto ssg = train_sampler->subgraph;
    for (int i = 0; i < epochs; ++i) {
      while (train_sampler->work_offset < train_sampler->work_range[1]) {
        train_sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
        for (int i = 0; i < layers; ++i) {
          auto p = ssg->sampled_sgs[i]->src();
          for (const auto& v : p) {
            node_sample_cnt[v]++;
          }
          // LOG_DEBUG("layer %d %d", i, p.size());
        }
        train_sampler->reverse_sgs();
      }
      train_sampler->restart();
    }
    sort(node_idx.begin(), node_idx.end(),
         [&](const int x, const int y) { return node_sample_cnt[x] > node_sample_cnt[y]; });
    // for (int i = 1; i < node_idx.size(); ++i) {
    //   assert(node_sample_cnt[node_idx[i - 1]] >= node_sample_cnt[node_idx[i]]);
    // }
    mark_cache_node(node_idx);
  }



    double get_gpu_idle_mem() {
    // store degree
    VertexId* outs_bak = new VertexId[graph->vertices];
    VertexId* ins_bak = new VertexId[graph->vertices];
    for (int i = 0; i < graph->vertices; ++i) {
      outs_bak[i] = graph->out_degree_for_backward[i];
      ins_bak[i] = graph->in_degree_for_backward[i];
    }

    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
    }
    double max_gpu_used = 0;

    int epochs = 1;
    for (int i = 0; i < epochs; ++i) {
      auto ssg = train_sampler->subgraph;
      while (train_sampler->work_offset < train_sampler->work_range[1]) {
        if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
        train_sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
        ssg->trans_graph_to_gpu_async(cuda_stream_list[0].stream,
                                      graph->config->mini_pull > 0);  // trans subgraph to gpu
        if (hosts > 1) {
          X[0] = nts::op::get_feature_from_global(*rpc, ssg->sampled_sgs[0]->src().data(),
                                                  ssg->sampled_sgs[0]->src_size, F, graph);
        } else {
          X[0] = nts::op::get_feature(ssg->sampled_sgs[0]->src().data(), ssg->sampled_sgs[0]->src_size, F, graph);
        }
        X[0] = X[0].cuda().set_requires_grad(true);

        if (hosts > 1) {
          target_lab = nts::op::get_label_from_global(ssg->sampled_sgs.back()->dst().data(),
                                                      ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
        } else {
          target_lab =
              nts::op::get_label(ssg->sampled_sgs.back()->dst().data(), ssg->sampled_sgs.back()->v_size, L_GT_C, graph);
        }
        target_lab = target_lab.cuda();

        for (int l = 0; l < layers; l++) {
          graph->rtminfo->curr_layer = l;
          NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l], &cuda_stream_list[0]);
          X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
        }

        auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
        if (ctx->training == true) {
          ctx->appendNNOp(X[layers], loss_);
          ctx->self_backward(false);
          // Update();
        }

        if (graph->config->classes == 1) {
          get_correct(X[layers], target_lab, graph->config->classes == 1);
          // target_lab.size(0);
        } else {
          f1_score(X[layers], target_lab, graph->config->classes == 1);
        }
        train_sampler->reverse_sgs();
      }
      train_sampler->restart();
      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      max_gpu_used = std::max(used_gpu_mem, max_gpu_used);
      LOG_DEBUG("get_gpu_idle_mem(): used %.3f max_used %.3f total %.3f", used_gpu_mem, max_gpu_used, total_gpu_mem);
    }

    // restore degree
    for (int i = 0; i < graph->vertices; ++i) {
      graph->out_degree_for_backward[i] = outs_bak[i];
      graph->in_degree_for_backward[i] = ins_bak[i];
    }
    delete[] outs_bak;
    delete[] ins_bak;

    return max_gpu_used;
  }

  // pre train some epochs to get idle memory of GPU when training
  double get_gpu_idle_mem_pipe() {
    // store degree
    // std::cout << "pipelines " << pipelines << std::endl;
    // std::cout << "vertices " << graph->vertices << std::endl;
    VertexId* outs_bak = new VertexId[graph->vertices];
    VertexId* ins_bak = new VertexId[graph->vertices];
    for (int i = 0; i < graph->vertices; ++i) {
      outs_bak[i] = graph->out_degree_for_backward[i];
      ins_bak[i] = graph->in_degree_for_backward[i];
    }
    // std::cout << "here " <<std::endl;

    double max_gpu_used = 0;

    NtsVar tmp_X0[pipelines];
    NtsVar tmp_target_lab[pipelines];
    // std::cout << "here " <<std::endl;

    for (int i = 0; i < pipelines; i++) {
      tmp_X0[i] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
      if (graph->config->classes > 1) {
        tmp_target_lab[i] =
            graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
      } else {
        tmp_target_lab[i] = graph->Nts->NewLabelTensor({graph->config->batch_size}, torch::DeviceType::CUDA);
      }
    }
    // std::cout << "here " <<std::endl;

    auto sampler = train_sampler;

    for (int i = 0; i < 1; ++i) {
      std::thread threads[pipelines];
      for (int tid = 0; tid < pipelines; ++tid) {
        threads[tid] = std::thread(
            [&](int thread_id) {
              ////////////////////////////////// sample //////////////////////////////////
              // std::cout << "start thread" << std::endl;
              std::unique_lock<std::mutex> sample_lock(sample_mutex, std::defer_lock);
              sample_lock.lock();
              // std::cout << "start sample" << std::endl;
              while (sampler->work_offset < sampler->work_range[1]) {
                auto ssg = sampler->subgraph_list[thread_id];
                sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
                // std::cout << "sample one done" << std::endl;
                cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
                sample_lock.unlock();
                // get_gpu_mem(used_gpu_mem, total_gpu_mem);

                ////////////////////////////////// transfer //////////////////////////////////
                std::unique_lock<std::mutex> transfer_lock(transfer_mutex, std::defer_lock);
                transfer_lock.lock();
                ssg->trans_graph_to_gpu_async(cuda_stream_list[thread_id].stream, graph->config->mini_pull > 0);
                // std::cout << "train_graph done" << std::endl;
                sampler->load_feature_gpu(&cuda_stream_list[thread_id], ssg, tmp_X0[thread_id],
                                          gnndatum->dev_local_feature);
                // std::cout << "load_featrure done" << std::endl;
                sampler->load_label_gpu(&cuda_stream_list[thread_id], ssg, tmp_target_lab[thread_id],
                                        gnndatum->dev_local_label);
                // std::cout << "load_label done" << std::endl;
                cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
                transfer_lock.unlock();

                ////////////////////////////////// train //////////////////////////////////
                std::unique_lock<std::mutex> train_lock(train_mutex, std::defer_lock);
                train_lock.lock();
                at::cuda::setCurrentCUDAStream(torch_stream[thread_id]);
                if (ctx->training == true) zero_grad();  // should zero grad after every mini batch compute
                for (int l = 0; l < layers; l++) {       // forward
                  graph->rtminfo->curr_layer = l;
                  if (l == 0) {
                    NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, tmp_X0[thread_id],
                                                                                  &cuda_stream_list[thread_id]);
                    X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                  } else {
                    NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(ssg, graph, l, X[l],
                                                                                  &cuda_stream_list[thread_id]);
                    X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
                  }
                }

                auto loss_ = Loss(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                if (ctx->training == true) {
                  ctx->appendNNOp(X[layers], loss_);
                  ctx->self_backward(false);
                  // Update();
                }

                if (graph->config->classes == 1) {
                  get_correct(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                } else {
                  f1_score(X[layers], tmp_target_lab[thread_id], graph->config->classes == 1);
                }
                cudaStreamSynchronize(cuda_stream_list[thread_id].stream);
                train_lock.unlock();

                sample_lock.lock();
                sampler->reverse_sgs(ssg);
              }
              sample_lock.unlock();
            },
            tid);
      }
    // std::cout << "here " <<std::endl;


      for (int tid = 0; tid < pipelines; ++tid) {
        threads[tid].join();
      }

      sampler->restart();

      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      max_gpu_used = std::max(used_gpu_mem, max_gpu_used);
      LOG_DEBUG("get_gpu_idle_mem_pipe(): epoch %d used %.3f max_used %.3f total %.3f", i, used_gpu_mem, max_gpu_used,
                total_gpu_mem);
    }

    // restore degree
    for (int i = 0; i < graph->vertices; ++i) {
      graph->out_degree_for_backward[i] = outs_bak[i];
      graph->in_degree_for_backward[i] = ins_bak[i];
    }
    delete[] outs_bak;
    delete[] ins_bak;

    return max_gpu_used;
  }

  void determine_cache_node_idx(int node_nums) {
    if (node_nums > graph->vertices) node_nums = graph->vertices;
    cache_node_num = node_nums;
    LOG_DEBUG("cache_node_num %d (%.3f)", cache_node_num, 1.0 * cache_node_num / graph->vertices);

    cache_node_idx_seq.resize(graph->vertices);
    std::iota(cache_node_idx_seq.begin(), cache_node_idx_seq.end(), 0);
    // cache_node_hashmap.resize(graph->vertices);
    cache_node_hashmap = (VertexId*)cudaMallocPinned(1ll * graph->vertices * sizeof(VertexId));
    dev_cache_node_hashmap = (VertexId*)getDevicePointer(cache_node_hashmap);

    // #pragma omp parallel for
    // #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < graph->vertices; ++i) {
      cache_node_hashmap[i] = -1;
      // assert(cache_node_hashmap[i] == -1);
    }

    if (graph->config->cache_policy == "sample") {
      LOG_DEBUG("cache_sample");
      cache_sample(cache_node_idx_seq);
    } else if (graph->config->cache_policy == "degree") {  // default cache high degree
      LOG_DEBUG("cache_high_degree");
      cache_high_degree(cache_node_idx_seq);
    } else if (graph->config->cache_policy == "random") {  // default cache high degree
      LOG_DEBUG("cache_random_node");
      cache_random_node(cache_node_idx_seq);
    }
    gater_cpu_cache_feature_and_trans_to_gpu();
  }

    void empty_gpu_cache() {
    for (int ti = 0; ti < 5; ++ti) {  // clear gpu cache memory
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
  }


void count_sample_hop_nodes(Sampler* sampler) {
    long tmp = 0;
    while (sampler->work_offset < sampler->work_range[1]) {
      auto ssg = sampler->subgraph;
      sampler->sample_one(ssg, graph->config->batch_type, ctx->is_train());
      sampler->reverse_sgs();
      for (auto sg : ssg->sampled_sgs) {
        tmp += sg->e_size;
      }
    }
    printf("all batch edges %ld", tmp);
    assert(sampler->work_offset == sampler->work_range[1]);
    sampler->restart();
  }

  // std::pair<int,int>
  float cnt_suit_explicit_block(SampledSubgraph* ssg, VertexId* cache_node_hashmap = nullptr) {
    int node_num_block = 256 * 1024 / 4 / graph->gnnctx->layer_size[0];
    int threshold_node_num_block = node_num_block * graph->config->threshold_trans;
    auto csc_layer = ssg->sampled_sgs[0];
    std::vector<int> need_transfer(graph->vertices, 0);
    for (auto& v : csc_layer->src()) {
      need_transfer[v] = 1;
    }
    std::unordered_map<int, int> freq;
    for (int i = 0; i < graph->vertices; i += node_num_block) {
      int cnt = 0;
      for (int j = i; j < std::min(i + node_num_block, (int)graph->vertices); ++j) {
        if (!cache_node_hashmap) {
          if (need_transfer[j] == 1) cnt++;
        } else {
          if (need_transfer[j] == 1 && cache_node_hashmap[j] == -1) cnt++;
        }
      }
      if (freq.find(cnt) == freq.end()) freq[cnt] = 0;
      freq[cnt]++;
    }

    // check freq record all the src nodes
    int all_cnt = 0, block_cnt = 0;
    for (auto& v : freq) {
      all_cnt += v.first * v.second;
      block_cnt += v.second;
    }
    assert(block_cnt == (graph->vertices + node_num_block - 1) / node_num_block);
    if (!cache_node_hashmap)
      assert(all_cnt == csc_layer->src().size());
    else {
      int tmp = 0;
      for (int j = 0; j < (int)graph->vertices; ++j) {
        if (need_transfer[j] == 1 && cache_node_hashmap[j] == -1) tmp++;
      }
      assert(tmp == all_cnt);
    }

    std::vector<std::pair<int, int>> all(freq.begin(), freq.end());
    sort(all.begin(), all.end(), [&](auto& x, auto& y) { return x.first > y.first; });
    // // check sort is correct
    // for (int i = 1; i < all.size(); ++i) {
    //   assert(all[i - 1].first > all[i].first);
    // }

    int rate_cnt = 0;
    int rate_all = 0;
    for (auto& v : all) {
      if (v.first > threshold_node_num_block) {
        // std::cout << v.first << " " << v.second << std::endl;
        rate_cnt += v.second;
      }
      if (v.first > 0) rate_all += v.second;
    }
    float rate_trans = rate_cnt > 0 ? rate_cnt * 1.0 / rate_all : 0;
    LOG_DEBUG("block %d threshold %d, rate %.2f (%d/%d)", node_num_block, threshold_node_num_block, rate_trans,
              rate_cnt, rate_all);
    // return {rate_cnt, rate_all};
    return rate_trans;
  }


  double* PageRank(Graph<Empty> * graph, int iterations) {
  const double d = (double)0.85;
  double exec_time = 0;
  exec_time -= get_time();

  double * curr = graph->alloc_vertex_array<double>();
  double * next = graph->alloc_vertex_array<double>();
  VertexSubset * active = graph->alloc_vertex_subset();
  active->fill();
  printf("start PageRank!!!");
  double delta = graph->process_vertices<double>(
    [&](VertexId vtx){
      curr[vtx] = (double)1;
      if (graph->out_degree[vtx]>0) {
        curr[vtx] /= graph->out_degree[vtx];
        // printf("vtx %d,curr: %f",vtx,curr[vtx]);
      }
      return (double)1;
    },
    active
  );
  delta /= graph->vertices;
  printf("start Iterations!!!");
  for (int i_i=0;i_i<iterations;i_i++) {
    if (graph->partition_id==0) {
      printf("delta(%d)=%lf\n", i_i, delta);
    }
    graph->fill_vertex_array(next, (double)0);
    graph->process_edges<int,double>(
      [&](VertexId src){
        graph->emit(src, curr[src]);
      },
      [&](VertexId src, double msg, VertexAdjList<Empty> outgoing_adj){
        for (AdjUnit<Empty> * ptr=outgoing_adj.begin;ptr!=outgoing_adj.end;ptr++) {
          VertexId dst = ptr->neighbour;
          write_add(&next[dst], msg);
        }
        return 0;
      },
      [&](VertexId dst, VertexAdjList<Empty> incoming_adj) {
        double sum = 0;
        for (AdjUnit<Empty> * ptr=incoming_adj.begin;ptr!=incoming_adj.end;ptr++) {
          VertexId src = ptr->neighbour;
          sum += curr[src];
        }
        graph->emit(dst, sum);
      },
      [&](VertexId dst, double msg) {
        write_add(&next[dst], msg);
        return 0;
      },
      active
    );
    if (i_i==iterations-1) {
      delta = graph->process_vertices<double>(
        [&](VertexId vtx) {
          next[vtx] = 1 - d + d * next[vtx];
          return 0;
        },
        active
      );
    } else {
      delta = graph->process_vertices<double>(
        [&](VertexId vtx) {
          next[vtx] = 1 - d + d * next[vtx];
          if (graph->out_degree[vtx]>0) {
            next[vtx] /= graph->out_degree[vtx];
            return fabs(next[vtx] - curr[vtx]) * graph->out_degree[vtx];
          }
          return fabs(next[vtx] - curr[vtx]);
        },
        active
      );
    }
    delta /= graph->vertices;
    std::swap(curr, next);
  }

  exec_time += get_time();
  if (graph->partition_id==0) {
    printf("exec_time=%lf(s)\n", exec_time);
  }

  double pr_sum = graph->process_vertices<double>(
    [&](VertexId vtx) {
      return curr[vtx];
    },
    active
  );
  if (graph->partition_id==0) {
    printf("pr_sum=%lf\n", pr_sum);
  }

  graph->gather_vertex_array(curr, 0);
  if (graph->partition_id==0) {
    VertexId max_v_i = 0;
    VertexId min_v_i = 0;
    for (VertexId v_i=0;v_i<graph->vertices;v_i++) {
      if (curr[v_i] > curr[max_v_i]) max_v_i = v_i;
      if(curr[v_i] < curr[min_v_i]) min_v_i = v_i;
    }
    printf("pr[%u]=%lf\n", max_v_i, curr[max_v_i]);
    printf("pr[%u]=%lf\n", min_v_i, curr[min_v_i]);
  }

  // graph->dealloc_vertex_array(curr);
  // graph->dealloc_vertex_array(next);
  delete active;

  return curr;
}

  int* k_core()
  {
    int *finalArry = graph->alloc_vertex_array<int>();
    int *rank = graph->alloc_vertex_array<int>();
    // int res = graph->process_vertices<int>(
    // [&](VertexId vtx){
    //   finalArry[vtx] = 0;
    //   // if (graph->out_degree[vtx]>0) {
    //   //   curr[vtx] /= graph->out_degree[vtx];
    //   //   // printf("vtx %d,curr: %f",vtx,curr[vtx]);
    //   // }
    //   return 1;
    // },
    // active
    // );
    std::vector<pair<int,VertexId*>>tmp_vec;
    // VertexId *column_offset_tmp = new VertexId[graph->vertices + 1];
    // VertexId *row_indices_tmp = new VertexId[graph->edges];
    // memset(column_offset_tmp, 0, sizeof(VertexId) * (graph->vertices + 1));
    // memset(row_indices_tmp, 0, sizeof(VertexId) * graph->edges);
    // memccpy(column_offset_tmp,fully_rep_graph->column_offset,sizeof(VertexId) * (graph->vertices + 1));
    // memccpy(row_indices_tmp,fully_rep_graph->column_offset,sizeof(VertexId) * (graph->edges));
    VertexId *out_degree_tmp = new VertexId[graph->vertices];
    memcpy(out_degree_tmp,graph->out_degree,sizeof(VertexId) * (graph->vertices));
    for(int i=0; i<graph->vertices; i++)
    {
      finalArry[i] = 0;
      rank[i] = 0;
      tmp_vec.push_back(make_pair(i,out_degree_tmp+i));
    }
    stable_sort(tmp_vec.begin(),tmp_vec.end(),cmp_degree);
    for(int i=0 ;i<graph->vertices; i++)
    {
      // printf("out %d\n",graph->out_degree[i]);
      // printf("in  %d\n",graph->in_degree[i]);
      int tmp_nid = tmp_vec[i].first;
      // printf("%d %d\n",tmp_nid,*(tmp_vec[i].second));
      // printf("%d %d\n",tmp_nid,graph->out_degree[tmp_nid]);
      if(finalArry[tmp_nid]!=1)
      {
        int tmp_degree = *(tmp_vec[i].second);
        #pragma omp parallel
        for(int j =fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++)
        {
          int tmp_nid = fully_rep_graph->row_indices[j];
          if(out_degree_tmp[tmp_nid] > tmp_degree)
          {
            out_degree_tmp[tmp_nid]-=1;
            // *(tmp_vec[tmp_nid].second) -= 1; 
            // __sync_fetch_and_sub(out_degree_tmp+tmp_nid,1);
          }
        }
        // printf("%d\n",tmp_degree);
        rank[tmp_nid] = tmp_degree;
        // printf("%d\n",rank[tmp_nid]);
        finalArry[tmp_nid] = 1;
        stable_sort(tmp_vec.begin(),tmp_vec.end(),cmp_degree);
      }
      // fully_rep_graph->row_indices
    }
    // printf("kcore test %d %d",rank[0],rank[1]);
    // for(auto tmp:out_degree_tmp)
    // {
    //   delete *tmp;
    // }
    tmp_vec.clear();
    return rank;
  }

  static bool cmp_degree(const pair<int,VertexId*> &p1,const pair<int,VertexId*> &p2)
  {
    return *(p1.second) < *(p2.second);
  }

  static bool cmp(const pair<int,double> &p1,const pair<int,double> &p2)
  {
    return p1.second > p2.second;
  }

  // show the number of sketch and trian nodes in different classes (supple nodes for fewer classes )
  // this function can be add to the fun sketch_same_with_origin_train_set()
  void get_diff_class_nodes(std::vector<VertexId> &sketch_train_nids)
  {
    LOG_INFO("Before Add sketch nodes nums: %d",sketch_train_nids.size());
    // array
    long *class_arry_origin = new long[graph->gnnctx->label_num];
    long *class_arry_sketch = new long[graph->gnnctx->label_num];
    int *sketch_array = graph->alloc_vertex_array<int>();
    std::vector<std::pair<int,int>>label_nid_vector;
    // VertexId *column_offset_tmp = new VertexId[graph->vertices + 1];
    // VertexId *row_indices_tmp = new VertexId[graph->edges];
    memset(class_arry_origin, 0, sizeof(long) * (graph->gnnctx->label_num ));
    memset(class_arry_sketch, 0, sizeof(long) * (graph->gnnctx->label_num ));
    int sum_train = 0;
    // int sum_sketch = 0;
    // #pragma omp parallel for
    LOG_INFO("test1");
    for(int i =0;i<graph->vertices;i++)
    {
      if(gnndatum->local_mask[i]==0)
      {
        // __sync_fetch_and_add(class_arry_origin+gnndatum->local_label[i],1);
        // __sync_fetch_and_add(&sum_train,1);
        class_arry_origin[gnndatum->local_label[i]]+=1;
        sum_train+=1;
        label_nid_vector.push_back(std::make_pair(gnndatum->local_label[i],i));
        // printf("%d\n",sum_train);
      }
      
      // class_arry_origin[gnndatum->local_label[i]]++;
    }
    LOG_INFO("test2");
    sort(label_nid_vector.begin(),label_nid_vector.end(),[&](auto& x, auto& y) { return x.first < y.first; });
    std::vector<std::vector<int>>result_vector;
    std::vector<int>tmp_vector;
    LOG_INFO("test3");
    for(int i=0; i<label_nid_vector.size(); i++)
    {
      
      if(i==label_nid_vector.size()-1||label_nid_vector[i].first!=label_nid_vector[i+1].first)
      {
        result_vector.push_back(tmp_vector);
        tmp_vector.clear();
      }
      else{
        tmp_vector.push_back(label_nid_vector[i].second);
      }
    }
    LOG_INFO("test4");
    for(int i=0; i<sketch_train_nids.size(); i++)
    {
      int tmp_nid = sketch_train_nids[i];
      sketch_array[tmp_nid] = 1;
      class_arry_sketch[gnndatum->local_label[tmp_nid]]++;
    }
    LOG_INFO("test5");
    float *per_every_class = new float[graph->gnnctx->label_num];
    memset(per_every_class, 0.0, sizeof(float) * (graph->gnnctx->label_num ));
    #pragma omp parallel for
    for(int i = 0;i<graph->gnnctx->label_num;i++)
    {
      if(class_arry_origin[i]==0)
      {
        per_every_class[i]=0;
      }
      else{
        per_every_class[i] = float(class_arry_sketch[i]) / class_arry_origin[i] * 100.0;
      }
      
    }
    LOG_INFO("test6");

    for(int i = 0;i<graph->gnnctx->label_num; i++)
    {
      LOG_INFO("label class: %d origin nums: %d sketch nums: %d per: %.2f",i,class_arry_origin[i],class_arry_sketch[i],per_every_class[i]);
    }
    // float avg_per = float(sketch_train_nids.size()) / sum_train;
    // for(int i =0;i<graph->gnnctx->label_num; i++)
    // {
    //   int avg_num = int(avg_per*class_arry_origin[i]);
    //   sort(result_vector[i].begin(),result_vector[i].end(),[&](auto& x, auto& y) { return graph->out_degree[x] > graph->out_degree[y]; });
    //   if(class_arry_sketch[i]<avg_num)
    //   {
    //     int flag = 0;
    //     if(result_vector[i].size()>avg_num)
    //     {
    //       flag = avg_num-class_arry_sketch[i];
    //     }else{
    //       flag = result_vector[i].size();
    //     }
    //     for(int j=0;j<flag;j++)
    //     {
    //         int tmp_nid = result_vector[i][j];
    //         if(gnndatum->local_mask[tmp_nid]==0&&sketch_array[tmp_nid]!=1)
    //         {
    //           sketch_array[tmp_nid] = 1;
    //           sketch_train_nids.push_back(tmp_nid);
    //         }
    //     }
    //   }
    // }


    // LOG_INFO("After Add sketch nodes nums:%d",sketch_train_nids.size());
    // memset(class_arry_sketch, 0, sizeof(long) * (graph->gnnctx->label_num + 1));
    // memset(per_every_class, 0.0, sizeof(float) * (graph->gnnctx->label_num + 1));
    // for(int i=0; i<sketch_train_nids.size(); i++)
    // {
    //   int tmp_nid = sketch_train_nids[i];
    //   // sketch_array[tmp_nid] = 1;
    //   class_arry_sketch[gnndatum->local_label[tmp_nid]]++;
    // }
    // #pragma omp parallel for
    // for(int i = 0;i<graph->gnnctx->label_num;i++)
    // {
    //   if(class_arry_origin[i]==0)
    //   {
    //     per_every_class[i]=0;
    //   }
    //   else{
    //     per_every_class[i] = float(class_arry_sketch[i]) / class_arry_origin[i] * 100.0;
    //   }
      
    // }
    // for(int i = 0;i<graph->gnnctx->label_num; i++)
    // {
    //   LOG_INFO("label class: %d origin nums: %d sketch nums: %d per: %.2f",i,class_arry_origin[i],class_arry_sketch[i],per_every_class[i]);
    // }
    
    
  }

  // show the classes, neighbors, and other information of sketch nodes
  void show_sketch_label_class(std::vector<VertexId> &sketch_train_nids)
  {
      std::vector<int>neighbor_class;
      int sum_class = 0;
      int diff_class = 0;
      int cur_node_class = 0;
      NtsVar mul_cur_node_class = torch::ones({3,4});
      int cur_degree;
      int same_nodes_num = 0;
      int diff_nodes_num = 0;
      int tmp_class;
      NtsVar mul_tmp_class = torch::ones({3,4});
      float mul_sim_sum = 0.0;
      float diff_per;
      float same_per;
      // float *NCS_score = NCS_test();
    for(auto i:sketch_train_nids)
    {
      neighbor_class.clear();
      mul_sim_sum = 0;
      sum_class = 0;
      diff_class = 0;
      if(graph->config->classes == 1) {
        cur_node_class = gnndatum->local_label[i];
      } else {
        mul_cur_node_class = torch::from_blob(gnndatum->local_label+(i*graph->config->classes),{graph->config->classes});
      }
      
      cur_degree = graph->out_degree[i];
      same_nodes_num = 0;
      diff_nodes_num = 0;
      for(int j =fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++)
      {
        int nid = fully_rep_graph->row_indices[j];
        if(graph->config->classes == 1) {
          tmp_class = gnndatum->local_label[nid];
          if(neighbor_class.size()==0)
          {
            neighbor_class.push_back(tmp_class);
            sum_class++;
            if(tmp_class != cur_node_class)
            {
              diff_class++;
            }
          }else if(neighbor_class.size()==1){
            if(neighbor_class[0]!=tmp_class){
              neighbor_class.push_back(tmp_class);
              sum_class++;
              if(tmp_class != cur_node_class)
              {
                diff_class++;
              }
            }
          } else if(std::find(neighbor_class.begin(), neighbor_class.end(), tmp_class) == neighbor_class.end()){
            neighbor_class.push_back(tmp_class);
            sum_class++;
            if(tmp_class != cur_node_class)
            {
              diff_class++;
            }
          } 
  
          if(tmp_class != cur_node_class){
            diff_nodes_num++;
          }else{
            same_nodes_num++;
          }
        } else {
          mul_tmp_class = torch::from_blob(gnndatum->local_label+(nid*graph->config->classes),{graph->config->classes});
          mul_sim_sum += torch::nn::functional::cosine_similarity(mul_cur_node_class,mul_tmp_class).item<float>();
        }
         


      }
      if(graph->config->classes == 1) {
        diff_per = ((float)diff_nodes_num / cur_degree) *100;
        same_per = 100-diff_per;
      } else {
        same_per = (mul_sim_sum / cur_degree)*100;
        diff_per = 100-same_per;
      }


      LOG_INFO("Sketch nodes:%d degree:%d label class:%d 1-hop class nums:%d diff class:%d same class:%d same per:%f diff per:%f NCS:%f"
      ,i,cur_degree,cur_node_class,neighbor_class.size(),diff_class,neighbor_class.size() - diff_class,same_per,diff_per,0.0);
    }
  }

  // get sketch nodes from different classes of trian set with the same ratio
  void sketch_by_label_rate(float rate,std::vector<VertexId> &sketch_train_nids)
  {
    std::vector<std::pair<int,int>>label_nid_vector;
    for(int i =0;i<graph->vertices;i++)
    {
      if(gnndatum->local_mask[i]==0)
      {
        label_nid_vector.push_back(std::make_pair(gnndatum->local_label[i],i));
      }
    }
    sort(label_nid_vector.begin(),label_nid_vector.end(),[&](auto& x, auto& y) { return x.first < y.first; });
    std::vector<std::vector<int>>result_vector;
    std::vector<int>tmp_vector;
    // LOG_INFO("label: ")
    for(int i=0; i<label_nid_vector.size(); i++)
    {
      
      if(i==label_nid_vector.size()-1||label_nid_vector[i].first!=label_nid_vector[i+1].first)
      {
        if(i==label_nid_vector.size()-1){
          result_vector.push_back(tmp_vector);
          tmp_vector.clear();
          if(result_vector.size()!=graph->gnnctx->label_num)
          {
            // printf("1 %d  %d",result_vector.size(),graph->gnnctx->label_num);
            int tmp = graph->gnnctx->label_num-result_vector.size();
            for(int j=0;j<tmp;j++)
            {
              tmp_vector.clear();
              result_vector.push_back(tmp_vector);
            }
          }
        }else if(label_nid_vector[i].first+1!=label_nid_vector[i+1].first){
          // printf("2 %d %d",label_nid_vector[i].first,label_nid_vector[i+1].first);
          for(int j=label_nid_vector[i].first;j<label_nid_vector[i+1].first;j++)
          {
            
            result_vector.push_back(tmp_vector);
            tmp_vector.clear();
          }
        }else{
          result_vector.push_back(tmp_vector);
          tmp_vector.clear();
        }


        // if( label_nid_vector[i].first+1==label_nid_vector[i+1].first && i!=label_nid_vector.size()-1)
        // {
        //   result_vector.push_back(tmp_vector);
        //   tmp_vector.clear();
        // }else if(i==label_nid_vector.size()-1){

        //   result_vector.push_back(tmp_vector);
        //   tmp_vector.clear();
        //   // printf("%d %d",label_nid_vector[i].first,label_nid_vector[i+1].first)
        // }else{
        //   for(int j=label_nid_vector[i].first;j<label_nid_vector[i+1].first;j++)
        //   {
            
        //     result_vector.push_back(tmp_vector);
        //     tmp_vector.clear();
        //   }
        // }
      }
      else{
        tmp_vector.push_back(label_nid_vector[i].second);
      }
    }
    for(int i =0;i<result_vector.size();i++)
    {
      LOG_INFO("label: %d nodes_num: %d",i,result_vector[i].size());
    }
    for(int i=0; i<graph->gnnctx->label_num ;i++)
    {
      if(result_vector[i].size()!=0){
       sort(result_vector[i].begin(),result_vector[i].end(),[&](auto& x, auto& y) { return graph->out_degree[x] > graph->out_degree[y]; });
       int nodes_num = int(result_vector[i].size()*rate);
       if(nodes_num == 0)
       {
        nodes_num = 1;
       }
       for(int j=0;j<nodes_num;j++)
       {
        sketch_train_nids.push_back(result_vector[i][j]);
       }
      }
      // else{
      //   printf("hhhh");
      // }
    }

  }
  
  // get sketch nodes with the same distribution as the original training set (by degree up or down)
  // verify if the function is valid
  void sketch_same_with_origin_train_set(float rate,std::vector<VertexId> &sketch_train_nids)
  {
    std::map<int,std::vector<int>>class_nodes;
    int train_nums = 0;
    float per = 0.0;
    int sketch_nums = 0;
    for(int i=0;i<graph->vertices;i++)
    {
      if(gnndatum->local_mask[i] == 0){
        train_nums++;
        int tmp_class = gnndatum->local_label[i];
        if(class_nodes.count(tmp_class)==0){
          std::vector<int>tmp_vec;
          tmp_vec.push_back(i);
          class_nodes.insert(std::make_pair(tmp_class,tmp_vec));
        }else{
          class_nodes[tmp_class].push_back(i);
        }
      }
    }
    LOG_INFO("map size: %d",class_nodes.size());

    for(int i=0;i<graph->gnnctx->label_num;i++)
    {
      auto vec = class_nodes[i];
      sort(vec.begin(),vec.end(),[&](auto& x, auto& y) { return graph->out_degree[x] > graph->out_degree[y]; });
      per = vec.size()/float(train_nums);
      sketch_nums = per * rate *train_nums;
      
      if(sketch_nums > vec.size()){
        sketch_nums = vec.size();
      }else if (sketch_nums == 0 && vec.size()!=0)
      {
        sketch_nums = 1;
      }
      
      for(int j=0;j<sketch_nums;j++)
      {
        sketch_train_nids.push_back(vec[j]);
      }
    }
    
  }

  float comp_similarity(int nid)
  {
    NtsVar mul_label = torch::ones({graph->config->classes});
    int cur_label = 0;
    if(graph->config->classes==1) {
      cur_label = gnndatum->local_label[nid];
    } else {
      mul_label = torch::from_blob(gnndatum->local_label+(nid*graph->config->classes),{graph->config->classes});
      std::cout<<mul_label.sizes()<<endl;
    }
    int same_num = 0;
    float sim_sum =0.0;
    int nbr_num = fully_rep_graph->column_offset[nid+1] - fully_rep_graph->column_offset[nid];
    for(int i=fully_rep_graph->column_offset[nid];i<fully_rep_graph->column_offset[nid+1];i++)
    {
      int nbr = fully_rep_graph->row_indices[i];
      if(graph->config->classes==1) {
        if(gnndatum->local_label[nbr]==cur_label)
        {
          same_num++;
        }
      } else {
        NtsVar nbr_label = torch::from_blob(gnndatum->local_label+(nbr*graph->config->classes),{graph->config->classes});
        std::cout<<nbr_label.sizes()<<endl;
        sim_sum +=torch::nn::functional::cosine_similarity(mul_label,nbr_label).item<float>();
        std::cout<<"sim: "<<sim_sum<<endl;
      }

    }
    if(graph->config->classes==1) {
      return float(same_num) / nbr_num;
    } else {
      return sim_sum / nbr_num;
    }
    
  }

  // get sketch nodes by nbr similarity? 
  // todo:vertify the fun (comp a score of sketch set and adjust this score when new nodes are added ?)
  void sketch_by_similarity(std::vector<VertexId> &sketch_train_nids,float low_1, float high_1,float low_2,float high_2)
  {
    // float low_1 = 0.5;
    // float high_1 = 0.6;
    // float low_2 = 0.9;
    // float high_2 = 1.0;
    // || (v.second > low_2 && v.second < high_2)
    printf("L1: %f H1: %f L2: %f H2: %f\n",low_1,high_1,low_2,high_2);
    std::vector<std::pair<int,float>>nid_score;
    float tmp_score = 0.0;
    for(int i=0;i<graph->vertices;i++)
    {
      // && graph->out_degree[i] >= 20
      if(gnndatum->local_mask[i]==0)
      {
        tmp_score = comp_similarity(i);
        nid_score.push_back(make_pair(i,tmp_score));
      }
    }
    sort(nid_score.begin(),nid_score.end(),[&](auto& x, auto& y) { return x.second < y.second; });
    for(const auto &v:nid_score){
      // if(gnndatum->local_label[v.first]==7 ||gnndatum->local_label[v.first]==4 ||gnndatum->local_label[v.first]==13||gnndatum->local_label[v.first]==16|| gnndatum->local_label[v.first]==22 ||gnndatum->local_label[v.first]==24 || gnndatum->local_label[v.first]==25 || gnndatum->local_label[v.first]==27||gnndatum->local_label[v.first]==30 ||gnndatum->local_label[v.first]==32 ||gnndatum->local_label[v.first]==37 ||gnndatum->local_label[v.first]==39)
      // {
      //   if(v.second >=0.0 && v.second <= 1.0){
      //     sketch_train_nids.push_back(v.first);
      //   }
      // } else  || (v.second >= low_2 && v.second <= high_2)
      if((v.second >=low_1 && v.second <= high_1) || (v.second >= low_2 && v.second <= high_2))
      {
        sketch_train_nids.push_back(v.first);
      }else if(v.second > high_2){
        break;
      }
    }
  }

  void sketch_from_file(std::vector<VertexId> &sketch_train_nids) {
    ifstream fin;
    fin.open("/home/yuanh/neutron-sanzo/exp/exp-sketch/ogbn-products-tmp.log",ios::in);
    string buf;
    while(fin >> buf) {
      // printf("%d",int(buf[0]));
      sketch_train_nids.push_back(atoi(buf.c_str()));
      is_sketch[atoi(buf.c_str())] = 1;
    }
    change_CSC_by_sketch();
  }

  void change_CSC_by_sketch() {
    VertexId *coloffset = new VertexId[graph->vertices+1];
    VertexId *rowindex = new VertexId[graph->edges];
    memset(coloffset,0,sizeof(VertexId)*(graph->vertices+1));
    memset(rowindex,0,sizeof(VertexId)*(graph->edges));
    int nodes_num = 0;
    int edge_nums = 0;
    int tmp = 0;
    for(int i=0;i<graph->vertices;i++) {
      int tmp_num = 0;
      if(gnndatum->local_mask[i]==0) {
        if(is_sketch[i]==1) {
          for(int j=fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++) {
            if(gnndatum->local_mask[fully_rep_graph->row_indices[j]]!=0 || is_sketch[fully_rep_graph->row_indices[j]]==1 ) {
              rowindex[edge_nums++] = fully_rep_graph->row_indices[j];
              tmp_num ++;
            }
          }
          coloffset[i+1] = coloffset[i] + tmp_num;
          nodes_num += 1;
          // memcpy(rowindex+edge_nums,fully_rep_graph->row_indices+fully_rep_graph->column_offset[i],fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i]);
          // edge_nums += fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i];
        } else {
          coloffset[i+1] = coloffset[i];
          tmp++;
        }
      } else {
        for(int j=fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++) {
              if(gnndatum->local_mask[fully_rep_graph->row_indices[j]]!=0 || is_sketch[fully_rep_graph->row_indices[j]]==1 ) {
                rowindex[edge_nums++] = fully_rep_graph->row_indices[j];
                tmp_num ++;
              }
            }
        coloffset[i+1] = coloffset[i] + tmp_num;
        nodes_num += 1;
      }
    }
    printf("node nums: %d %d origin: %d tmp: %d\n",nodes_num,edge_nums,fully_rep_graph->global_edges,tmp);
    delete fully_rep_graph->column_offset;
    delete fully_rep_graph->row_indices;
    fully_rep_graph->column_offset = coloffset;
    fully_rep_graph->row_indices = rowindex;
  }

  double comp_cos_Similarity(float *va,float *vb) {
    double cossu = 0.0;
    double cossda = 0.0;
    double cossdb = 0.0;

    // for (int i = 0; i < graph->gnnctx->layer_size[0]; i++) {
    //   printf("%f ",vb[i]);
    // }
    // printf("\n");

    for (int i = 0; i < graph->gnnctx->layer_size[0]; i++) {
        cossu += va[i] * vb[i];
        cossda += va[i] * va[i];
        cossdb += vb[i] * vb[i];
    }
    // printf("%lf %lf %lf\n",cossu,cossda,cossdb);
    if(cossdb == 0.0) {
      return 0.0;
    } else {
      return cossu / (sqrt(cossda) * sqrt(cossdb));
    }
    
  }


  float tmp_comp_nbr_sim(VertexId tmp,VertexId nid) {
    int cur_class = gnndatum->local_label[nid];
    int nbr_num = fully_rep_graph->column_offset[nid+1] - fully_rep_graph->column_offset[nid];
    int same = 0;
    for(int i=fully_rep_graph->column_offset[nid];i<fully_rep_graph->column_offset[nid+1];i++) {
      VertexId nbr_nid = fully_rep_graph->row_indices[i];
      if(cur_class == gnndatum->local_label[nbr_nid]) {
        same++;
      }
    }
    return float(same)/nbr_num;
  }

  double* get_diff_class_cos_similarity(std::vector<VertexId> &sketch_train_nids) {
    // for(int i=0;i<graph->vertices;i++)
    for(auto i:sketch_train_nids)
    {
      // if(gnndatum->local_mask[i] == 0){
        int tmp_class = gnndatum->local_label[i];
        if(class_nodes.count(tmp_class)==0){
          std::vector<int>tmp_vec;
          tmp_vec.push_back(i);
          class_nodes.insert(std::make_pair(tmp_class,tmp_vec));
        }else{
          class_nodes[tmp_class].push_back(i);
        }
      // }
    }
    // LOG_INFO("map size: %d",class_nodes.size());
    int x = graph->gnnctx->label_num * graph->gnnctx->label_num;
    double *cosS = (double *)malloc(sizeof(double)*x);
    double *innerSim = (double *)malloc(sizeof(double)*graph->gnnctx->label_num);
    // int *class_num = new int[graph->gnnctx->label_num];
    // double cosS[1600] = {0.0};
    float *mean_feat = (float *)malloc(sizeof(float)*graph->gnnctx->label_num*graph->gnnctx->layer_size[0]);
    // float tmp_feat[40*128] = {0.0};
    int feat_size = graph->gnnctx->layer_size[0];
    int class_num = graph->gnnctx->label_num;
    for(auto it:class_nodes) {
      auto label = it.first;
      auto vec = it.second;
      for(int j=0;j<feat_size;j++)
      {
        float tmp = 0.0;
        for(int i=0;i<vec.size();i++)
        {
          tmp +=gnndatum->local_feature[vec[i]*feat_size+j];
        }
        mean_feat[label*feat_size+j] = tmp/vec.size();
      }  
    }
    // mean_feat = tmp_feat;
    for(int i=0;i<graph->gnnctx->label_num-1;i++) {
      cosS[i*class_num+i] = 1.0;
      for(int j=i+1;j<graph->gnnctx->label_num;j++) {
        cosS[i*class_num+j] = comp_cos_Similarity(mean_feat+feat_size*i,mean_feat+feat_size*j);
      }
    }

    for(auto it:class_nodes) {
      auto label = it.first;
      auto vec = it.second;
      double sum = 0.0;
      for(int i=0;i<vec.size();i++) {
        sum += comp_cos_Similarity(mean_feat+feat_size*label,gnndatum->local_feature+feat_size*vec[i]);
      }
      innerSim[label] = sum / vec.size();
    }

   for(int i=0;i<graph->gnnctx->label_num;i++) {
      for(int j=0;j<graph->gnnctx->label_num;j++) {
        if(j < i) {
          printf("0 ");
        }else {
          printf("%lf ",cosS[i*class_num+j]);
        }
        
      }
      printf("\n");
    }

  //   for(int i=0;i<graph->gnnctx->label_num;i++) {
  //     printf("class: %d inner-Sim: %lf \n",i,innerSim[i]);
  //   }

  //   double sums = 0.0;
  //  for(int i=0;i<graph->gnnctx->label_num-1;i++) {
  //     for(int j=i;j<graph->gnnctx->label_num;j++) {
  //       sums += cosS[i*class_num+j];
  //     }
  //   }
  //   double sim_inter = sums/(2*(class_num*class_num-class_num));
  //   LOG_INFO("sim inter: %lf",sim_inter);

  //     for(int i=0;i<graph->vertices;i++) {
        
  //         // printf("test before sim\n");
  //         std::vector<std::pair<int,float>>tmp_vec;
  //         for(int j=fully_rep_graph->column_offset[i],index=0;j<fully_rep_graph->column_offset[i+1];j++,index++) {
  //           VertexId nbr = fully_rep_graph->row_indices[j];
  //           // printf("test in sim\n");
  //           // if(gnndatum->local_mask[nbr]==0 && is_sketch[nbr]==0) {
  //           //   fully_rep_graph->sim_value[j] = 0;
  //           // } else {
  //           //   fully_rep_graph->sim_value[j] = comp_cos_Similarity(gnndatum->local_feature+(graph->gnnctx->layer_size[0]*i),gnndatum->local_feature+(graph->gnnctx->layer_size[0]*nbr));
  //           // mean_feat+(graph->gnnctx->layer_size[0]*gnndatum->local_label[i])
  //           // }
  //           // fully_rep_graph->sim_value[j] = comp_cos_Similarity(gnndatum->local_feature+(graph->gnnctx->layer_size[0]*i),gnndatum->local_feature+(graph->gnnctx->layer_size[0]*nbr));            
  //           fully_rep_graph->sim_value[j] = tmp_comp_nbr_sim(i,nbr);
  //           // 搞个vector，将每个顶点邻居按相似度降序存储，采样时按邻居相似度采样！
  //           tmp_vec.push_back(std::make_pair(index,fully_rep_graph->sim_value[j]));
  //           // printf("test after sim\n");
  //         }
  //         sort(tmp_vec.begin(), tmp_vec.end(), [&](auto& x, auto& y) { return x.second > y.second; });
  //         nid_nbr.push_back(tmp_vec);

  //     }    

    return cosS;
  }

  float comp_2NCS(VertexId nid) {
    std::vector<VertexId>nbr_list;
    int cur_label =  gnndatum->local_label[nid];
    int one_nbr_num = fully_rep_graph->column_offset[nid+1] - fully_rep_graph->column_offset[nid];
    int two_hop_nbr_num = 0;
    int two_hop_nbr_nid = 0;
    int two_hop_same_num = 0;
    float NCS_result = 0.0;
    for(int i=fully_rep_graph->column_offset[nid];i<fully_rep_graph->column_offset[nid+1];i++)
    {
      int nbr = fully_rep_graph->row_indices[i];
      nbr_list.push_back(nbr);
    }
    float tmp = 0.0;
// #pragma omp parallel for
    for(int i=0;i<nbr_list.size();i++) {
      tmp = 0.0;
      two_hop_same_num = 0;
      if(nbr_list[i] != nid) {
        two_hop_nbr_num = fully_rep_graph->column_offset[nbr_list[i]+1] - fully_rep_graph->column_offset[nbr_list[i]];
        for(int j=fully_rep_graph->column_offset[nbr_list[i]];j<fully_rep_graph->column_offset[nbr_list[i]+1];j++) {
          two_hop_nbr_nid = fully_rep_graph->row_indices[j];
          if(two_hop_nbr_nid != nid && gnndatum->local_label[two_hop_nbr_nid] == cur_label) {
            two_hop_same_num++;
          }
        }
        tmp = two_hop_same_num / (two_hop_nbr_num-1);
        NCS_result += tmp;
      }
      
    }
    return NCS_result / one_nbr_num;
  }

  float* NCS_test() {
    float *NCS_score = new float[graph->vertices]{};
    if(graph->config->classes == 1) {
      for(int i=0;i<graph->vertices;i++){
        NCS_score[i] = comp_2NCS(i);
      }
    } 
    return NCS_score;
  }

  void get_sketch_nodes(std::vector<VertexId> &sketch_train_nids,int sketch_mode)
  {
    // && graph->in_degree[i]<=1000
    if (sketch_mode == 0){
      sketch_train_nids.clear();
      // printf("%d",gnndatum->local_mask.len);
      for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
        int type = gnndatum->local_mask[i];
        if (type == 0) {
          if(graph->in_degree[i]>graph->config->target_degree && graph->in_degree[i]<=1500)
            sketch_train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
        }
      }
      get_diff_class_cos_similarity(sketch_train_nids);
      get_diff_class_nodes(sketch_train_nids);
      // show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 1){
      sketch_train_nids.clear();
      int i = 0;
      while(i<graph->config->nodes_num)
      {
        int tmp_nid = rand_int(graph->vertices);
        if (gnndatum->local_mask[tmp_nid] == 0)
        {
          sketch_train_nids.push_back(tmp_nid + graph->partition_offset[graph->partition_id]);
          i+=1;
        }
      }
      get_diff_class_nodes(sketch_train_nids);
      show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 2){
      double *score = PageRank(graph,5);
      // std::map<int,double>tmp;
      std::vector<pair<int,double>>tmp_vec;
      for(int i = 0; i<graph->vertices; i++)
      {
        tmp_vec.push_back(std::make_pair(i,score[i]));
      }
      sort(tmp_vec.begin(),tmp_vec.end(),cmp);
      int i = 0;
      int j = 0;
      while(i<graph->config->nodes_num){
        int tmp_nid = tmp_vec[j].first;
        if (gnndatum->local_mask[tmp_nid] == 0)
        {
          sketch_train_nids.push_back(tmp_nid + graph->partition_offset[graph->partition_id]);
          i+=1;
          j+=1;
        }
        else {
          j+=1;
        }
      }
      get_diff_class_nodes(sketch_train_nids);
      show_sketch_label_class(sketch_train_nids);
      // printf("%d,%lf",tmp_vec[0].first,tmp_vec[0].second);
    } else if(sketch_mode == 3){
      int *rank = graph->alloc_vertex_array<int>();
      rank = k_core();
      for(int i=0; i<graph->vertices; i++)
      {
        int core_rank = rank[i];
        LOG_INFO("nid: %d label: %d mask: %d degree: %d score: %d",i,gnndatum->local_label[i],gnndatum->local_mask[i],graph->out_degree[i],rank[i]);

        // if(core_rank >= graph->config->nodes_num && gnndatum->local_mask[i] == 0)
        // {
        //   printf("%d\n",i);
        //   sketch_train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
        // }
      }
      // get_diff_class_nodes(sketch_train_nids);
      // show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 4){
      sketch_by_label_rate(0.05,sketch_train_nids);
      // printf("123");
      get_diff_class_nodes(sketch_train_nids);
      // printf("567");
      show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 5){
      sketch_same_with_origin_train_set(0.2,sketch_train_nids);
      get_diff_class_nodes(sketch_train_nids);
      show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 6){
      printf("get sketch nodes! %d\n",graph->config->classes);

      sketch_by_similarity(sketch_train_nids,graph->config->low_1,graph->config->high_1,graph->config->low_2,graph->config->high_2);
      double *cosS = get_diff_class_cos_similarity(sketch_train_nids);
      // get_diff_class_nodes(sketch_train_nids);
      // show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 7){
      sketch_from_file(sketch_train_nids);
      get_diff_class_cos_similarity(sketch_train_nids);
      if(graph->config->run_mode=="test") {
        get_class_score();
      }
      
      get_diff_class_nodes(sketch_train_nids);
      // show_sketch_label_class(sketch_train_nids);
    } else if(sketch_mode == 8){
      // float *NCS_socre = NCS_test(sketch_train_nids);
      double *cosS = get_diff_class_cos_similarity(sketch_train_nids);
      std::map<int,int>class_nid_nums;
      std::vector<std::pair<int,float>>class_nid_per;
      int cur_class = 0;
      int nbr_num = 0;
      int flag = 0;
      for(int i=0; i<graph->vertices; i++)
      {
        class_nid_nums.clear();
        class_nid_per.clear();
        if(gnndatum->local_mask[i] == 0){
          flag = 0;
          cur_class = gnndatum->local_label[i];
          nbr_num = fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i];
          for(int j=fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++) {
            int nbr_nid = fully_rep_graph->row_indices[j];
            int nbr_class = gnndatum->local_label[nbr_nid];
            if(class_nid_nums.count(nbr_class)==0) {
              class_nid_nums.insert(std::make_pair(nbr_class,1));
            } else {
              class_nid_nums[nbr_class]++;
            }
          }
          for(int j=0;j<graph->gnnctx->label_num;j++) {
            class_nid_per.push_back(std::make_pair(j,float(class_nid_nums[j])/nbr_num));
          }
          sort(class_nid_per.begin(), class_nid_per.end(), [&](auto& x, auto& y) { return x.second > y.second; });
          int tmp1 = 0;
          int tmp2 = 0;
          for(int j=0;j<3;j++) {
            if(class_nid_per[j].first != cur_class && class_nid_per[j].second >= 0.3) {
              if(class_nid_per[j].first>cur_class) {
                tmp1 = cur_class;
                tmp2 = class_nid_per[j].first;
              } else if (class_nid_per[j].first<cur_class){
                tmp1 = class_nid_per[j].first;
                tmp2 = cur_class;
              }
              if(fabs(cosS[tmp1*graph->gnnctx->label_num+tmp2])>0.2){
                flag =1;
                break;
              }
            }
          }
          if(flag == 0) {
            sketch_train_nids.push_back(i);
            is_sketch[i] = 1;
          }
        }
      }
      change_CSC_by_sketch();
      // get_diff_class_nodes(sketch_train_nids);
      // show_sketch_label_class(sketch_train_nids);
    }


    LOG_INFO("Sketch Mode: (%d) Sketch train nodes: (%d) (%.3f)",
              sketch_mode,
              sketch_train_nids.size(),
              sketch_train_nids.size() * 1.0 / graph->vertices);
  }

  void get_class_score() {
    float *score = NCS_test();
    // float score[100] = {0};
    printf("test\n");
    for(int i=0;i<graph->vertices;i++)
    {
      if(gnndatum->local_mask[i] == 0){
        int tmp_class = gnndatum->local_label[i];
        if(class_nodes.count(tmp_class)==0){
          std::vector<int>tmp_vec;
          tmp_vec.push_back(i);
          class_nodes.insert(std::make_pair(tmp_class,tmp_vec));
        }else{
          class_nodes[tmp_class].push_back(i);
        }
      }
    }
    LOG_INFO("map size: %d",class_nodes.size());

    for(int i=0;i<graph->gnnctx->label_num;i++)
    {
      auto vec = class_nodes[i];
      sort(vec.begin(),vec.end(),[&](auto& x, auto& y) { return score[x] > score[y]; });
      LOG_INFO("class: %d nodes num: %d",i,class_nodes[i].size());
    }
    
    // for(int i=0;i<graph->gnnctx->label_num;i++) {
    //   LOG_INFO("class: %d nodes num: %d",i,class_score[i].size());
    // }
  }

  int* supply_sketch(std::vector<VertexId> &sketch_train_nids,float ratio,int *pre) {
    // return 0;
    for(int i=0;i<graph->gnnctx->label_num;i++) {
      int add_nums = int(ratio*class_nodes[i].size());
      for(int j=0;j<add_nums;j++)
      {
        if(j+pre[i]>class_nodes[i].size()){
          break;
        } else {
          int nid = class_nodes[i][j+pre[i]];
          if(is_sketch[nid]==0) {
            sketch_train_nids.push_back(nid);
            is_sketch[nid] = 1;
            pre[i] +=1;
          }
        }

      }
      // pre[i] += add_nums;
    }
    return pre;
  }

  float run() {
    double pre_time = -get_time();
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    std::vector<VertexId> sketch_train_nids;
    double time_sketch = 0;
    time_sketch -= get_time();
    get_sketch_nodes(sketch_train_nids,graph->config->sketch_mode);
    time_sketch += get_time();
    LOG_INFO("Sketch Time: %f",time_sketch);
    BatchType batch_type = graph->config->batch_type;
    best_val_acc == 0;
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      int type = gnndatum->local_mask[i];
      if (type == 0) {
        train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 1) {
        val_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      } else if (type == 2) {
        test_nids.push_back(i + graph->partition_offset[graph->partition_id]);
      }
    }

    LOG_INFO("label rate: %.3f, train/val/test: (%d/%d/%d) (%.3f/%.3f/%.3f)",
             1.0 * (train_nids.size() + val_nids.size() + test_nids.size()) / graph->vertices, train_nids.size(),
             val_nids.size(), test_nids.size(), train_nids.size() * 1.0 / graph->vertices,
             val_nids.size() * 1.0 / graph->vertices, test_nids.size() * 1.0 / graph->vertices);

    cuda_stream_list = new Cuda_Stream[pipelines];
    auto default_stream = at::cuda::getDefaultCUDAStream();
    for (int i = 0; i < pipelines; i++) {
      torch_stream.push_back(at::cuda::getStreamFromPool(true));
      auto stream = torch_stream[i].stream();
      cuda_stream_list[i].setNewStream(stream);
      for (int j = 0; j < i; j++) {
        if (cuda_stream_list[j].stream == stream || stream == default_stream) {
          LOG_DEBUG("stream i:%p is repeat with j: %p, default: %p\n", stream, cuda_stream_list[j].stream,
                    default_stream);
          exit(3);
        }
      }
    }

    train_sampler = new Sampler(fully_rep_graph, train_nids, pipelines, false);
    train_sampler->show_fanout("train sampler");
    // eval_sampler = new Sampler(fully_rep_graph, val_nids, true);  // true mean full batch
    eval_sampler = new Sampler(fully_rep_graph, val_nids);  // true mean full batch
    if (graph->config->val_batch_size == 0) graph->config->val_batch_size = graph->config->batch_size;
    eval_sampler->update_batch_size(graph->config->val_batch_size);  // true mean full batch
    // eval_sampler->update_fanout(-1);        ÷            // val not sample
    eval_sampler->update_fanout(graph->gnnctx->val_fanout);  // val not sample
    std::cout << train_nids.size() << " " << test_nids.size() << std::endl;
    test_sampler = new Sampler(fully_rep_graph, test_nids);  // true mean full batch
    test_sampler->update_batch_size(graph->config->val_batch_size);
    test_sampler->update_fanout(graph->gnnctx->val_fanout);
    std::cout << "train_fanout:" << std::endl;
    train_sampler->show_fanout("train sampler");
    train_sampler->subgraph->show_fanout("train subgraph sampler");
    std::cout << "eval_fanout:" << std::endl;
    eval_sampler->show_fanout("val sampler");
    eval_sampler->subgraph->show_fanout("val subgraph sampler");
    std::cout << "test_fanout:" << std::endl;
    test_sampler->show_fanout("test sampler");
    test_sampler->subgraph->show_fanout("test subgraph sampler");


    pre_time += get_time();

    float config_run_time = graph->config->run_time;
    if (config_run_time > 0) {
      start_time = get_time();
      iterations = INT_MAX;
      LOG_DEBUG("iterations %d config_run_time %.3f", iterations, config_run_time);
    }

    double cache_init_time = -get_time();
    if (graph->config->cache_type == "gpu_memory") {
      // LOG_DEBUG("start get_gpu_idle_mem()");
      // double max_gpu_mem = get_gpu_idle_mem();
      LOG_DEBUG("start get_gpu_idle_mem_pipe()");
      double max_gpu_mem = get_gpu_idle_mem_pipe();
      LOG_DEBUG("release gpu memory");
      empty_gpu_cache();
      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      LOG_DEBUG("used %.3f total %.3f (after emptyCache)", used_gpu_mem, total_gpu_mem);
      // double free_memory = total_gpu_mem - max_gpu_mem - 200;
      double free_memory = total_gpu_mem - max_gpu_mem - 100;
      int memory_nodes = free_memory * 1024 * 1024 / sizeof(ValueType) / graph->gnnctx->layer_size[0];
      determine_cache_node_idx(memory_nodes);
      get_gpu_mem(used_gpu_mem, total_gpu_mem);
      LOG_DEBUG("used %.3f total %.3f (after cache feature)", used_gpu_mem, total_gpu_mem);
    } else if (graph->config->cache_type == "rate") {
      assert(graph->config->cache_rate >= 0 && graph->config->cache_rate <= 1);
      determine_cache_node_idx(graph->vertices * graph->config->cache_rate);
    } else if (graph->config->cache_type == "random") {
      assert(graph->config->cache_rate >= 0 && graph->config->cache_rate <= 1);
      determine_cache_node_idx(graph->vertices * graph->config->cache_rate);
    } else if (graph->config->cache_type == "none") {
      LOG_DEBUG("There is no cache_type!");
    } else {
      std::cout << "cache_type: " << graph->config->cache_type << " is not support!" << std::endl;
      assert(false);
    }
    cache_init_time += get_time();


    gcn_run_time = 0;
    // NtsVar tmp0;
    // NtsVar tmp1;
    // NtsVar res0;
    // NtsVar res1;
    VertexId run_flag = 0;
    VertexId nums = 0;
    // float best_val_acc = 0.0;
    std::string suffix = "-sketch.pt";
    // ValueType best_acc = 0.0;
    VertexId pre_flag = 0;
    int *pre = new int[graph->gnnctx->label_num]();
    Sampler* train_sampler_full = new Sampler(fully_rep_graph, train_nids, pipelines, false);
    Sampler* train_sampler_sketch = new Sampler(fully_rep_graph, sketch_train_nids, pipelines, false);
    train_sampler = train_sampler_sketch;
    double sum_run_time = 0;
    // sum_run_time -= get_time();
    for (int i_i = 0; i_i < iterations; i_i++) {

      if(graph->config->run_mode == "turn") {
        if(graph->config->degree_switch!=-1 ) {
          if((i_i+1)%graph->config->degree_switch==0){
            // train_sampler = new Sampler(fully_rep_graph, train_nids, pipelines, false);
            // pre = supply_sketch(sketch_train_nids,0.1,pre);
            train_sampler_sketch = new Sampler(fully_rep_graph, sketch_train_nids, pipelines, false);
            train_sampler = train_sampler_full;
            LOG_INFO("Epoch:(%d) Full Train!!!! nodes num: (%d)",i_i,train_nids.size());
          }
          else if((i_i+1)%graph->config->degree_switch==1){
            // train_sampler = new Sampler(fully_rep_graph, sketch_train_nids, pipelines, false);

            train_sampler = train_sampler_sketch;
            LOG_INFO("Epoch:(%d) Sketch Train!!!! nodes num: (%d)",i_i,sketch_train_nids.size());
          }
        }
      } else if(graph->config->run_mode == "best") {
        // printf("hhhtest\n");
        switch (run_flag)
        {
        case 0:
          train_sampler = train_sampler_sketch;
          LOG_INFO("Epoch:(%d) Sketch Train!!!! nodes num: (%d)",i_i,sketch_train_nids.size());
          break;
        case 1:
          // pre = supply_sketch(sketch_train_nids,0.1,pre);
          // train_sampler_sketch = new Sampler(fully_rep_graph, sketch_train_nids, pipelines, false);
          train_sampler = train_sampler_full;
          if(graph->config->best_parameter > 0) {
            loadW(suffix);
            LOG_INFO("loadW!!!");
          }
          pre_flag = run_flag;
          run_flag = 2;
          LOG_INFO("Epoch:(%d) Full Train!!!! nodes num: (%d)",i_i,train_nids.size());
          break;
        
        default:
          break;
        }
      } else if(graph->config->run_mode == "test") {
        if((i_i+1)%graph->config->degree_switch==0) {
          if(sketch_train_nids.size()<120000) {
            supply_sketch(sketch_train_nids,0.15,pre);
            LOG_INFO("Epoch:(%d) Sketch Train!!!! nodes num: (%d)",i_i,sketch_train_nids.size());
            // get_diff_class_nodes(sketch_train_nids);
            train_sampler = new Sampler(fully_rep_graph, sketch_train_nids, pipelines, false);
          }
         
        }
      } 
      else if(graph->config->run_mode == "none") {
        printf("run mode error!!!\n");
        LOG_DEBUG("No select run mode!!!");
        assert(false);
      }


      // train_sampler = new Sampler(fully_rep_graph, train_nids, pipelines, false);
      // LOG_INFO("sample_workrange:%d",train_sampler->work_range[1]);
      // tmp1 = P[layers-1]->W.clone();
      // tmp0 = P[0]->W.clone();
      if (config_run_time > 0 && gcn_run_time >= config_run_time) {
        iterations = i_i;
        break;
      }
      graph->rtminfo->epoch = i_i;

      // update batch size should before Forward()
      // if (graph->config->batch_switch_time > 0) {
      //   bool ret = train_sampler->update_batch_size_from_time(gcn_run_time);
      //   if (ret) eval_sampler->update_batch_size(train_sampler->batch_size);


      //   // load best parameter
      //   if (ret && graph->config->best_parameter > 0) {
      //     std::string suffix = "";
      //     if (graph->config->sample_rate > 0) { // fanout
      //       std::string tmp = std::to_string(graph->config->sample_rate);
      //       tmp = tmp.substr(0, tmp.find(".") + 4 + 1);
      //       suffix = "-" + std::to_string(graph->config->batch_size) + "sample-rate-" + tmp + ".pt";
      //     } else {
      //       suffix = "-" + std::to_string(graph->config->batch_size) + "fanout-" + graph->config->fanout_string + ".pt";
      //     }
      //     loadW(suffix);
      //     // double tmp_val_acc = EvalForward(eval_sampler, 1);
      //     // LOG_DEBUG("after loadW val_acc %.3f best_val_acc %.3f", tmp_val_acc, best_val_acc);
      //   }
      // }

      //  if (graph->config->fanout_switch_time > 0) {
      //   bool ret = train_sampler->update_fanout_from_time(gcn_run_time);
      //   if (ret) {
      //     // eval_sampler->update_fanout(train_sampler->fanout);
          
      //     std::cout << "train_fanout:" << std::endl;
      //     train_sampler->show_fanout("train sampler");
      //     train_sampler->subgraph->show_fanout("train subgraph sampler");
      //     std::cout << "eval_fanout:" << std::endl;
      //     eval_sampler->show_fanout("val sampler");
      //     eval_sampler->subgraph->show_fanout("val subgraph sampler");
      //   }
      // }

      ctx->train();
      auto [train_acc, epoch_train_time] = Forward(train_sampler, 0);
      float train_loss = loss_epoch;
      // std::cout << "forward done" << std::endl;

      // float val_loss = 0.0;
      // float val_acc = 0.0;
      // double val_train_cost = 0.0;
      // if((i_i+1) % 50 ==0) {
      //   ctx->eval();
      //   val_train_cost = -get_time();
      //   val_acc = EvalForward(test_sampler, 1);
      //   val_train_cost += get_time();
      //   // float val_acc_2 = EvalForward_2(eval_sampler, 1);
      //   val_loss = loss_epoch;        
      // }

      ctx->eval();
      double val_train_cost = -get_time();
      float val_acc = EvalForward(eval_sampler, 1);
      val_train_cost += get_time();
      // float val_acc_2 = EvalForward_2(eval_sampler, 1);
      float val_loss = loss_epoch;
      
      // res1 = ((P[layers-1]->W - tmp1).flatten()).to(torch::kCPU);
      // res0 = ((P[0]->W - tmp0).flatten()).to(torch::kCPU);
      // float *c = res1.data_ptr<float>();
      // float *b = res0.data_ptr<float>();
      // float sum0 = 0;
      // float sum1 = 0;
      // for(int m = 0;m<128*40;m++)
      // {
      //   if(c[m]<0)
      //   {
      //     sum1 = sum1 - c[m];
      //   }
      //   else{
      //     sum1 = sum1 + c[m];
      //   }
          
      // }

      // for(int m = 0;m<128*128;m++)
      // {
      //   if(b[m]<0)
      //   {
      //     sum0 = sum0 - b[m];
      //   }
      //   else{
      //     sum0 = sum0 + b[m];
      //   }
      // }
      // printf("diff layer 0:%f\n",sum0);
      // printf("diff layer 1:%f\n",sum1);

      if (graph->partition_id == 0) {
        LOG_INFO(
            "Epoch %03d train_loss %.3f train_acc %.3f val_loss %.3f val_acc %.3f (train_time %.3f train_sample_time %.3f val_time %.3f, "
            "gcn_run_time %.3f) batch_size (%d, %d) train_fanout (%d,%d)",
            i_i, train_loss, train_acc, val_loss, val_acc, epoch_train_time, gcn_sample_time,val_train_cost, gcn_run_time, train_sampler->batch_size, eval_sampler->batch_size, train_sampler->fanout[0],train_sampler->fanout[1]);
      }
      sum_run_time  = sum_run_time + val_train_cost + epoch_train_time + gcn_sample_time;
      LOG_INFO("All Run Time: %f",sum_run_time);
      // if (graph->partition_id == 0) {
      //   LOG_INFO(
      //       "Epoch %03d train_loss %.3f train_acc %.3f val_loss %.3f val_acc_2 %.3f (train_time %.3f train_sample_time %.3f val_time %.3f, "
      //       "gcn_run_time %.3f) batch_size (%d, %d) train_fanout (%d,%d)",
      //       i_i, train_loss, train_acc, val_loss, val_acc_2, epoch_train_time, gcn_sample_time,val_train_cost, gcn_run_time, train_sampler->batch_size, eval_sampler->batch_size, train_sampler->fanout[0],train_sampler->fanout[1]);
      // }



      if(graph->config->run_mode == "best") {
        if (run_flag == 2) {

          best_val_acc = val_acc;
          nums = 0;
          pre_flag = run_flag;
          run_flag = 0;

        } else if (pre_flag == 2) {
          best_val_acc = val_acc;
          pre_flag = 0;
        } else if (val_acc > best_val_acc) {
          best_val_acc = val_acc;
          nums = 0;
          // save best parameter
          if (graph->config->best_parameter > 0) {
            // std::string suffix = "-sketch.pt";
            // if (graph->config->sample_rate > 0) { // fanout
            //   std::string tmp = std::to_string(graph->config->sample_rate);
            //   tmp = tmp.substr(0, tmp.find(".") + 4 + 1);
            //   suffix = "-" + std::to_string(graph->config->batch_size) + "sample-rate-" + tmp + ".pt";
            // } else {
            //   suffix = "-" + std::to_string(graph->config->batch_size) + "fanout-" + graph->config->fanout_string + ".pt";
            // }
            saveW(suffix);
            LOG_INFO("saveW: best_val_acc %.3f", best_val_acc);
            // loadW();
            // double tmp_val_acc = EvalForward(eval_sampler, 1);
            // LOG_DEBUG("after loadW val_acc %.3f best_val_acc %.3f", tmp_val_acc, best_val_acc);
            // exit(-1);
          }
        } else {
          nums += 1;
          if(nums == 4) {
            pre_flag = run_flag;
            run_flag = 1;
          }
        }
      } else if(graph->config->run_mode == "turn") {
        if(val_acc > best_val_acc) {
          best_val_acc = val_acc;
          nums = 0;
        } else {
          nums++;
          if(nums == 10) {
            float val_acc_2 = EvalForward_2(eval_sampler, 1);
            break;
            }
        }
      }


      // if (graph->config->sample_switch_time > 0) {
      //   bool ret = train_sampler->update_sample_rate_from_time(gcn_run_time);
      //   if (ret) eval_sampler->update_batch_size(train_sampler->batch_size);
      // }
      
      // if (graph->config->batch_switch_acc > 0) {
      //   bool ret = train_sampler->update_batch_size_from_acc(i_i, val_acc, gcn_run_time);
      //   // printf("train_bs: %d",train_sampler->batch_size);
      //   if (ret) eval_sampler->update_batch_size(train_sampler->batch_size);
      // }

      // // LOG_INFO("fanout_switch_acc:%d",graph->config->fanout_switch_acc);
      // if (graph->config->fanout_switch_acc > 0) {

      //   bool ret = train_sampler->update_fanout_from_acc(i_i, val_acc, gcn_run_time);
      //   // printf("train_bs: %d",train_sampler->batch_size);
      //   if (ret) {
      //     // eval_sampler->update_fanout(train_sampler->fanout);

      //     std::cout << "train_fanout:" << std::endl;
      //     train_sampler->show_fanout("train sampler");
      //     train_sampler->subgraph->show_fanout("train subgraph sampler");
      //     std::cout << "eval_fanout:" << std::endl;
      //     eval_sampler->show_fanout("val sampler");
      //     eval_sampler->subgraph->show_fanout("val subgraph sampler");
      //   }
      // }
    }


    delete active;
    return best_val_acc;
  }
};