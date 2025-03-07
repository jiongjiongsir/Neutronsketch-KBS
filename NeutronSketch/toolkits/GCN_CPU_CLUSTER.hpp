#include "core/neutronstar.hpp"
#include "core/ntsPeerRPC.hpp"
#include "utils/torch_func.hpp"
#include "utils/cuda_memory.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
class GCN_CPU_CLUSTER_impl {
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
  NtsVar local_mask;
  int  train_correct;
  int train_num;
  int val_correct;
  int val_num;
  int test_correct;
  int test_num;

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
  NtsVar global_label;
  NtsVar global_mask;
  NtsVar MASK;
  NtsVar MASK_gpu;
  // GraphOperation *gt;
  PartitionedGraph* partitioned_graph;
  // Variables
  std::vector<Parameter*> P;
  std::vector<NtsVar> X;
  nts::ctx::NtsContext* ctx;
  // Sampler* train_sampler;
  // Sampler* val_sampler;
  // Sampler* test_sampler;
  FullyRepGraph* fully_rep_graph;

  double gcn_run_time = 0;
  double epoch_sample_time = 0;
  // double epoch_gather_label_time = 0;
  // double epoch_gather_feat_time = 0;
  double epoch_transfer_graph_time = 0;
  double epoch_transfer_feat_time = 0;
  double epoch_transfer_label_time = 0;
  double epoch_train_time = 0;

  Cuda_Stream* cuda_stream;
  int pipelines;
  Cuda_Stream* cuda_stream_list;
  std::vector<at::cuda::CUDAStream> torch_stream;  

  NtsVar F;
  NtsVar loss;
  NtsVar tt;
  float acc;
  int batch;
  long correct;
  int max_batch_num;
  int min_batch_num;
  torch::nn::Dropout drpmodel;
  std::vector<torch::nn::BatchNorm1d> bn1d;
  ntsPeerRPC<ValueType, VertexId>* rpc;
  int hosts;

  int *is_sketch;
  float loss_epoch = 0;
  // float *mean_feat;
  std::map<int,std::vector<int>>class_nodes;
  std::vector<std::vector<std::pair<int,float>>>nid_nbr;

  GCN_CPU_CLUSTER_impl(Graph<Empty>* graph_, int iterations_, bool process_local = false,
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
    cuda_stream = new Cuda_Stream();

    pipelines = graph->config->pipelines;
    pipelines = std::max(1, pipelines);
    torch_stream.clear();
    is_sketch = graph->alloc_vertex_array<int>();
  }
  void init_graph() {
    fully_rep_graph = new FullyRepGraph(graph);
    fully_rep_graph->GenerateAll();
    fully_rep_graph->SyncAndLog("read_finish");
    // sampler=new Sampler(fully_rep_graph,0,graph->vertices);

    // cp = new nts::autodiff::ComputionPath(gt, subgraphs);
    ctx = new nts::ctx::NtsContext();
  }
  void init_nn() {
    learn_rate = graph->config->learn_rate;
    weight_decay = graph->config->weight_decay;
    drop_rate = graph->config->drop_rate;
    alpha = graph->config->learn_rate;
    decay_rate = graph->config->decay_rate;
    decay_epoch = graph->config->decay_epoch;
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-9;
    layers = graph->gnnctx->layer_size.size() - 1;
    gnndatum = new GNNDatum(graph->gnnctx, graph);
    // gnndatum->random_generate();
    if (0 == graph->config->feature_file.compare("random")) {
      gnndatum->random_generate();
    } else {
      gnndatum->readFeature_Label_Mask(graph->config->feature_file, graph->config->label_file,
                                       graph->config->mask_file);
    }

    // creating tensor to save Label and Mask
    gnndatum->registLabel(L_GT_C);
    // gnndatum->registGlobalLabel(global_label);
    gnndatum->registMask(MASK);
    MASK_gpu = MASK.cuda();
    gnndatum->generate_gpu_data();
    // gnndatum->registGlobalMask(global_mask);
    
    

    torch::Device GPU(torch::kCUDA, 0);
    // if (graph->partition_id == 0) {
    //  std::cout << global_label << " " << global_mask << std::endl;
    // }

    // initializeing parameter. Creating tensor with shape [layer_size[i],
    // layer_size[i + 1]]
    // for (int i = 0; i < graph->gnnctx->layer_size.size() - 1; i++) {
    //   P.push_back(new Parameter(graph->gnnctx->layer_size[i], graph->gnnctx->layer_size[i + 1], alpha, beta1, beta2,
    //                             epsilon, weight_decay));
    // }

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

    // synchronize parameter with other processes
    // because we need to guarantee all of workers are using the same model
    // for (int i = 0; i < P.size(); i++) {
    //   P[i]->init_parameter();
    //   P[i]->set_decay(decay_rate, decay_epoch);
    // }

    for (int i = 0; i < P.size(); i++) {
      P[i]->init_parameter();
      P[i]->set_decay(decay_rate, decay_epoch);
      P[i]->to(GPU);
      P[i]->Adam_to_GPU();
    }

    // drpmodel = torch::nn::Dropout(torch::nn::DropoutOptions().p(drop_rate).inplace(true));

    F = graph->Nts->NewLeafTensor(gnndatum->local_feature, {graph->gnnctx->l_v_num, graph->gnnctx->layer_size[0]},
                                  torch::DeviceType::CPU);

    // X[i] is vertex representation at layer i
    for (int i = 0; i < graph->gnnctx->layer_size.size(); i++) {
      NtsVar d;
      X.push_back(d);
    }

    X[0] = F.set_requires_grad(true);

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
  }

  long getCorrect(NtsVar input, NtsVar target, int type = 0) {
    // NtsVar predict = input.log_softmax(1).argmax(1);
    // NtsVar mask = (local_mask == type);
    // left = left.masked_select(mask.unsqueeze(1).expand({left.size(0), left.size(1)})).view({-1, left.size(1)});
    // right = right.masked_select(mask.view({mask.size(0)}));

    // std::cout << "before " << input.size(0) << std::endl;
    // input = input.masked_select(mask.view({mask.size(0)}));
    NtsVar mask = (local_mask == type);
    // std::cout<<mask.sizes()<<input.sizes()<<target.sizes()<<endl;
    input = input.masked_select(mask.unsqueeze(1).expand({-1, input.size(1)})).view({-1, input.size(1)});
    // std::cout << "after " << input.size(0) << std::endl;
    target = target.masked_select(mask.view({mask.size(0)}));
    NtsVar predict = input.argmax(1);
    NtsVar output = predict.to(torch::kLong).eq(target).to(torch::kLong);
    
    if (type == 0) {
      // printf("rigiht num %d pre num %d\n",output.sum().item<long>(),output.size(0));
      train_correct += output.sum().item<long>();
      train_num += output.size(0);
      // printf("train_correct %d train_num %d\n",train_correct,train_num);
    } else if (type == 1) {
      // printf("rigiht num %d pre num %d\n",output.sum().item<long>(),output.size(0));
      val_correct += output.sum().item<long>();
      val_num += output.size(0);
      // printf("val_correct %d val_num %d\n",val_correct,val_num);
    } else {
      // printf("rigiht num %d pre num %d\n",output.sum().item<long>(),output.size(0));
      test_correct += output.sum().item<long>();
      test_num += output.size(0);
      // printf("test_correct %d test_num %d\n",test_correct,test_num);
    }
    return output.sum(0).item<long>();
  }

  void Test(long s, NtsVar& target, NtsVar& mask) {  // 0 train, //1 eval //2 test
    NtsVar mask_train = mask.eq(s);
    NtsVar all_train = X[graph->gnnctx->layer_size.size() - 1]
                           .argmax(1)
                           .to(torch::kLong)
                           .eq(target)
                           .to(torch::kLong)
                           .masked_select(mask_train.view({mask_train.size(0)}));
    long p_correct = all_train.sum(0).item<long>();
    long p_train = all_train.size(0);
    float acc_train = 1.0 * p_correct / p_train;
    if (graph->partition_id == 0) {
      if (s == 0) {
        LOG_INFO("Train Acc: %f %d %d", acc_train, p_train, p_correct);
      } else if (s == 1) {
        LOG_INFO("Eval Acc: %f %d %d", acc_train, p_train, p_correct);
      } else if (s == 2) {
        LOG_INFO("Test Acc: %f %d %d", acc_train, p_train, p_correct);
      }
    }
  }

  // void Loss(NtsVar left, NtsVar right, int type = 0) {
  //   NtsVar mask = (local_mask == type);
  //   left = left.masked_select(mask.unsqueeze(1).expand({left.size(0), left.size(1)})).view({-1, left.size(1)});
  //   // left = left[mask.nonzero().view(-1)];
  //   right = right.masked_select(mask.view({mask.size(0)}));
  //   //  return torch::nll_loss(a,L_GT_C);
  //   torch::Tensor a = left.log_softmax(1);
  //   NtsVar loss_ = torch::nll_loss(a, right);
  //   if (ctx->training == true) {
  //     ctx->appendNNOp(left, loss_);
  //   }
  // }

  // void Update() {
  //   for (int i = 0; i < P.size(); i++) {
  //     // accumulate the gradient using all_reduce
  //     // P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
  //     if (graph->gnnctx->l_v_num == 0) {
  //       P[i]->all_reduce_to_gradient(torch::zeros_like(P[i]->W));
  //     } else {
  //       P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
  //     }
  //     // update parameters with Adam optimizer
  //     P[i]->learnC2C_with_decay_Adam();
  //     P[i]->next();
  //   }
  // }

  void Update() {
    for (int i = 0; i < P.size(); i++) {
      // accumulate the gradient using all_reduce
      // if (ctx->is_train() && graph->rtminfo->epoch >= 3) mpi_comm_time -= get_time();
      if (graph->gnnctx->l_v_num == 0) {
        P[i]->all_reduce_to_gradient(torch::zeros_like(P[i]->W));
      } else {
        P[i]->all_reduce_to_gradient(P[i]->W.grad().cpu());
      }
      // if (ctx->is_train() && graph->rtminfo->epoch >= 3) mpi_comm_time += get_time();
      // update parameters with Adam optimizer
      // printf("adam!!");
      // P[i]->learnC2G(learn_rate);
      // P[i]->learnC2C_with_Adam();
      P[i]->learnC2G_with_decay_Adam();
      // P[i]->learnC2C_with_decay_Adam();
      P[i]->next();
      // P[i]->learnC2G_with_decay_SGD();
      // P[i]->learnC2G_with_SGD();
    }
  }  

  void UpdateZero() {
    for (int l = 0; l < (graph->gnnctx->layer_size.size() - 1); l++) {
      //          std::printf("process %d epoch %d last before\n", graph->partition_id, curr_epoch);
      P[l]->all_reduce_to_gradient(torch::zeros({P[l]->row, P[l]->col}, torch::kFloat));
      //          std::printf("process %d epoch %d last after\n", graph->partition_id, curr_epoch);
      P[l]->learnC2C_with_decay_Adam();
      P[l]->next();
    }
  }

  void zero_grad() {
    for (int i = 0; i < P.size(); i++) {
      P[i]->zero_grad();
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

  void Forward(Sampler* sampler, int type = 0) {
    epoch_sample_time = 0;
    epoch_train_time = 0;
    epoch_transfer_graph_time = 0;
    epoch_transfer_feat_time = 0;
    epoch_transfer_label_time = 0;
    graph->rtminfo->forward = true;
    X[0] = graph->Nts->NewLeafTensor({1000, F.size(1)}, torch::DeviceType::CUDA);
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("used %.3f total %.3f (after new feat)", used_gpu_mem, total_gpu_mem);
    NtsVar target_lab;
    if (graph->config->classes > 1) {
      target_lab =
          graph->Nts->NewLabelTensor({graph->config->batch_size, graph->config->classes}, torch::DeviceType::CUDA);
    } else {
      target_lab = graph->Nts->NewLabelTensor({1000}, torch::DeviceType::CUDA);
    }
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("used %.3f total %.3f (after new label)", used_gpu_mem, total_gpu_mem);
    epoch_sample_time -= get_time();
    sampler->ClusterGCNSample(graph->gnnctx->layer_size.size() - 1, graph->config->batch_size, 1000);
    epoch_sample_time += get_time();
    // get_gpu_mem(used_gpu_mem, total_gpu_mem);
    // LOG_DEBUG("used %.3f total %.3f (after cluster sampler)", used_gpu_mem, total_gpu_mem);

    // auto sg = sampler->subgraph;

    int batch_num = 0;
    if(1000%graph->config->batch_size!=0) {
      batch_num = 1000/graph->config->batch_size + 1;
    } else {
      batch_num = 1000/graph->config->batch_size;
    }  


    train_correct = 0;
    train_num = 0;
    val_correct = 0;
    val_num = 0;
    test_correct = 0;
    test_num = 0;
    batch = 0;    
    loss_epoch = 0.0;

    for(int i=0; i<batch_num; i++)
    {
      if (ctx->training == true) zero_grad();
      auto sg = sampler->get_one();
      sampler->subgraph = sg;
      if (graph->config->mini_pull > 0) {  // generate csr structure for backward of pull mode
        for (auto p : sg->sampled_sgs) {
          p->generate_csr_from_csc();
        }
      }
      sg->alloc_dev_array(graph->config->mini_pull > 0);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("used %.3f total %.3f (after alloc_dev_array)", used_gpu_mem, total_gpu_mem);      
      // printf("test111!\n");    
      sg->compute_weight(graph);
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("used %.3f total %.3f (after compute_weight)", used_gpu_mem, total_gpu_mem);          
      // printf("test112!\n");       
      // for(int i=0;i<sg->sampled_sgs[0]->v_size;i++) {
      //   printf("nbr num: %d\n",sg->sampled_sgs[0]->column_offset[i+1]-sg->sampled_sgs[0]->column_offset[i]);
      // } 
      // std::cout<<sg->sampled_sgs[0]->v_size<<sg->sampled_sgs[0]->src_size<<sg->sampled_sgs[1]->v_size<<sg->sampled_sgs[1]->src_size<<endl;
      // int batch_num = sampler->size();
      // MPI_Allreduce(&batch_num, &max_batch_num, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      // MPI_Allreduce(&batch_num, &min_batch_num, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

      // acc=0.0;
      // correct = 0;

      
      // sg = sampler->get_one();
      epoch_transfer_graph_time -= get_time();
      // sg->trans_graph_to_gpu(graph->config->mini_pull > 0);
      sg->trans_graph_to_gpu_async(cuda_stream_list[0].stream, graph->config->mini_pull > 0);
      epoch_transfer_graph_time += get_time();
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("used %.3f total %.3f (after transfer_graph)", used_gpu_mem, total_gpu_mem);        
      // std::vector<NtsVar> X;
      NtsVar d;
      // X.resize(graph->gnnctx->layer_size.size(), d);

      //  X[0]=nts::op::get_feature(sg->sampled_sgs[graph->gnnctx->layer_size.size()-2]->src(),F,graph);
      // X[0]=nts::op::get_feature(sg->sampled_sgs[0]->src(),F,graph);
      // std::cout << "dst num " << sg->sampled_sgs.back()->dst().size() << std::endl;
      // X[0] =
          // nts::op::get_feature_from_global(rpc, sg->sampled_sgs[0]->src().data(), sg->sampled_sgs[0]->src_size, F, graph);
      
      // X[0] = nts::op::get_feature(sg->sampled_sgs[0]->src().data(), sg->sampled_sgs[0]->src_size, F, graph);
      // epoch_transfer_feat_time -= get_time();
      // X[0] = X[0].cuda().set_requires_grad(true);
      // epoch_transfer_feat_time += get_time();

        epoch_transfer_feat_time -= get_time();
        sampler->load_feature_gpu(&cuda_stream_list[0], sg, X[0], gnndatum->dev_local_feature);
        epoch_transfer_feat_time += get_time();      
        // get_gpu_mem(used_gpu_mem, total_gpu_mem);
        // LOG_DEBUG("used %.3f total %.3f (after transfer_feat)", used_gpu_mem, total_gpu_mem);     
          
      // sampler->load_feature_gpu(X[0], gnndatum->dev_local_feature);
      // std::cout<<X[0]<<endl;
      // printf("get feature done\n");

      // NtsVar target_lab=nts::op::get_label(sg->sampled_sgs.back()->dst(),L_GT_C,graph);
      // NtsVar target_lab = nts::op::get_label_from_global(sg->sampled_sgs.back()->dst().data(),
      //                                                    sg->sampled_sgs.back()->v_size, global_label, graph);

      // target_lab =
      //       nts::op::get_label(sg->sampled_sgs.back()->dst().data(), sg->sampled_sgs.back()->v_size, L_GT_C, graph);
      // epoch_transfer_label_time -= get_time();
      // target_lab = target_lab.cuda();
      // epoch_transfer_label_time += get_time();

      epoch_transfer_label_time -= get_time();
      sampler->load_label_gpu(&cuda_stream_list[0], sg, target_lab, gnndatum->dev_local_label);
      epoch_transfer_label_time += get_time();
      // get_gpu_mem(used_gpu_mem, total_gpu_mem);
      // LOG_DEBUG("used %.3f total %.3f (after transfer_label)", used_gpu_mem, total_gpu_mem); 

      // sampler->load_label_gpu(target_lab, gnndatum->dev_local_label);
      // sampler->load_label_gpu(cuda_stream, sg, target_lab, gnndatum->dev_local_label);
      // printf("get label done\n");
      // std::cout << "target_lab " << target_lab.sizes()  << std::endl;

      //  graph->rtminfo->forward = true;
      // for (int l = 0; l < (graph->gnnctx->layer_size.size() - 1); l++) {  // forward
      //   printf("process layer %d\n", l);
      //   //  int hop=(graph->gnnctx->layer_size.size()-2)-l;
      //   // if(l!=0){
      //   //     X[l] = drpmodel(X[l]);
      //   // }
      //   NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(sg, graph, l, X[l]);




      //   printf("finished graph op layers %d\n", l);
      //   X[l + 1] = ctx->runVertexForward(
      //       [&](NtsVar n_i) {
      //         if (l == (graph->gnnctx->layer_size.size() - 2)) {
      //           return P[l]->forward(n_i);
      //         } else {
      //           // return torch::relu(P[l]->forward(n_i));
      //           return torch::dropout(P[l]->forward(n_i), drop_rate, ctx->is_train());
      //         }
      //       },
      //       Y_i);
      // }
        epoch_train_time -= get_time();
        at::cuda::setCurrentCUDAStream(torch_stream[0]);
        for (int l = 0; l < layers; l++) {  // forward
          graph->rtminfo->curr_layer = l;
          // std::cout<<X[l]<<endl;
          NtsVar Y_i = ctx->runGraphOp<nts::op::SingleGPUSampleGraphOp>(sg, graph, l, X[l], cuda_stream);
          // std::cout<<X[l].sizes()<<Y_i.sizes()<<endl;
          X[l + 1] = ctx->runVertexForward([&](NtsVar n_i) { return vertexForward(n_i); }, Y_i);
          // std::cout<<X[l+1].sizes()<<endl;
        }    

      local_mask = torch::zeros_like(target_lab, at::TensorOptions().dtype(torch::kLong));
      auto vec_dst = sg->sampled_sgs.back()->dst();
      // printf("start local mask\n");
      for (int i = 0; i < sg->sampled_sgs.back()->v_size; ++i) {
        // local_mask[i] = MASK[vec_dst[i]].item<long>();
        local_mask[i] = MASK[vec_dst[i]].item<long>();
      }
      // printf("end local mask\n");

      // Loss(X[graph->gnnctx->layer_size.size() - 1], target_lab, 0);
      auto loss_ = Loss(X[layers], target_lab, graph->config->classes == 1);
      loss_epoch += loss_.item<float>();
      getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab, 0);
      // MPI_Allreduce(MPI_IN_PLACE, &train_correct, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      // MPI_Allreduce(MPI_IN_PLACE, &train_num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      // acc = 1.0 * correct / train_nodes;

      // graph->rtminfo->forward = false;
      // if (ctx->training) {
      //   ctx->self_backward(false);
      //   Update();
      // }

        if (ctx->is_train()) {
          ctx->appendNNOp(X[layers], loss_);
          ctx->self_backward(false);
          Update();
        }    
      // printf("batch %d %d %d %d %d %d %d\n",batch,train_correct,train_num,val_correct,val_num,test_correct,test_num);
      epoch_train_time += get_time();
      
      getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab, 1);
      // MPI_Allreduce(MPI_IN_PLACE, &val_correct, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      // MPI_Allreduce(MPI_IN_PLACE, &val_num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      getCorrect(X[graph->gnnctx->layer_size.size() - 1], target_lab, 2);
      // MPI_Allreduce(MPI_IN_PLACE, &test_correct, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      // MPI_Allreduce(MPI_IN_PLACE, &test_num, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
      sg->free_dev_array(graph->config->mini_pull > 0);
      batch++;
      
      // Test(0, target_lab, mask);
      // Test(1, target_lab, mask);
      // Test(2, target_lab, mask);

      // sampler->clear_queue();
      // sampler->restart();
    }
  }

void sketch_from_file(std::vector<VertexId> &sketch_train_nids) {
    ifstream fin;
    fin.open("/home/yuanh/neutron-sanzo/exp/exp-sketch/reddit-tmp.log",ios::in);
    string buf;
    while(fin >> buf) {
      // printf("%d",int(buf[0]));
      sketch_train_nids.push_back(atoi(buf.c_str()));
      is_sketch[atoi(buf.c_str())] = 1;
    }
    change_CSC_by_sketch(sketch_train_nids);
  }

  void change_CSC_by_sketch(std::vector<VertexId> &sketch_train_nids) {
    VertexId *coloffset = new VertexId[graph->vertices+1];
    VertexId *rowindex = new VertexId[graph->edges];
    // VertexId egde_nums = 0;
    memset(coloffset,0,sizeof(VertexId)*(graph->vertices+1));
    memset(rowindex,0,sizeof(VertexId)*(graph->edges));
    int nodes_num = 0;
    int edge_nums = 0;
    int tmp = 0;
    int index = 0;
    // std::map<VertexId,VertexId>global_local;
    for(int i=0;i<graph->vertices;i++){
      if(gnndatum->local_mask[i]!=0) {
        fully_rep_graph->global_local.insert(std::make_pair(i,index++));
      } else if(is_sketch[i]==1) {
        fully_rep_graph->global_local.insert(std::make_pair(i,index++));
      }
      
    }

    for(auto &i:sketch_train_nids) {
      i = fully_rep_graph->global_local[i];
    }


    for(int i=0;i<graph->vertices;i++) {
      int tmp_num = 0;
      if(gnndatum->local_mask[i]==0) {
        if(is_sketch[i]==1) {
          for(int j=fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++) {
            if(gnndatum->local_mask[fully_rep_graph->row_indices[j]]!=0 || is_sketch[fully_rep_graph->row_indices[j]]==1 ) {
              rowindex[edge_nums++] = fully_rep_graph->global_local[fully_rep_graph->row_indices[j]];
              tmp_num ++;
            }
          }
          coloffset[nodes_num+1] = coloffset[nodes_num] + tmp_num;
          // printf("nodes %d coloffset %d\n",nodes_num,coloffset[nodes_num+1]);
          fully_rep_graph->local_global.insert(std::make_pair(nodes_num,i));
          nodes_num += 1;
          // memcpy(rowindex+edge_nums,fully_rep_graph->row_indices+fully_rep_graph->column_offset[i],fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i]);
          // edge_nums += fully_rep_graph->column_offset[i+1]-fully_rep_graph->column_offset[i];
        } else {
          // coloffset[i+1] = coloffset[i];
          tmp++;
        }
      } else {
        for(int j=fully_rep_graph->column_offset[i];j<fully_rep_graph->column_offset[i+1];j++) {
              if(gnndatum->local_mask[fully_rep_graph->row_indices[j]]!=0 || is_sketch[fully_rep_graph->row_indices[j]]==1 ) {
                rowindex[edge_nums++] = fully_rep_graph->global_local[fully_rep_graph->row_indices[j]];
                tmp_num ++;
              }
            }
        coloffset[nodes_num+1] = coloffset[nodes_num] + tmp_num;
        // printf("nodes %d coloffset %d\n",nodes_num,coloffset[nodes_num+1]);
        fully_rep_graph->local_global.insert(std::make_pair(nodes_num,i));
        nodes_num += 1;
      }
    }
    // printf("node nums: %d %d origin: %d tmp: %d\n",nodes_num,edge_nums,fully_rep_graph->global_edges,tmp,edge_nums);
    // printf("co end %d edge num %d\n",coloffset[nodes_num ],edge_nums);
    delete fully_rep_graph->column_offset;
    delete fully_rep_graph->row_indices;
    VertexId *rowindex_tmp = new VertexId[edge_nums];
    VertexId *coloffset_tmp = new VertexId[nodes_num + 1];
    
    memset(rowindex_tmp,0,sizeof(VertexId)*(edge_nums));
    memset(coloffset_tmp,0,sizeof(VertexId)*(nodes_num + 1));
    std::copy(rowindex,rowindex+edge_nums,rowindex_tmp);
    std::copy(coloffset,coloffset+nodes_num+1,coloffset_tmp);
    // memccpy(rowindex_tmp,rowindex,edge_nums,sizeof(VertexId));
    // memccpy(coloffset_tmp,coloffset,nodes_num + 1,sizeof(VertexId));
    // delete rowindex;
    // delete coloffset;
    fully_rep_graph->column_offset = coloffset_tmp;
    fully_rep_graph->row_indices = rowindex_tmp;
    fully_rep_graph->global_vertices = nodes_num;
    fully_rep_graph->global_edges = edge_nums;
    // for(int i=0;i<fully_rep_graph->global_vertices;i++) {
    //   printf("%d %d\n",coloffset[i+1],fully_rep_graph->column_offset[i+1] );
    // }
    // printf("co end %d edge num %d\n",fully_rep_graph->column_offset[fully_rep_graph->global_vertices],fully_rep_graph->global_edges);
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
    for(int i=0;i<graph->vertices;i++)
    // for(auto i:sketch_train_nids)
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

    for(int i=0;i<graph->gnnctx->label_num;i++) {
      printf("class: %d inner-Sim: %lf \n",i,innerSim[i]);
    }

    double sums = 0.0;
   for(int i=0;i<graph->gnnctx->label_num-1;i++) {
      for(int j=i;j<graph->gnnctx->label_num;j++) {
        sums += cosS[i*class_num+j];
      }
    }
    double sim_inter = sums/(2*(class_num*class_num-class_num));
    LOG_INFO("sim inter: %lf",sim_inter);

      // for(int i=0;i<graph->vertices;i++) {
        
      //     // printf("test before sim\n");
      //     std::vector<std::pair<int,float>>tmp_vec;
      //     for(int j=fully_rep_graph->column_offset[i],index=0;j<fully_rep_graph->column_offset[i+1];j++,index++) {
      //       VertexId nbr = fully_rep_graph->row_indices[j];
      //       // printf("test in sim\n");
      //       // if(gnndatum->local_mask[nbr]==0 && is_sketch[nbr]==0) {
      //       //   fully_rep_graph->sim_value[j] = 0;
      //       // } else {
      //       //   fully_rep_graph->sim_value[j] = comp_cos_Similarity(gnndatum->local_feature+(graph->gnnctx->layer_size[0]*i),gnndatum->local_feature+(graph->gnnctx->layer_size[0]*nbr));
      //       // mean_feat+(graph->gnnctx->layer_size[0]*gnndatum->local_label[i])
      //       // }
      //       // fully_rep_graph->sim_value[j] = comp_cos_Similarity(gnndatum->local_feature+(graph->gnnctx->layer_size[0]*i),gnndatum->local_feature+(graph->gnnctx->layer_size[0]*nbr));            
      //       fully_rep_graph->sim_value[j] = tmp_comp_nbr_sim(i,nbr);
      //       // 搞个vector，将每个顶点邻居按相似度降序存储，采样时按邻居相似度采样！
      //       tmp_vec.push_back(std::make_pair(index,fully_rep_graph->sim_value[j]));
      //       // printf("test after sim\n");
      //     }
      //     sort(tmp_vec.begin(), tmp_vec.end(), [&](auto& x, auto& y) { return x.second > y.second; });
      //     nid_nbr.push_back(tmp_vec);

      // }    

    return cosS;
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

// show the number of sketch and trian nodes in different classes (supple nodes for fewer classes )
  // this function can be add to the fun sketch_same_with_origin_train_set()
  void get_diff_class_nodes(std::vector<VertexId> &sketch_train_nids)
  {
    LOG_INFO("Before Add sketch nodes nums: %d",sketch_train_nids.size());
    // array
    long *class_arry_origin = new long[graph->gnnctx->label_num];
    long *class_arry_sketch = new long[graph->gnnctx->label_num];
    // int *sketch_array = graph->alloc_vertex_array<int>();
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
      // sketch_array[tmp_nid] = 1;
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
      // get_diff_class_nodes(sketch_train_nids);
      // show_sketch_label_class(sketch_train_nids);
    } 
    // else if(sketch_mode == 1){
    //   sketch_train_nids.clear();
    //   int i = 0;
    //   while(i<graph->config->nodes_num)
    //   {
    //     int tmp_nid = rand_int(graph->vertices);
    //     if (gnndatum->local_mask[tmp_nid] == 0)
    //     {
    //       sketch_train_nids.push_back(tmp_nid + graph->partition_offset[graph->partition_id]);
    //       i+=1;
    //     }
    //   }
    //   get_diff_class_nodes(sketch_train_nids);
    //   show_sketch_label_class(sketch_train_nids);
    // } else if(sketch_mode == 2){
    //   double *score = PageRank(graph,5);
    //   // std::map<int,double>tmp;
    //   std::vector<pair<int,double>>tmp_vec;
    //   for(int i = 0; i<graph->vertices; i++)
    //   {
    //     tmp_vec.push_back(std::make_pair(i,score[i]));
    //   }
    //   sort(tmp_vec.begin(),tmp_vec.end(),cmp);
    //   int i = 0;
    //   int j = 0;
    //   while(i<graph->config->nodes_num){
    //     int tmp_nid = tmp_vec[j].first;
    //     if (gnndatum->local_mask[tmp_nid] == 0)
    //     {
    //       sketch_train_nids.push_back(tmp_nid + graph->partition_offset[graph->partition_id]);
    //       i+=1;
    //       j+=1;
    //     }
    //     else {
    //       j+=1;
    //     }
    //   }
    //   get_diff_class_nodes(sketch_train_nids);
    //   show_sketch_label_class(sketch_train_nids);
    //   // printf("%d,%lf",tmp_vec[0].first,tmp_vec[0].second);
    // } else if(sketch_mode == 3){
    //   int *rank = graph->alloc_vertex_array<int>();
    //   rank = k_core();
    //   for(int i=0; i<graph->vertices; i++)
    //   {
    //     int core_rank = rank[i];
    //     LOG_INFO("nid: %d label: %d mask: %d degree: %d score: %d",i,gnndatum->local_label[i],gnndatum->local_mask[i],graph->out_degree[i],rank[i]);

    //     // if(core_rank >= graph->config->nodes_num && gnndatum->local_mask[i] == 0)
    //     // {
    //     //   printf("%d\n",i);
    //     //   sketch_train_nids.push_back(i + graph->partition_offset[graph->partition_id]);
    //     // }
    //   }
    //   // get_diff_class_nodes(sketch_train_nids);
    //   // show_sketch_label_class(sketch_train_nids);
    // } else if(sketch_mode == 4){
    //   sketch_by_label_rate(0.05,sketch_train_nids);
    //   // printf("123");
    //   get_diff_class_nodes(sketch_train_nids);
    //   // printf("567");
    //   show_sketch_label_class(sketch_train_nids);
    // } else if(sketch_mode == 5){
    //   sketch_same_with_origin_train_set(0.2,sketch_train_nids);
    //   get_diff_class_nodes(sketch_train_nids);
    //   show_sketch_label_class(sketch_train_nids);
    // } else if(sketch_mode == 6){
    //   printf("get sketch nodes! %d\n",graph->config->classes);

    //   sketch_by_similarity(sketch_train_nids);
    //   // get_diff_class_nodes(sketch_train_nids);
    //   show_sketch_label_class(sketch_train_nids);
    // } 
    else if(sketch_mode == 7){
      sketch_from_file(sketch_train_nids);
      get_diff_class_cos_similarity(sketch_train_nids);
      // if(graph->config->run_mode=="test") {
      //   get_class_score();
      // }
      
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
            if(class_nid_per[j].first != cur_class and class_nid_per[j].second >= 0.3) {
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
      printf("test11\n");
      change_CSC_by_sketch(sketch_train_nids);
      printf("test22\n");
      // get_diff_class_nodes(sketch_train_nids);
      // printf("test33\n");
      // show_sketch_label_class(sketch_train_nids);
    }


    LOG_INFO("Sketch Mode: (%d) Sketch train nodes: (%d) (%.3f)",
              sketch_mode,
              sketch_train_nids.size(),
              sketch_train_nids.size() * 1.0 / graph->vertices);
  }


    void empty_gpu_cache() {
    for (int ti = 0; ti < 5; ++ti) {  // clear gpu cache memory
      c10::cuda::CUDACachingAllocator::emptyCache();
    }
  }

  void run() {
    if (graph->partition_id == 0) {
      LOG_INFO("GNNmini::[Dist.GPU.GCNimpl] running [%d] Epoches\n", iterations);
    }
    // get train/val/test node index. (may be move this to GNNDatum)
    std::vector<VertexId> train_nids, val_nids, test_nids;
    std::vector<VertexId> sketch_train_nids;
    get_sketch_nodes(sketch_train_nids,graph->config->sketch_mode);    
    for (int i = 0; i < graph->gnnctx->l_v_num; ++i) {
      // printf("%d\n",i);
      int type = gnndatum->local_mask[i];
      if (type == 0) {
        train_nids.push_back(fully_rep_graph->global_local[i + graph->partition_offset[graph->partition_id]]);
      } else if (type == 1) {
        val_nids.push_back(fully_rep_graph->global_local[i + graph->partition_offset[graph->partition_id]]);
      } else if (type == 2) {
        test_nids.push_back(fully_rep_graph->global_local[i + graph->partition_offset[graph->partition_id]]);
      }
    }
    printf("test000 %d\n",train_nids.size());
    Sampler* train_sampler = new Sampler(fully_rep_graph, train_nids);
    printf("test111\n");
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
    // printf("test111\n");
    // Sampler* train_sampler;
    // printf("%d\n",fully_rep_graph->global_vertices);

    Sampler* train_sampler_sketch = new Sampler(fully_rep_graph, sketch_train_nids);
    train_sampler = train_sampler_sketch;

    best_val_acc = 0.0;
    int tmp = 0;
    gcn_run_time = 0;
    for (int i_i = 0; i_i < iterations; i_i++) {
      graph->rtminfo->epoch = i_i;
      empty_gpu_cache();
      // printf("########### epoch %d ###########\n", i_i);
      if (i_i != 0) {
        for (int i = 0; i < P.size(); i++) {
          P[i]->zero_grad();
        }
      }

      ctx->train();
      Forward(train_sampler, 0);
      double epoch_trans_time = epoch_transfer_graph_time + epoch_transfer_feat_time + epoch_transfer_label_time;
      double epoch_all_train_time = epoch_sample_time + epoch_train_time + epoch_trans_time;
      gcn_run_time += epoch_all_train_time;
      float train_acc = float(train_correct) / train_num;
      float val_acc = float(val_correct) / val_num;
      float test_acc = float(test_correct) / test_num;
      loss_epoch /= batch;
      if(best_val_acc < val_acc) {
        best_val_acc = val_acc;
        tmp = 0;
      }else {
        tmp++;
        if(tmp == 10) {
          break;
        }
      }
      // LOG_INFO("Epoch %03d train_acc %.3f val_acc %.3f test_acc %.3f\n", i_i, train_acc, val_acc, test_acc);
      // printf("train_correct %d train_num %d\n",train_correct,train_num);
      // printf("val_correct %d val_num %d\n",val_correct,val_num);
      // printf("test_correct %d test_num %d\n",test_correct,test_num);
      LOG_INFO("Epoch %03d loss %.3f train_acc %.3f val_acc %.3f test_acc %.3f epoch_sample_time %.3f epoch_train_time %.3f gcn_run_time %.3f\n", i_i, loss_epoch ,train_acc, val_acc, test_acc,epoch_sample_time,epoch_train_time,gcn_run_time);
      // printf("Epoch %03d train_acc %.3f train_correct %d train_nums %d val_acc %.3f val_correct %d val_nums %d test_acc %.3f test_correct %d correct_nums %d \n", i_i, train_acc, train_correct,train_num, val_acc,val_correct,val_num,test_acc,test_correct,test_num);

      //      if (graph->partition_id == 0)
      //        std::cout << "Nts::Running.Epoch[" << i_i << "]:loss\t" << loss
      //                  << std::endl;
    train_sampler->clear_queue();
    }
    delete active;
    // rpc.exit();
  }
};



