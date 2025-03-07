/*
Copyright (c) 2021-2022 Qiange Wang, Northeastern University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GRAPHSEGMENT_H
#define GRAPHSEGMENT_H
#include <fcntl.h>
#include <malloc.h>
#include <numa.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cuda/cuda_type.h"
#include "dep/gemini/atomic.hpp"
#include "dep/gemini/bitmap.hpp"
#include "dep/gemini/constants.hpp"
#include "dep/gemini/filesystem.hpp"
#include "dep/gemini/time.hpp"
#include "dep/gemini/type.hpp"
#if CUDA_ENABLE
#include "cuda/ntsCUDA.hpp"
#endif

const bool NOT_SUPPORT_DEVICE_TYPE = false;

enum BatchType { SHUFFLE, SEQUENCE, RANDOM, DELLOW, DELHIGH, METIS };

class CSC_segment_pinned {
 public:
  VertexId *column_offset;  // VertexNumber
  VertexId *row_indices;    // edge_size also the source nodes
  VertexId *row_offset;     // VertexNumber
  VertexId *column_indices;

  long *source;
  long *destination;
  //  long *source_backward;

  ValueType *edge_weight_forward;  // edge_size
  ValueType *edge_weight_backward;

  //  VertexId *backward_message_index;
  //  VertexId *forward_message_index;
  VertexId *forward_multisocket_message_index;
  BackVertexIndex *backward_multisocket_message_index;

  VertexId *column_offset_gpu;  // VertexNumber
  VertexId *row_indices_gpu;
  VertexId *row_offset_gpu;      // VertexNumber
  VertexId *column_indices_gpu;  // edge_size

  long *source_gpu;
  long *destination_gpu;

  //  long *source_backward_gpu;

  ValueType *edge_weight_forward_gpu;   // edge_size
  ValueType *edge_weight_backward_gpu;  // edge_size

  int edge_size;
  int batch_size_forward;
  int batch_size_backward;
  int input_size;
  int output_size;
  int feature_size;
  int src_range[2];
  int dst_range[2];
  Bitmap *source_active;
  Bitmap *destination_active;
  // std::vector<Bitmap *> VertexToComm;

  DeviceLocation dt;
  void init(VertexId src_start, VertexId src_end, VertexId dst_start, VertexId dst_end, VertexId edge_size_,
            DeviceLocation dt_);
  void optional_init_sample(int layers);

  // Allocate bitmap for forward and backward vertex
  // and row_offset and column_offset, for CSC/CSR format
  void allocVertexAssociateData();

  // allocate space for edge associated data.
  // e.g. destination vertexID, edge data
  void allocEdgeAssociateData();
  void getDevicePointerAll();
  void CopyGraphToDevice();
  void freeAdditional();

  inline bool src_get_active(VertexId v_i) { return this->source_active->get_bit(v_i - src_range[0]); }

  inline bool dst_get_active(VertexId v_i) { return this->destination_active->get_bit(v_i - dst_range[0]); }

  //  inline bool get_forward_active(VertexId v_i) {
  //      return this->forward_active->get_bit(v_i);
  //  }
  //
  //  inline void set_forward_active(VertexId v_i) {
  //      this->forward_active->set_bit(v_i);
  //  }
  //
  //  inline bool get_backward_active(VertexId v_i) {
  //      return this->source_active->get_bit(v_i - src_range[0]);
  //  }

  inline void src_set_active(VertexId v_i) { this->source_active->set_bit(v_i - src_range[0]); }

  inline void dst_set_active(VertexId v_i) { this->destination_active->set_bit(v_i - dst_range[0]); }
};

class SampleGraph {
  VertexId *source;
  VertexId *destination;
  VertexId *column_offset;
  VertexId *row_indices;
};

class InputInfo {
 public:
  size_t vertices;

  // engine related
  bool overlap;
  bool process_local;
  bool with_weight;
  size_t epochs;
  size_t repthreshold;
  bool lock_free;
  bool optim_kernel_enable;
  std::string algorithm;
  std::string layer_string;
  std::string fanout_string;
  std::string val_fanout_string = "";
  std::string feature_file;
  std::string edge_file;
  std::string label_file;
  std::string mask_file;
  std::string dataset_name;
  float del_frac = 0;
  bool with_cuda;
  int batch_norm = 0;
  int time_skip = 3;
  int runs = 1;
  int mini_pull = 1;
  float sample_rate = -1;
  int target_degree = -1;
  int sample_mode = -1;
  double run_time = -1;
  int best_parameter = 0;
  float cache_rate_end;
  float cache_rate_num;
  float cache_rate_start = 0;


  float threshold_trans = 0;
  float trans_threshold_start = 0;
  float trans_threshold_end;
  float trans_threshold_num;
  int block_size = 256;

  int cache_exp = 0;
  // int dynamic_sample = -1;

  float batch_switch_time = -1;
  int threads = -1;
  std::string batch_size_vec_string;
  std::vector<int> batch_size_vec;

  float sample_switch_time = -1;
  float fanout_switch_time = -1;
  std::string sample_rate_vec_string;
  std::vector<float> sample_rate_vec;
  std::string fanout_vec_string;
  std::vector<int> fanout_vec;
  int lower_fanout = 0;
  float cache_rate = 0;
  std::string cache_type = "none";
  std::string cache_policy;
  int batch_switch_acc = -1;
  int fanout_switch_acc = -1;
  int pipelines = 1;
  int degree_switch = -1;
  int sketch_mode = -1;
  int nodes_num = 0;
  float low_1 = 0.0;
  float high_1 = 0.0;
  float low_2 = 0.0;
  float high_2 = 0.0;
  std::string run_mode = "none";
  // int target_degree = -1;
  // algorithm related:
  VertexId batch_size;
  VertexId val_batch_size;
  ValueType learn_rate;
  ValueType weight_decay;
  ValueType decay_rate;
  ValueType decay_epoch;
  ValueType drop_rate;
  BatchType batch_type;
  VertexId partition_num = 0;
  int classes = 1;
  std::string mode;

  void readFromCfgFile(std::string config_file);
  void print();
};

class RuntimeInfo {
 public:
  bool process_local;
  bool with_cuda;
  bool process_overlap;
  bool with_weight;
  bool reduce_comm;
  size_t epoch;
  size_t curr_layer;
  size_t embedding_size;
  bool copy_data;
  bool forward;
  bool lock_free;
  bool optim_kernel_enable;
#if CUDA_ENABLE
  Cuda_Stream *cuda_stream_public;
#endif

  void init_rtminfo();

#if CUDA_ENABLE
  void device_sync() { cuda_stream_public->CUDA_DEVICE_SYNCHRONIZE(); }
#endif

  void set(InputInfo *gnncfg);
};

class GNNContext {
 public:
  std::vector<int> layer_size;  // feature size at each layer, 0 is input feature
  std::vector<int> fanout;      // feature size at each layer, 0 is input feature
  std::vector<int> val_fanout;  // feature size at each layer, 0 is input feature
  size_t max_layer;
  size_t label_num;
  size_t p_id;     // partition id
  size_t p_v_s;    // partition vertex start
  size_t p_v_e;    // partition vertex end
  size_t w_num;    // workernum
  size_t l_v_num;  // local |V|
  size_t l_e_num;  // local |E|
};

class BlockInfo {
 public:
  std::vector<VertexId> vertex;
  std::vector<VertexId> global_index;
  std::vector<VertexId> local_index;
  std::vector<VertexId> block_index;  // num of blocks+1
  VertexId *vertex_gpu_buffer;
  VertexId *index_gpu_buffer;
  VertexId max_buffer_size;  // for alloc
};

class GraphStorage {
 public:
  std::vector<Bitmap *> sampled_vertices;
  VertexId *column_offset;
  VertexId *row_indices;

  COOChunk *_graph_cpu_in;
  COOChunk *_graph_cpu_out;
  std::vector<COOChunk *> graph_shard_in;
  std::vector<COOChunk *> graph_shard_out;

  void optional_generate_sample_graph(GNNContext *gnnctx, COOChunk *_graph_cpu_in);
};

#endif
