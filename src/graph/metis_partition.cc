/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/metis_partition.cc
 * @brief Call Metis partitioning
 */

#include <dgl/graph_op.h>
#include <dgl/packed_func_ext.h>
#include "../../third_party/KaHIP/interface/kaHIP_interface.h"
#include <iostream>
#include <sstream>
#include "../c_api_common.h"

using namespace dgl::runtime;

namespace dgl {

#if !defined(_WIN32)

IdArray KaHIPPartition(GraphPtr g, int k, NDArray vwgt_arr, bool obj_cut) {
  // The index type of KaHIP needs to be compatible with DGL index type.
  //CHECK_EQ(sizeof(idx_t), sizeof(dgl_id_t));
  ImmutableGraphPtr ig = std::dynamic_pointer_cast<ImmutableGraph>(g);
  CHECK(ig) << "The input graph must be an immutable graph.";
  // This is a symmetric graph, so in-csr and out-csr are the same.
  const auto mat = ig->GetInCSR()->ToCSRMatrix();

  int nvtxs = g->NumVertices();
  int *xadj = static_cast<int *>(mat.indptr->data);
  int *adjncy = static_cast<int *>(mat.indices->data);
  int nparts = k;
  IdArray part_arr = aten::NewIdArray(nvtxs);
  int objval = 0;
  int *part = static_cast<int *>(part_arr->data);
  double imbalance = 0.03
  int vwgt_len = vwgt_arr->shape[0];
  CHECK_EQ(sizeof(int), vwgt_arr->dtype.bits / 8)
      << "The vertex weight array doesn't have right type";
  CHECK(vwgt_len % g->NumVertices() == 0)
      << "The vertex weight array doesn't have right number of elements";
  int *vwgt = nullptr;
  if (vwgt_len > 0) {
    vwgt = static_cast<int *>(vwgt_arr->data);
  }

  kaffpa(
      &nvtxs,  // The number of vertices
      vwgt,    // the weights of the vertices
      xadj,    // indptr
      nullptr, //adjcwgt 
      adjncy,  // indices
      &nparts,  // The number of partitions.
      &imbalance,     //imbalance
      false,    //supress output
      34349123, //seed
      0,  // Option of KaHIP, 0 = FAST
      &objval,  // the edge-cut or the total communication volume of
      // the partitioning solution
      part);

    LOG(INFO) << "Partition a graph with " << g->NumVertices() << " nodes and "
              << g->NumEdges() << " edges into " << k << " parts and "
              << "get " << objval << " edge cuts";

  // return an array of 0 elements to indicate the error.
  return aten::NullArray();
}

#endif  // !defined(_WIN32)

DGL_REGISTER_GLOBAL("transform._CAPI_DGLMetisPartition")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      GraphRef g = args[0];
      int k = args[1];
      NDArray vwgt = args[2];
      bool obj_cut = args[3];
#if !defined(_WIN32)
      *rv = KaHIPPartition(g.sptr(), k, vwgt, obj_cut);
#else
      LOG(FATAL) << "KaHIP partition does not support Windows.";
#endif  // !defined(_WIN32)
    });

}  // namespace dgl
