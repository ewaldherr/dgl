/**
 *  Copyright (c) 2020 by Contributors
 * @file graph/kahip_partition.cc
 * @brief Call KaHIP partitioning
 */

#include <dgl/base_heterograph.h>
#include <dgl/packed_func_ext.h>
#include <kaHIP_interface.h>

#include "../heterograph.h"
#include "../unit_graph.h"

using namespace dgl::runtime;

namespace dgl {

namespace transform {

#if !defined(_WIN32)

IdArray KaHIPPartition(
    UnitGraphPtr g, int k, NDArray vwgt_arr) {
  // This is a symmetric graph, so in-csr and out-csr are the same.
  const auto mat = g->GetCSCMatrix(0);
  //   const auto mat = g->GetInCSR()->ToCSRMatrix();

  int nvtxs = g->NumVertices(0);
  int64_t *xadj = static_cast<int64_t *>(mat.indptr->data);
  int64_t* adjncy = static_cast<int64_t *>(mat.indices->data);
  int nparts = k;
  IdArray part_arr = aten::NewIdArray(nvtxs);
  int64_t * part = static_cast<int64_t *>(part_arr->data);
  double imbalance = 0.03;
  int64_t vwgt_len = vwgt_arr->shape[0];
  CHECK_EQ(sizeof(int), vwgt_arr->dtype.bits / 8)
      << "The vertex weight array doesn't have right type";
  CHECK(vwgt_len % g->NumVertices(0) == 0)
      << "The vertex weight array doesn't have right number of elements";
  int *vwgt = NULL;
  if (vwgt_len > 0) {
    vwgt = static_cast<int64_t *>(vwgt_arr->data);
  }
  std::cout << "kaffpa begins" << std::endl;
  kaffpa(
      &nvtxs,  // The number of vertices
      nullptr,    // the weights of the vertices
      xadj,    // indptr
      nullptr, //adjcwgt 
      adjncy,  // indices
      &nparts,  // The number of partitions.
      &imbalance,     //imbalance
      false,    //supress output
      0, //seed
      0,  // Option of KaHIP, 0 = FAST
      &objval,  // the edge-cut or the total communication volume of
      // the partitioning solution
      part);
  std::cout << "kaffpa concludes" << std::endl;
  return part_arr;
  // return an array of 0 elements to indicate the error.
  //return aten::NullArray();
}

#endif  // !defined(_WIN32)

DGL_REGISTER_GLOBAL("partition._CAPI_DGLKaHIPPartition_Hetero")
    .set_body([](DGLArgs args, DGLRetValue *rv) {
      HeteroGraphRef g = args[0];
      auto hgptr = std::dynamic_pointer_cast<HeteroGraph>(g.sptr());
      CHECK(hgptr) << "Invalid HeteroGraph object";
      CHECK_EQ(hgptr->relation_graphs().size(), 1)
          << "KaHIP partition only supports HomoGraph";
      auto ugptr = hgptr->relation_graphs()[0];
      int k = args[1];
      NDArray vwgt = args[2];
#if !defined(_WIN32)
      *rv = KaHIPPartition(ugptr, k, vwgt);
#else
      LOG(FATAL) << "KaHIP partition does not support Windows.";
#endif  // !defined(_WIN32)
    });
}  // namespace transform
}  // namespace dgl
