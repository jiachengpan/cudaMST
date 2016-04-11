#include <iostream>
#include <algorithm>
#include <cstdint>

#include "graph.cuh"
#include "gettime.h"
#include "MST.h"
#include "parallel.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>

const int BlockSize = 256;

using namespace std;

struct edgei {
  intT u;
  intT v;
  intT id;
  __host__ __device__
  edgei() {}
  __host__ __device__
  edgei(intT _u, intT _v, intT _id) 
    : u(_u), v(_v), id(_id) {}
};


template<typename T>
void print_vector(const T& vec, string text) {
  cout << text << endl;
  for (size_t i = 0; i < vec.size(); ++i) {
    cout << " " << vec[i];
  }
  cout << endl;
}

void print_edges(const thrust::device_vector<edgei>& vec, size_t size, string text) {
  cout << text << endl;
  for (size_t i = 0; i < size; ++i) {
    edgei e = vec[i];
    cout << " " << e.u << " " << e.v << /*" " << e.weight <<*/ endl;
  }
  cout << endl;
}

//--------------------------------------------------------------------------------
// kernels for mst
//--------------------------------------------------------------------------------
__global__
void transform_edge_to_edgei(
    wghEdge<intT> *input,
    edgei *output, double *output_weight, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    wghEdge<intT> edge = input[pos];
    output[pos] = edgei(edge.u, edge.v, pos);
    output_weight[pos] = edge.weight;
  }

}

__global__
void extract_sources(wghEdge<intT> *input, intT *output, intT size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    output[pos] = input[pos].u;
  }
}

__global__
void scatter_successors(edgei *input, intT *output, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    edgei edge = input[pos];
    output[edge.u] = edge.v;
  }
}

__global__
void remove_circles(intT *successors, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    intT successor   = successors[pos];
    intT s_successor = successors[successor];

    if ((successor > pos) && (s_successor != successor)) {
      successors[pos] = pos;
    }
  }
}

__global__
void merge_vertices(intT *successors, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    bool goon = true;
    while (goon) {
      intT successor = successors[pos];
      intT ssuccessor= successors[successor];

      if (ssuccessor != successor) {
        successors[pos] = ssuccessor;
      }
      goon = __any(ssuccessor != successor);
      __syncthreads();
    }
  }
}

__global__
void mark_segments(intT *input, intT *output, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    output[pos] = ((pos == size-1) || (input[pos] != input[pos+1]));
  }
}

__global__
void mark_edges_to_keep(
    const edgei *edges, 
    intT *new_vertices, intT *output, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    edgei edge = edges[pos];
    // true means the edge will be kept
    output[pos] = (new_vertices[edge.u] != new_vertices[edge.v]);
  }
}

__global__
void update_edges_with_new_vertices(
    edgei *edges, intT *new_vertices, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    edgei edge = edges[pos];
    edge.u = new_vertices[edge.u];
    edge.v = new_vertices[edge.v];
    edges[pos] = edge;
  }
}

//--------------------------------------------------------------------------------
// functors
//--------------------------------------------------------------------------------

struct binop_sort_by_edge_src {
  __host__ __device__ __forceinline__
  bool operator() (const edgei& a, const edgei& b) const {
    return (a.u == b.u) ? (a.v < b.v) : (a.u < b.u);
  }
};

struct binop_equal_to {
  __host__ __device__ __forceinline__
  bool operator() (const edgei& a, const edgei& b) const {
    return (a.u == b.u);
  }
};

struct binop_tuple_minimum {
  typedef thrust::tuple<edgei, double> T;
  __host__ __device__
  T operator() (const T& a, const T& b) const {
    return (thrust::get<1>(a) < thrust::get<1>(b)) ? a : b;
  }
};

struct unary_get_edge_id {
  __host__ __device__ __forceinline__
  intT operator() (const edgei& edge) const {
    return edge.id;
  }
};

void recursive_mst(
    const thrust::device_vector<edgei>& edges,
    const thrust::device_vector<double>& edge_weights,
    intT n,
    thrust::device_vector<intT>& mst_edges,
    intT& mst_size)
{
  if (edges.empty()) return;
  if (edges.size() == 1) {
    edgei edge = edges[0];
    mst_edges[mst_size++] = edge.id;
    return;
  }

  //cout << "edges: " << edges.size() << endl;
  thrust::device_vector<edgei> new_edges(edges.size());
  thrust::device_vector<double> new_edge_weights(edges.size());
  thrust::device_vector<edgei> new_edges_temp(edges.size());

  //print_edges(edges, edges.size(), "edges");
  //print_vector(edge_weights, "edge_weights");

  // new_edges here refer to the edges with minimum weight in its segment (segmented by u)
  auto new_last = thrust::reduce_by_key(
      edges.begin(), edges.end(), 
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.begin(), edge_weights.begin())),
      new_edges_temp.begin(), 
      thrust::make_zip_iterator(thrust::make_tuple(
          new_edges.begin(), new_edge_weights.begin())),
      binop_equal_to(),
      binop_tuple_minimum());

  size_t forest_size = new_last.first - new_edges_temp.begin();
  //cout << "forest_size: " << forest_size << endl;
  // save mst edges
  thrust::transform(
      new_edges.begin(), new_edges.begin() + forest_size,
      mst_edges.begin() + mst_size,
      unary_get_edge_id());
  mst_size += forest_size;

  //print_edges(new_edges, forest_size, "new edges");
  //print_vector(new_edge_weights, "new edge_weights");

  thrust::counting_iterator<intT> first(0);
  thrust::device_vector<intT> successors(first, first+n);
  thrust::device_vector<intT> successor_idx(first, first+n);
  thrust::device_vector<intT> new_vertices(n);

  scatter_successors<<<(forest_size + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(new_edges.data()),
     thrust::raw_pointer_cast(successors.data()), forest_size);

  //print_vector(successors, "successors");

  // we dont need to remove circles as long as we get an edge list
  // which doesnt have duplicate of the same undirected edge
  remove_circles<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(successors.data()), n);
  merge_vertices<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(successors.data()), n);
  //print_vector(successors, "merged successors");
  
  thrust::sort_by_key(successors.begin(), successors.end(), successor_idx.begin());
  //print_vector(successor_idx, "sorted successors_idx");
  //print_vector(successors, "sorted successors");

  mark_segments<<<(n + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(successors.data()),
     thrust::raw_pointer_cast(new_vertices.data()), n);
  //print_vector(new_vertices, "marked segments");

  thrust::exclusive_scan(new_vertices.begin(), new_vertices.end(), successors.begin());
  //print_vector(successors, "shuffled new vertices");

  intT new_vertices_size = 
    new_vertices[new_vertices.size()-1] + successors[successors.size()-1];
  //cout << "new_vertices_size: " << new_vertices_size << endl;

  thrust::scatter(successors.begin(), successors.end(),
      successor_idx.begin(), new_vertices.begin());

  //print_vector(new_vertices, "new_vertices");

  thrust::device_vector<intT> new_edge_idx(edges.size());
  thrust::device_vector<intT> new_edge_keep(edges.size());

  // new_edges here refer to the new edges for the next recursion
  mark_edges_to_keep<<<(edges.size() + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(edges.data()),
     thrust::raw_pointer_cast(new_vertices.data()),
     thrust::raw_pointer_cast(new_edge_keep.data()), edges.size());

  //print_vector(new_edge_keep, "new edges keep");

  thrust::exclusive_scan(new_edge_keep.begin(), new_edge_keep.end(), new_edge_idx.begin());

  //print_vector(new_edge_idx, "new edges idx");
  intT new_edge_size = new_edge_idx[edges.size()-1] + new_edge_keep[edges.size()-1];

  //cout << "new_edge_size: " << new_edge_size << endl;
  if (!new_edge_size) { return; }

  thrust::scatter_if(
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.begin(), edge_weights.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.end(), edge_weights.end())),
      new_edge_idx.begin(),
      new_edge_keep.begin(), 
      thrust::make_zip_iterator(thrust::make_tuple(
          new_edges.begin(), new_edge_weights.begin())));

  update_edges_with_new_vertices<<<(new_edge_size + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(new_edges.data()),
     thrust::raw_pointer_cast(new_vertices.data()), new_edge_size);

  //for (int i = 0; i < new_edge_size; ++i) {
  //  edgei e = new_edges[i];
  //  double w = new_edge_weights[i];
  //  printf(" (%d, %d) %lf\n", e.u, e.v, w);
  //}
  //cout << endl;

  new_edges.resize(new_edge_size);
  new_edge_weights.resize(new_edge_size);
  recursive_mst(
      new_edges, new_edge_weights,
      new_vertices_size,
      mst_edges,
      mst_size);
}


//--------------------------------------------------------------------------------
// top level mst
//--------------------------------------------------------------------------------
std::pair<intT*,intT> mst(wghEdgeArray<intT> G)
{
  startTime();

  thrust::device_vector<wghEdge<intT>> input_edges(G.E, G.E+G.m);
  thrust::device_vector<edgei> edges(G.m);
  thrust::device_vector<double> edge_weights(G.m);
  thrust::device_vector<intT> mst_edges(G.m);

  transform_edge_to_edgei<<<(G.m + BlockSize - 1) / BlockSize, BlockSize>>>
    (thrust::raw_pointer_cast(input_edges.data()),
     thrust::raw_pointer_cast(edges.data()),
     thrust::raw_pointer_cast(edge_weights.data()), G.m);

  thrust::sort_by_key(edges.begin(), edges.end(), edge_weights.begin(),
      binop_sort_by_edge_src());

  nextTime("SortGraph");

  intT mst_size = 0;
  recursive_mst(edges, edge_weights, G.n, mst_edges, mst_size);

  intT *result_mst_edges = new intT[mst_size];
  cudaMemcpy(result_mst_edges, thrust::raw_pointer_cast(mst_edges.data()),
      sizeof(intT) * mst_size, cudaMemcpyDeviceToHost);

  //thrust::host_vector<intT> result_mst_edges(mst_edges.begin(), mst_edges.begin()+mst_size);
  //
  nextTime("MST");
  return make_pair(result_mst_edges, mst_size);
}
