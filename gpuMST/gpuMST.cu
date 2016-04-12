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

void print_edges(const thrust::device_vector<edgei>& vec, const thrust::device_vector<double>& weights, size_t size, string text) {
  cout << text << endl;
  for (size_t i = 0; i < size; ++i) {
    edgei e = vec[i];
    cout << " " << e.id << " " << e.u << " " << e.v << " " << weights[i] << endl;
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
void scatter_successors(edgei *input, intT *output, intT* output_idx, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    edgei edge = input[pos];
    output[edge.u] = edge.v;
    output_idx[edge.u] = edge.id;
  }
}

__global__
void remove_circles(intT *successors, size_t size, intT *aux)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    intT successor   = successors[pos];
    intT s_successor = successors[successor];

    if ((successor > pos) && (s_successor == pos)) {
      successors[pos] = pos;
      successor = pos;
    }
    aux[pos] = (successor != pos);
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
      __syncthreads();

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

struct binop_sort_by_edge {
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
    return (thrust::get<1>(a) == thrust::get<1>(b)) ? 
      ((thrust::get<0>(a).v < thrust::get<0>(b).v) ? a : b) :
      ((thrust::get<1>(a) < thrust::get<1>(b) ? a : b));
  }
};

struct unary_get_edge_id {
  __host__ __device__ __forceinline__
  intT operator() (const edgei& edge) const {
    return edge.id;
  }
};

void recursive_mst_loop(
    thrust::device_vector<edgei>&   edges,
    thrust::device_vector<double>&  edge_weights,
    intT n_vertices,
    thrust::device_vector<intT>&    mst_edges,
    intT &n_mst)
{
  size_t n_edges = edges.size();
  thrust::device_vector<edgei>  edges_temp(n_edges);
  thrust::device_vector<double> edge_weights_temp(n_edges);

  thrust::device_vector<edgei>  temp(n_edges);
  thrust::device_vector<intT>   edges_aux(n_edges);
  thrust::device_vector<intT>   edges_idx(n_edges);

  thrust::device_vector<intT>   succ(n_vertices);
  thrust::device_vector<intT>   succ_idx(n_vertices);
  thrust::device_vector<intT>   succ_aux(n_vertices);
  thrust::device_vector<intT>   succ_tmp(n_vertices);

  while (1) {
    if (n_edges == 1) {
      edgei edge = edges[0];
      mst_edges[n_mst++] = edge.id;
      //cout << "MST: " << n_mst << endl;
      print_vector(mst_edges, "mst_edges");
      return;
    }
    thrust::sort_by_key(edges.begin(), edges.begin()+n_edges, edge_weights.begin(),
        binop_sort_by_edge());

    //print_edges(edges, edge_weights, n_edges, "edges: ");
    auto new_last = thrust::reduce_by_key(
        edges.begin(), edges.begin()+n_edges,
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.begin(), edge_weights.begin())),
        temp.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.begin(), edge_weights_temp.begin())),
        binop_equal_to(),
        binop_tuple_minimum());

    size_t n_min_edges = new_last.first - temp.begin();

    //print_edges(edges_temp, edge_weights_temp, n_edges, "edges temp: ");

    thrust::sequence(succ.begin(), succ.begin()+n_vertices);

    scatter_successors<<<(n_min_edges + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges_temp.data()),
       thrust::raw_pointer_cast(succ.data()),
       thrust::raw_pointer_cast(succ_idx.data()), n_min_edges);

    // succ_tmp stores which succ are to be saved (1)/ dumped
    remove_circles<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()), n_vertices,
       thrust::raw_pointer_cast(succ_tmp.data()));
    thrust::exclusive_scan(succ_tmp.begin(), succ_tmp.begin()+n_vertices, succ_aux.begin());
    // save new mst edges
    thrust::scatter_if(succ_idx.begin(), succ_idx.begin()+n_vertices,
        succ_aux.begin(), succ_tmp.begin(), mst_edges.begin()+n_mst);
    n_mst += succ_aux[n_vertices-1]+1;

    //print_vector(succ_idx, "succ_idx");
    //print_vector(succ_aux, "succ_aux target");
    //print_vector(succ_tmp, "succ_tmp stencil");
    //print_vector(mst_edges, "mst_edges");
    //cout << "n_mst: " << n_mst << endl;


    // generating super vertices (new vertices)
    thrust::sequence(succ_idx.begin(), succ_idx.begin()+n_vertices);
    merge_vertices<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()), n_vertices);

    thrust::sort_by_key(succ.begin(), succ.begin()+n_vertices, succ_idx.begin());

    mark_segments<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()),
       thrust::raw_pointer_cast(succ_tmp.data()), n_vertices);

    // new_vertices stored for subsequent calls to do query about next-vertice id
    thrust::device_vector<intT>& new_vertices = succ;
    thrust::exclusive_scan(succ_tmp.begin(), succ_tmp.begin()+n_vertices, succ_aux.begin());
    thrust::scatter(succ_aux.begin(), succ_aux.begin()+n_vertices, 
        succ_idx.begin(), new_vertices.begin());

    intT new_vertice_size = succ_aux[n_vertices-1] + succ_tmp[n_vertices-1];

    //cout << "new_vertice_size: " << new_vertice_size << endl;

    // generating new edges
    mark_edges_to_keep<<<(n_edges + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges.data()),
       thrust::raw_pointer_cast(new_vertices.data()),
       thrust::raw_pointer_cast(edges_aux.data()), n_edges);
    thrust::exclusive_scan(edges_aux.begin(), edges_aux.begin()+n_edges, edges_idx.begin());

    intT new_edge_size = edges_idx[n_edges-1] + edges_aux[n_edges-1];
    if (!new_edge_size) { return; }

    //cout << "new_edge_size: " << new_edge_size << endl;

    thrust::scatter_if(
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.begin(), edge_weights.begin())),
      thrust::make_zip_iterator(thrust::make_tuple(
          edges.begin()+n_edges, edge_weights.begin()+n_edges)),
      edges_idx.begin(),
      edges_aux.begin(), 
      thrust::make_zip_iterator(thrust::make_tuple(
          edges_temp.begin(), edge_weights_temp.begin())));

    update_edges_with_new_vertices<<<(new_edge_size + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges_temp.data()),
       thrust::raw_pointer_cast(new_vertices.data()), new_edge_size);

    edges.swap(edges_temp);
    edge_weights.swap(edge_weights_temp);
    n_vertices = new_vertice_size;
    n_edges = new_edge_size;
  }
}

//--------------------------------------------------------------------------------
// top level mst
//--------------------------------------------------------------------------------
std::pair<intT*,intT> mst(wghEdgeArray<intT> G)
{
  startTime();
  vector<edgei> h_edges(G.m<<1);
  vector<double> h_weights(G.m<<1);
  for (size_t i = 0; i < G.m; ++i) {
    h_edges[i*2]    = edgei(G.E[i].v, G.E[i].u, i);
    h_edges[i*2+1]  = edgei(G.E[i].u, G.E[i].v, i);
    h_weights[i*2]  = G.E[i].weight;
    h_weights[i*2+1]= G.E[i].weight;
  }

  thrust::device_vector<edgei>  edges(h_edges.begin(), h_edges.end());
  thrust::device_vector<double> edge_weights(h_weights.begin(), h_weights.end());
  thrust::device_vector<intT>   mst_edges(G.m);

  nextTime("prepare graph");

  intT mst_size = 0;
  recursive_mst_loop(edges, edge_weights, G.n, mst_edges, mst_size);

  intT *result_mst_edges = new intT[mst_size];
  cudaMemcpy(result_mst_edges, thrust::raw_pointer_cast(mst_edges.data()),
      sizeof(intT) * mst_size, cudaMemcpyDeviceToHost);

  nextTime("MST");
  return make_pair(result_mst_edges, mst_size);
}
