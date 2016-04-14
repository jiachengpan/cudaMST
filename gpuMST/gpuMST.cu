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

const int BlockSize = 128;

using namespace std;

__global__
void init_Edges(wghEdge<intT> *input, intT size, intT *u, intT *v, double *w, intT *id) {
  const int pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    wghEdge<intT> e = input[pos];
    u[pos] = e.u;
    v[pos] = e.v;
    w[pos] = e.weight;
    id[pos] = pos;

    u[pos+size] = e.v;
    v[pos+size] = e.u;
    w[pos+size] = e.weight;
    id[pos+size] = pos;
  }
}


struct Edges {
  thrust::device_vector<intT> u;
  thrust::device_vector<intT> v;
  thrust::device_vector<intT> id;
  thrust::device_vector<double> w;

  intT n_edges;
  intT n_vertices;

  Edges() { }

  Edges(const wghEdgeArray<intT>& G) :
    u(G.m*2), v(G.m*2), id(G.m*2), w(G.m*2), n_edges(G.m*2), n_vertices(G.n) { 
    thrust::device_vector<wghEdge<intT>> E(G.E, G.E + G.m);

    init_Edges<<<(G.m + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(E.data()), G.m, 
       thrust::raw_pointer_cast(u.data()),
       thrust::raw_pointer_cast(v.data()),
       thrust::raw_pointer_cast(w.data()),
       thrust::raw_pointer_cast(id.data()));
  }

  Edges(intT m, intT n) : u(m), v(m), id(m), w(m), n_edges(m), n_vertices(n) { }
};


template<typename T>
void print_vector(const T& vec, string text) {
  cout << text << endl;
  for (size_t i = 0; i < vec.size() && i < 100; ++i) {
    cout << " " << vec[i];
  }
  cout << endl;
}

//--------------------------------------------------------------------------------
// kernels for mst
//--------------------------------------------------------------------------------
__global__
void remove_circles(intT *input, size_t size, intT* output, intT *aux)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) {
    intT successor   = input[pos];
    intT s_successor = input[successor];

    successor = ((successor > pos) && (s_successor == pos)) ? pos : successor;
    //if ((successor > pos) && (s_successor == pos)) {
    //  successor = pos;
    //}
    aux[pos] = (successor != pos);
    output[pos] = successor;
  }
}

__global__
void merge_vertices(intT *successors, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    bool goon = true;
    int i = 0;

    while (goon && (i++ < 50)) {
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
    const intT *u, const intT *v,
    intT *new_vertices, intT *output, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    // true means the edge will be kept
    output[pos] = (new_vertices[u[pos]] != new_vertices[v[pos]]);
  }
}

__global__
void update_edges_with_new_vertices(
    intT *u, intT *v, intT *new_vertices, size_t size)
{
  const uint32_t pos = threadIdx.x + blockIdx.x * blockDim.x;

  if (pos < size) {
    u[pos] = new_vertices[u[pos]];
    v[pos] = new_vertices[v[pos]];
  }
}

//--------------------------------------------------------------------------------
// functors
//--------------------------------------------------------------------------------
__host__ __device__ bool operator< (const int2& a, const int2& b) {
    return (a.x == b.x) ? (a.y < b.y) : (a.x < b.x);
};

struct binop_tuple_minimum {
  typedef thrust::tuple<double, intT, intT> T; // (w, v, id)
  __host__ __device__ 
  T operator() (const T& a, const T& b) const {
    return (thrust::get<0>(a) == thrust::get<0>(b)) ? 
      ((thrust::get<1>(a) < thrust::get<1>(b)) ? a : b) :
      ((thrust::get<0>(a) < thrust::get<0>(b)) ? a : b);
  }
};

//--------------------------------------------------------------------------------
// GPU MST
//--------------------------------------------------------------------------------
void recursive_mst_loop(
    Edges& edges,
    thrust::device_vector<intT>&    mst_edges,
    intT &n_mst)
{
  size_t n_edges = edges.n_edges;
  size_t n_vertices = edges.n_vertices;

  thrust::device_vector<intT> succ(n_vertices);
  thrust::device_vector<intT> succ_id(n_vertices);
  thrust::device_vector<intT> succ_indices(n_vertices);
  thrust::device_vector<intT> succ_temp(n_vertices);

  thrust::device_vector<int>  indices(n_edges);
  thrust::device_vector<int>  flags(n_edges);
  Edges edges_temp(edges.n_edges, edges.n_vertices);

  while (1) {
    if (n_edges == 1) {
      mst_edges[n_mst++] = edges.id[0];
      return;
    }

    thrust::sequence(indices.begin(), indices.begin() + n_edges);
    thrust::sort_by_key(edges.u.begin(), edges.u.begin() + n_edges, indices.begin());

    thrust::gather(indices.begin(), indices.begin() + n_edges, 
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.v.begin(), edges.w.begin(), edges.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin(), edges_temp.w.begin(), edges_temp.id.begin())));

    edges_temp.v.swap(edges.v);
    edges_temp.w.swap(edges.w);
    edges_temp.id.swap(edges.id);

    auto new_last = thrust::reduce_by_key(
        edges.u.begin(), edges.u.begin() + n_edges,
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.w.begin(), edges.v.begin(), edges.id.begin())),
        edges_temp.u.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.w.begin(), edges_temp.v.begin(), edges_temp.id.begin())),
        thrust::equal_to<intT>(),
        binop_tuple_minimum());

    size_t n_min_edges = new_last.first - edges_temp.u.begin();

    //cout << "n_min_edges: " << n_min_edges << endl;

    thrust::sequence(succ_indices.begin(), succ_indices.begin() + n_vertices);
    thrust::scatter(
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin(), edges_temp.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.v.begin() + n_min_edges, edges_temp.id.begin() + n_min_edges)),
        edges_temp.u.begin(),
        thrust::make_zip_iterator(thrust::make_tuple(
            succ_indices.begin(), succ_id.begin())));

    // succ_tmp stores which succ are to be saved (1)/ dumped
    remove_circles<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ_indices.data()), n_vertices,
       thrust::raw_pointer_cast(succ.data()),
       thrust::raw_pointer_cast(succ_temp.data()));

    thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices, 
        succ_indices.begin());
    // save new mst edges
    thrust::scatter_if(succ_id.begin(), succ_id.begin() + n_vertices,
        succ_indices.begin(), succ_temp.begin(), mst_edges.begin() + n_mst);

    n_mst += succ_indices[n_vertices-1] + succ_temp[n_vertices-1];

    //cout << "n_mst: " << n_mst << endl;

    // generating super vertices (new vertices)
    thrust::sequence(succ_indices.begin(), succ_indices.begin() + n_vertices);
    merge_vertices<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()), n_vertices);

    thrust::sort_by_key(succ.begin(), succ.begin() + n_vertices, succ_indices.begin());

    mark_segments<<<(n_vertices + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(succ.data()),
       thrust::raw_pointer_cast(succ_temp.data()), n_vertices);

    // new_vertices stored for subsequent calls to do query about next-vertice id
    thrust::device_vector<intT>& new_vertices = succ;
    thrust::exclusive_scan(succ_temp.begin(), succ_temp.begin() + n_vertices, 
        succ_id.begin());
    thrust::scatter(succ_id.begin(), succ_id.begin() + n_vertices, 
        succ_indices.begin(), new_vertices.begin());

    intT new_vertice_size = succ_id[n_vertices-1] + succ_temp[n_vertices-1];

    //cout << "new_vertice_size: " << new_vertice_size << endl;

    // generating new edges
    mark_edges_to_keep<<<(n_edges + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges.u.data()),
       thrust::raw_pointer_cast(edges.v.data()),
       thrust::raw_pointer_cast(new_vertices.data()),
       thrust::raw_pointer_cast(flags.data()), n_edges);
    thrust::exclusive_scan(flags.begin(), flags.begin() + n_edges, 
        indices.begin());

    intT new_edge_size = indices[n_edges-1] + flags[n_edges-1];
    if (!new_edge_size) { return; }

    //cout << "new_edge_size: " << new_edge_size << endl;

    thrust::scatter_if(
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.u.begin(), edges.v.begin(), edges.w.begin(), edges.id.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(
            edges.u.begin() + n_edges, edges.v.begin() + n_edges, edges.w.begin() + n_edges, edges.id.begin() + n_edges)),
        indices.begin(), flags.begin(), 
        thrust::make_zip_iterator(thrust::make_tuple(
            edges_temp.u.begin(), edges_temp.v.begin(), edges_temp.w.begin(), edges_temp.id.begin()))
        );

    update_edges_with_new_vertices<<<(new_edge_size + BlockSize - 1) / BlockSize, BlockSize>>>
      (thrust::raw_pointer_cast(edges_temp.v.data()),
       thrust::raw_pointer_cast(edges_temp.u.data()),
       thrust::raw_pointer_cast(new_vertices.data()), new_edge_size);

    edges.u.swap(edges_temp.u);
    edges.v.swap(edges_temp.v);
    edges.w.swap(edges_temp.w);
    edges.id.swap(edges_temp.id);

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

  Edges edges(G);
  thrust::device_vector<intT> mst_edges(G.m);

  nextTime("prepare graph");

  intT mst_size = 0;
  recursive_mst_loop(edges, mst_edges, mst_size);

  intT *result_mst_edges = new intT[mst_size];
  cudaMemcpy(result_mst_edges, thrust::raw_pointer_cast(mst_edges.data()),
      sizeof(intT) * mst_size, cudaMemcpyDeviceToHost);

  return make_pair(result_mst_edges, mst_size);
}

