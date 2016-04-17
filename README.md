CUDA MST
========

This repo implements *Chapter 7 Fast Minimum Spanning Tree Computation* of *GPU Computing Gems*, a data-parallelism MST based on Bruvka's algorithm.

This is originally part of the [benchmark suite](http://www.cs.cmu.edu/~pbbs/benchmarks.html).
[Thrust](http://thrust.github.io/) is heavily used.

-----

The baselines are:
* serial Kruskal's algorithm with union find with weight and path-compression
* parallel Kruskal's algorithm

The compare flow is *gpuMST* which is my implementation of the aforementioned chapter 7.

Result shows that on a GTX750ti & E1231v2 machine, on-par performance is achieved on random graphs, whereas better performances can be observed with sparse (grid) graph and scale-free graphs.

##### gpuMST result

    1 : randLocalGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile765486_563367 : '0.759'
    1 : rMatGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile138860_218538 : '0.655'
    1 : 2Dgrid_WE_2000000 :  -r 1 -o /tmp/ofile8903_852545 : '0.159'
    gpuMST : 0 : weighted time, min=0.524 median=0.524 mean=0.524


##### serialMST result

    1 : randLocalGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile399034_715347 : '0.65'
    1 : rMatGraph_WE_5_2000000 :  -r 1 -o /tmp/ofile477678_439826 : '0.697'
    1 : 2Dgrid_WE_2000000 :  -r 1 -o /tmp/ofile983504_141272 : '0.395'
    serialMST : 0 : weighted time, min=0.58 median=0.58 mean=0.58


(I think I can do better... later...)
