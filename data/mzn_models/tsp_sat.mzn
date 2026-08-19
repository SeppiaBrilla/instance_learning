include "globals.mzn";
int: n;
array[1..n, 1..n] of int: dist;
int: min_dist;
int: max_dist;
array[1..n] of var 1..n: x;
var int: cost;
array [1..n] of var min_dist..max_dist: d;
int: sol;

constraint cost = sum(d);

constraint forall(i in 1..n)(
    d[i] = dist[i, x[i]]
);

constraint circuit(x);

constraint cost <= sol;

solve satisfy;
