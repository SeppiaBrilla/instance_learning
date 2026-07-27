% Instance Data
int: nc; % Number of colors
int: nv; % Number of vertices
int: ne; % Number of edges

array[1..ne, 1..2] of int: edges;

% Model
array[1..nv] of var 1..nc: color;

constraint forall(e in 1..ne) (
    color[edges[e,1]] != color[edges[e,2]]
);

solve satisfy;
