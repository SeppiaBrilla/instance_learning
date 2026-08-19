int: n; % number of objects
set of int: OBJ = 1..n;
int: capacity;
array[OBJ] of int: profit;
array[OBJ] of int: size;
int: desired_profit;

array[OBJ] of var 0..1: x; % how many of each object
var int: total_profit; 

constraint sum(i in OBJ)(size[i] * x[i]) <= capacity;
constraint total_profit = sum(i in OBJ)(profit[i] * x[i]);
constraint total_profit > desired_profit;
solve satisfy;