int: n; % number of objects
set of int: OBJ = 1..n;
int: capacity;
array[OBJ] of int: profit;
array[OBJ] of int: size;

array[OBJ] of var 0..1: x; % how many of each object
var int: total_profit; 

constraint sum(i in OBJ)(size[i] * x[i]) <= capacity;
constraint total_profit = sum(i in OBJ)(profit[i] * x[i]);
solve maximize total_profit;

output ["p = " ++ show(total_profit) ++ "\n"];