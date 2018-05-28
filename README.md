# Overview #
Implemented BFS, DFS, A* search algorithms to solve an 8 board sliding puzzle and recorded relevant statistics for time and memory complexity in order to assess algorithm performance. Refer to https://en.wikipedia.org/wiki/Sliding_puzzle for an overview of the sliding puzzle game.

# Theory #
Breadth-first search (BFS) and depth-first search (DFS) are two instances of uninformed search algorithms, while A* search is an informed search algorithm that uses a heuristic to estimate the total cost from the current state to the goal state. The heuristic allows the algorithm to be more selective when choosing the next state to explore and to explore fewer nodes overall in the search tree.

All three algorithms maintain an "explored" dictionary that will keep track of boards that have already been seen in order to prevent a board from being explored more than once. Additionally, all three algorithms will maintain a "frontier" of boards that are reachable from a board that has already been explored, but they themselves have yet to be explored. The choice of data structure used to represent the frontier is a primary driver of algorithm performance. 

## Breadth-first search ##
BFS uses a queue for its frontier and therefore proceeds level by level through the search tree. The memory and time complexity of this algorithm are both O(b<sup>d</sup>) where b is the branching factor of the search tree and d is the earliest level of the search tree at which a goal state is found. BFS will return an optimal path (lowest cost) if the path costs are uniform, which is true for the sliding puzzle problem. If the path costs were not uniform, then another algorithm such as Djikstra's algorithm or A* would have to be used in order to guarantee an optimal solution.

<p align="center">
  <img width="500" height="300" src="https://s3.amazonaws.com/sliding-puzzle/bfs_visual.png">
</p>

## Depth-first search ##
Unlike BFS, DFS cannot guarantee an optimal path. It uses a stack for its frontier and therefore proceeds branch by branch in the search tree to its maximum depth. DFS also has worse time complexity than BFS with O(b<sup>m</sup>) where b is again the branching factor of the search tree and m is the maximum depth of the search tree. However, DFS offers better memory performance than BFS with only O(bd). 

<p align="center">
  <img width="500" height="300" src="https://s3.amazonaws.com/sliding-puzzle/dfs_visual.png">
</p>

## A* Search ##
A* search is an informed search algorithm that uses a priority queue for its frontier. When selecting the next board to explore, the algorithm looks at the total cost up to that point and the expected cost to reach the goal state for each board. The heuristic used for this problem was the Manhatten distance between a given board and the goal state. In choosing a heuristic, a common technique is to relax the conditions of the problem. 

In this case, using Manhatten distance as the expected cost to reach the goal acts as though the player can pluck tiles from board and move them overhead from row to row and column to column and place them their final position regardless of whether another tile already occupies that spot. It ignores the complexity that comes with having to manuever tiles around one another and therefore will almost always understate the cost to reach the goal. However, as long as the heuristic never overstates the cost of reaching the goal from a given position, it will always provide an optimal solution. 

Stated another way, an admissible heuristic must be optimistic. Heuristics that closely approximate the actual cost of reaching the goal from a given state will allow the algorithm to explore fewer nodes overall. All else equal this will result in an optimal solution reached in less time, but one must consider the time it takes to compute the value of a heuristic function.

<p align="center">
  <img width="500" height="300" src="https://s3.amazonaws.com/sliding-puzzle/astar_visual.png">
</p>

# Output
When executed, the program will write to a file called output.txt, containing the following statistics:

	1. method: the search algorithm used in this instance
	
	2. path_to_goal: the series of actions taken to get from the starting board to the goal
	
	3. path_length: the number of moves in path_to_goal
	
	4. nodes_expanded: the total number of nodes that were explored
	
	5. max_search_depth: the maximum depth of the search tree reached in the lifetime of the algorithm
	
	6. max_frontier_size: the maximum number of nodes in the frontier reached in the lifetime of the algorithm
	
	7. running_time: the total time taken (in seconds) by the algorithm to find the goal 
	
