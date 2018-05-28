# import sys
import numpy as np
# import copy
import heapq
from collections import deque
from collections import OrderedDict
import time
import math
# import ast
# import resource

class Board:
    
    def __init__(self, board, current_score=0):
        """
        Board: 2d numpy array representing game board
        Goal: 2d numpy array representing end of game board
        Current Score: metric used to determine board quality
        """
        self.board = board
        self.goal = np.arange(self.board.shape[0]**2).reshape(self.board.shape[0], self.board.shape[0])
        self.current_score = current_score
        
    def __eq__(self, other):
        """
        Override equality operator to just check for same board
        """
        return np.array_equal(self.board, other.board)
    
    def __ne__(self, other):
        """
        Override equality operator to just check for different board
        """
        return not np.array_equal(self.board, other.board)
    
    def get_board(self):
        """
        Returns Game Board (2d numpy array) 
        """
        return self.board
    
    def get_goal(self):
        """
        Returns Goal
        """
        return self.goal
        
    def get_current_score(self):
        """
        Returns score of board 
        """
        return self.current_score
        
    def goal_test(self):
        """
        Determines whether given board is a finished board
        Input: board to test
        Output: True or False
        """
        return np.array_equal(self.board, self.goal)

    def find_num(self, board, num):
        """
        Gives the row and column numbers for the location of the element
        Helper function used in score_board function to determine board quality,
        as well as finding the row and column of the 0 in the board
        Input: board (2d numpy array), desired element
        Output: row and column values (integers) of the element
        """
        row, col = map(int, np.where(board==num))
        return (row, col)

    def string_board(self):
        """
        Gives board as a string representation, so it can be used as a key in a dict
        Input: board (2d numpy array)
        Output: string representation of board
        """
        sb = ''.join([''.join(item) for item in self.board.astype(str)])
        return sb

    def score_board(self):
        """
        Calculates and returns score of board for heuristic (Manhattan distance)
        """
        score = 0
        # for every tile, determine how far away it is from its position in the goal board
        for i in range(1, self.board.shape[0]**2):
            board_row, board_col = self.find_num(self.board, i)
            goal_row, goal_col = self.find_num(self.goal, i)
            score += abs(board_row - goal_row) + abs(board_col - goal_col)
        return score
    
    def move_up(self):
        """
        Gives new board (copy) after moving the 0 up if possible
        Input: board
        Output: new board (copy) after up move, or False if not possible
        """
        # find the row and column of the 0 tile
        row, col = self.find_num(self.board, 0)
        if row > 0:
            new_board = np.copy(self.board)
            # swap the 0 with the number above it
            new_board[row][col], new_board[row-1][col] = new_board[row-1][col], new_board[row][col]
            return Board(new_board, self.get_current_score() + 1)
        return False
    
    
    def move_down(self):
        """
        Gives new board (copy) after moving the 0 down if possible
        Input: board
        Output: new board (copy) after down move, or False if not possible
        """
        row, col = self.find_num(self.board, 0)
        if row < self.board.shape[0] - 1:
            new_board = np.copy(self.board)
            # swap the 0 with the number below it 
            new_board[row][col], new_board[row+1][col] = new_board[row+1][col], new_board[row][col]
            return Board(new_board, self.get_current_score() + 1)
        return False
        

    def move_left(self):
        """
        Gives new board (copy) after moving the 0 left if possible
        Input: board
        Output: new board (copy) after left move, or False if not possible
        """
        row, col = self.find_num(self.board, 0)
        if col > 0:
            new_board = np.copy(self.board)
            # swap the 0 with the number to its left
            new_board[row][col], new_board[row][col-1] = new_board[row][col-1], new_board[row][col]
            return Board(new_board, self.get_current_score() + 1)
        return False
    
    
    def move_right(self):
        """
        Gives new board (copy) after moving the 0 right if possible
        Input: board
        Output: new board (copy) after right move, or False if not possible
        """
        row, col = self.find_num(self.board, 0)
        if col < self.board.shape[1] - 1:
            new_board = np.copy(self.board)
            # swap the 0 with the number to its left
            new_board[row][col], new_board[row][col+1] = new_board[row][col+1], new_board[row][col]
            return Board(new_board, self.get_current_score() + 1)
        return False


def retrieve_state(method, frontier):
    """
    Returns item from frontier according to type of data structure
    Input: frontier and method of search
    Output: next state from frontier
    """
    # breadth-first search
    if method == 'bfs':
        return frontier.popleft()
    
    # depth-first search
    if method == 'dfs':
        return frontier.pop()
    
    # A* search
    if method == 'ast':
        return heapq.heappop(frontier)[2]


def initialize_frontier(method, board):
    """
    Initializes frontier depending on search method and returns frontier
    Input: search method and starting board
    Output: frontier
    """
    # use double-ended queue (operations will provide queue functionality) for bfs
    if method == 'bfs':
        frontier = deque()
        frontier.append(board)
    
    # use double-ended queue (operations will provide stack functionality) for dfs
    if method == 'dfs':
        frontier = deque()
        frontier.append(board)
    
    # use priority queue for A* search
    if method == 'ast':
        frontier = []
        heapq.heappush(frontier, (0, 0, board))
        
    return frontier
    
    
def update_frontier(method, state, parents, frontier, frontier_dict, max_search_depth, explored, astar_frontier_dict, nodes_expanded, entry_count):
    """
    Updates frontier for BFS, DFS and A* methods
    Input: search method, current board, dictionary of parent-child relationships, frontier data structure, 
    frontier dictionary for O(1) lookup, current maximum search depth achieved, dictionary of explored boards, 
    special dictionary for A* search, total number of nodes expanded, number of nodes added to frontier (only used for A*)
    Output: number of nodes expanded, current max_search_depth
    Mutated: parents, frontier, frontier_dict, explored, astar_frontier_dict
    """
    # update nodes_expanded
    nodes_expanded += 1
    # add this board to dictionary of explored boards
    explored[state.string_board()] = True
    
    # get all possible moves (or False if move is not possible)
    up_board = state.move_up()
    down_board = state.move_down()
    left_board = state.move_left()
    right_board = state.move_right()
    
    # keep particular order for exploration
    moves_bfs = [up_board, down_board, left_board, right_board]
    moves_dfs = [right_board, left_board, down_board, up_board]
    
    
    if method == 'bfs':
        # remove the board from the frontier
        del frontier_dict[state.string_board()]
        for move in moves_bfs:
            if move:
                ms = move.string_board()
                # check to see if board has already been explored or added to frontier
                if ms not in explored and ms not in frontier_dict:
                    frontier_dict[ms] = True
                    # remember relationship to previous board
                    parents[ms] = state.string_board()
                    # update max_search_depth if applicable
                    if move.get_current_score() > max_search_depth:
                        max_search_depth = move.get_current_score()
                    # add move to frontier
                    frontier.append(move)
        return (nodes_expanded, entry_count, max_search_depth)
    
    if method == 'dfs':
        # operations are similar to bfs above
        del frontier_dict[state.string_board()]
        for move in moves_dfs:
            if move:
                ms = move.string_board()
                if ms not in explored and ms not in frontier_dict:
                    frontier_dict[ms] = True
                    parents[ms] = state.string_board()
                    if move.get_current_score() > max_search_depth:
                        max_search_depth = move.get_current_score()
                    frontier.append(move)
        return (nodes_expanded, entry_count, max_search_depth)
                    
    if method == 'ast':
        del astar_frontier_dict[state.string_board()]
        for move in moves_bfs:
            if move:
                ms = move.string_board()
                if ms not in explored and ms not in astar_frontier_dict:
                    entry_count += 1
                    parents[ms] = state.string_board()
                    if move.get_current_score() > max_search_depth:
                        max_search_depth = move.get_current_score()
                    astar_frontier_dict[ms] = (move.get_current_score(), move.get_current_score() + move.score_board(), entry_count, move)
                    heapq.heappush(frontier, (move.get_current_score() + move.score_board(), entry_count, move))
                # if the board is in the frontier, compare the number of moves made in respective paths to same board
                elif ms in astar_frontier_dict:
                    old_score, old_priority_value, old_entry_count, old_board = astar_frontier_dict[ms]
                    if move.get_current_score() < old_score:
                        entry_count += 1
                        parents[ms] = state.string_board()
                        if move.get_current_score() > max_search_depth:
                            max_search_depth = move.get_current_score()
                        # remove old board from the frontier and replace with new board
                        del astar_frontier_dict[ms]
                        astar_frontier_dict[ms] = (move.get_current_score(), move.get_current_score() + move.score_board(), entry_count, move)
                        # remove old board from the frontier
                        del frontier[frontier.index((old_priority_value, old_entry_count, old_board))]
                        # add new board to frontier and heapify
                        heapq.heappush(frontier, (move.get_current_score() + move.score_board(), entry_count, move))
                        heapq.heapify(frontier)
        return (nodes_expanded, entry_count, max_search_depth)

def string_to_board(string):
    """
    Convert string into Board
    """
    return Board(np.asarray(list(map(int, [char for char in string]))).reshape(int(math.sqrt(len(string))),int(math.sqrt(len(string)))))

def get_direction(state, children):
    """
    Called from evaluate
    Get move that transforms current state to its child
    Input: board and children dictionary
    """
    # lazy checking -> won't continue to second conditional if first is False
    if state.move_up() and state.move_up().string_board() == children[state.string_board()]:
        return 'Up'
    if state.move_down() and state.move_down().string_board() == children[state.string_board()]:
        return 'Down'
    if state.move_left() and state.move_left().string_board() == children[state.string_board()]:
        return 'Left'
    if state.move_right() and state.move_right().string_board() == children[state.string_board()]:
        return 'Right'
    
def evaluate(board, parents):
    """
    Called from search
    Extracts three relevant metrics given parent dictionary from search
    Input: starting board and parents dictionary from search
    Output: path to goal, cost of path and search_depth
    """
    # get the goal board as a string
    goal = Board(board.get_goal()).string_board()
    # initialize dictionary 
    children = OrderedDict()
    children[goal] = None
    # build up children dictionary using parents dictionary
    next_board = parents[goal]
    if next_board:
        children[next_board] = goal
    while next_board:
        if parents[next_board] == None:
            break
        if parents[next_board] != None:
            children[parents[next_board]] = next_board
            next_board = parents[next_board]
    # initialize path_to_goal list
    path_to_goal = []
    # start at starting board and build up path to goal with children dictionary
    current_state = board.string_board()
    while current_state != goal:
        state = string_to_board(current_state)
        path_to_goal.append(get_direction(state, children))
        current_state = children[current_state]
    
    # cost of path is number of moves taken to get to goal
    cost_of_path = len(path_to_goal)
    
    return (path_to_goal, cost_of_path)
    
def search(method, board):
    """
    Performs search (BFS, DFS or A*) on sliding puzzle board to find goal
    Input: starting board and search method
    Output: path to goal, cost of the path, number of nodes expanded, search
    depth to goal, maximum search depth and total time taken
    """
    # get start time
    start_time = time.time()
    # get starting board as a string
    start = board.string_board()
    # initialized max_search_depth for search function
    max_search_depth = 0
    # remember parent-child relationships between boards to get path
    parents = OrderedDict()
    parents[start] = None
    # remember boards that have already been explored to prevent rework
    explored = {}
    # intialize frontier dictionaries for O(1) lookup
    frontier_dict = {start: True}
    # (current # moves made, current # moves made + board score, board)
    # just use 0 for starting board score since it will be only element in heap and automatically retrieved first
    astar_frontier_dict = {start: (0, 0, board)}
    # initialize entry count to be used in comparisons for heap for A*
    entry_count = 0
    # initialize number of nodes_expanded for search function
    nodes_expanded = 0
    # set up frontier based upon search method and add starting board
    frontier = initialize_frontier(method, board)
    # while there are boards in the frontier, continue to explore
    max_frontier_size = 0
    while frontier:
        if len(frontier) > max_frontier_size:
            max_frontier_size = len(frontier)
        # get next board to be explored from frontier
        state = retrieve_state(method, frontier)
        # check to see if board matches the goal board
        if state.goal_test():
            total_time = time.time() - start_time
            # max_ram_usage = resource.getrusage(resource.RUSAGE_SELF)[2]
            # get the path from start to goal, the cost of the path
            path_to_goal, cost_of_path = evaluate(board, parents)
            return (path_to_goal, cost_of_path, nodes_expanded, max_search_depth, total_time, max_frontier_size) 
        nodes_expanded, entry_count, max_search_depth = update_frontier(method, state, parents, frontier, frontier_dict, max_search_depth, explored, astar_frontier_dict, nodes_expanded, entry_count)
    
    
def output(method, path_to_goal, cost_of_path, nodes_expanded, max_search_depth, total_time, max_frontier_size):
    """
    Writes relevant metrics to output file
    """
    with open('output.txt', 'a') as f:
        f.write('method: ' + method + '\n')
        f.write('path_to_goal: ' + str(path_to_goal) + '\n')
        f.write('path_length: ' + str(cost_of_path) + '\n')
        f.write('nodes_expanded: ' + str(nodes_expanded) + '\n')
        f.write('max_search_depth: ' + str(max_search_depth) + '\n')
        f.write('max_frontier_size: ' + str(max_frontier_size) + '\n')
        f.write('running_time: ' + str(total_time) + '\n')
        f.write('\n')
        
        
def test_program(method, test_board):
    """
    Retrieves input, solves search problem and writes metrics to output file
    """
    # method, test_board = sys.argv[1], sys.argv[2]
    # test_board = ast.literal_eval(test_board)
    dim = int(math.sqrt(len(test_board)))
    test_board = np.asarray(test_board).reshape(dim, dim)
    board = Board(test_board)
    path_to_goal, cost_of_path, nodes_expanded, max_search_depth, total_time, max_frontier_size = search(method, board)
    output(method, path_to_goal, cost_of_path, nodes_expanded, max_search_depth, total_time, max_frontier_size)

test_program('bfs', [0,8,7,6,5,4,3,2,1])
test_program('dfs', [0,8,7,6,5,4,3,2,1])
test_program('ast', [0,8,7,6,5,4,3,2,1])
