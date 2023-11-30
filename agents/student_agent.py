from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
import world as world

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """
    global depth 
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.prunning = 0
        self.max_depth = 3
    def step(self, chess_board, my_pos, adv_pos, max_step): 
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direc of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        self.max_step = max_step
        self.moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        self.size = chess_board.shape[0] - 1
        # Some simple code to help you with timing. Consider checking 
        # time_taken during your search and breaking with the best answer
        # so far when it nears 2 seconds.
        self.prunning = 0
        start_time = time.time()
        _, new_pos, wall = self.minimax(chess_board, my_pos, adv_pos, max_step, start_time, float('-inf'), float('inf'), True, self.max_depth)
        time_taken = time.time() - start_time
        if new_pos == None or wall == None:
            return self.random_move(chess_board, my_pos, adv_pos, max_step)
        return new_pos, self.dir_map[wall]
    
    def minimax(self, chess_board, my_pos, adv_pos, max_step, start_time, alpha, beta, maximizing_player, depth):
        if ((time.time() - start_time) >= 1.97) or (self.sort_positions(chess_board, my_pos,adv_pos, max_step) == None) or self.sort_positions(chess_board, adv_pos, my_pos, max_step) == None or depth == 0:
            return self.calculate_area_size(chess_board, my_pos, adv_pos), None, None
        self.prunning+=1
        if maximizing_player:
            max_score = float('-inf')
            best_action = None, None
            for move in self.sort_positions(chess_board, my_pos,adv_pos, max_step):
                for wall in self.dir_map.keys():
                    if not self.check_valid_step(my_pos, move, adv_pos, wall, chess_board):
                        continue
                    temp_board = self.simulate_move(chess_board, move, wall)
                    score, _, _  = self.minimax(temp_board, move, adv_pos, max_step, start_time, alpha, beta, False, depth-1)
                    if score >= max_score:
                        max_score = score
                        best_action = (move, wall)
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
            action, wall = best_action
            return max_score, action, wall
        else:
            min_score = float('inf')
            best_action = None, None
            for move in self.sort_positions(chess_board, adv_pos, my_pos, max_step):
                for wall in self.dir_map.keys():
                    temp_board = self.simulate_move(chess_board, move, wall)
                    score, _, _  = self.minimax(temp_board, my_pos, move, max_step, start_time, alpha, beta, True, depth -1)
                    if score <= min_score:
                        min_score = score
                        best_action = (move, wall)
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
            
            action, wall = best_action
            return min_score, action, wall

    def get_new_position(self, pos, action):
        if pos != None:
            (x ,y) = pos
            match action:
                case "u":
                    return (x - 1, y)
                case "r":
                    return (x, y + 1)
                case "d":
                    return (x + 1, y)
                case "l":
                    return (x, y - 1)
    
    def move_zone(self, chess_board, my_pos, adv_pos, max_step):
        return self.get_all_moves(chess_board, my_pos, max_step) - self.get_all_moves(chess_board, adv_pos, max_step)
    
    def calculate_l_size(self, chess_board, start_pos):
        l_size = 0
        for direc in ["u", "r", "d", "l"]:
            new_pos = start_pos
            while self.valid_action(new_pos, direc, chess_board):
                new_pos = self.get_new_position(new_pos, direc)
                l_size += 1
        return l_size
        
    
    def get_all_moves(self, chess_board, start_pos, max_step):
        visited = set()
        stack = [start_pos]
        area_size = 0

        while stack and max_step > 0:
            max_step -= 1
            current_pos = stack.pop()
            if current_pos in visited:
                continue

            visited.add(current_pos)
            area_size += 1

            x, y = current_pos
            for direc in ["u", "r", "d", "l"]:
                new_x, new_y = self.get_new_position(current_pos, direc)
                if self.is_valid_position(chess_board, (new_x, new_y)) and (new_x, new_y) not in visited:
                    stack.append((new_x, new_y))

        return area_size
    
    def calculate_area_size(self, chess_board, my_pos, adv_pos):
        father = dict()
        for r in range(self.size+1):
            for c in range(self.size+1):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(self.size):
            for c in range(self.size):
                for dir, move in enumerate(
                    self.moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(self.size):
            for c in range(self.size):
                find((r, c))
        p0_r = find(tuple(my_pos))
        p1_r = find(tuple(adv_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        return p0_score - p1_score

    def is_valid_position(self, chess_board, pos):
        x, y = pos
        x_max, y_max, _ = chess_board.shape
        return 0 <= x < x_max and 0 <= y < y_max
    
    def random_move(self, chess_board, my_pos, adv_pos, max_step):
        for i in self.iterate_positions(my_pos[0], my_pos[1], max_step):
            for j in self.dir_map.keys():
                if self.check_valid_step(my_pos, i, adv_pos, j, chess_board):
                    return i, self.dir_map[j]
    
    def valid_action(self, pos, action, chess_board):
        (x, y) = pos
        match action:
            case "u":
                return x > 0 and not chess_board[x, y, 0]
            case "r":
                return y < self.size and not chess_board[x, y, 1]
            case "d":
                return x < self.size and not chess_board[x, y, 2]
            case "l":
                return y > 0 and not chess_board[x, y, 3]    
        return False
        
    def iterate_positions(self,x, y, radius):
        positions = []
        for i in range(-radius, radius):
            for j in range(-radius, radius):
                if abs(i) + abs(j) <= radius and 0 <= (x + i) <= self.size and 0 <= (y + j) <= self.size:
                    positions.append((x + i, y + j))
        return positions

    def scoreuate_position(self, chess_board, my_pos, adv_pos, max_step):
        x,y = my_pos
        x2, y2 = adv_pos
        count = 0
        for i in chess_board[x][y] :
            if i == True:
                count += 1
        factor = abs(x2-x) + abs(y2-y)
        return count, factor, my_pos

    def sort_positions(self, chess_board, my_pos, adv_pos, max_step):
        positions = []
        for pos in self.iterate_positions(my_pos[0], my_pos[1], max_step):
            if self.check_valid_move(my_pos, pos, adv_pos, chess_board):
                positions.append(self.scoreuate_position(chess_board, pos, adv_pos, max_step))
        positions.sort(key=lambda x: (x[1],x[0]))
        return list(map(lambda c: c[2], positions[:6]))
        
    def check_valid_move(self, start_pos, end_pos, adv_pos,chess_board):
        
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

            
    
    def check_valid_step(self, start_pos, end_pos,adv_pos, barrier_dir, chess_board):
        r, c = end_pos
        if chess_board[r, c, self.dir_map[barrier_dir]]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == self.max_step:
                break
            for dir, move in enumerate(self.moves):
                if chess_board[r, c, dir]:
                    continue
                next_pos = (cur_pos[0] + move[0], cur_pos[1] + move[1])
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached
            
    def simulate_move(self, chess_board, end_pos, action):
        x,y = end_pos
        temp_board = chess_board.copy()
        temp_board[x,y, self.dir_map[action]] = True
        return temp_board
