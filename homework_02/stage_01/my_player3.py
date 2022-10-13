import json
from copy import deepcopy
from typing import List

import numpy as np

from host import GO
from submission.refs.my_player import BOARD_SIZE

#############################################################
### Constants
#############################################################

# Game related file paths
FILE_PATH = "."
INPUT_FILE_PATH = f"{FILE_PATH}/input.txt"
OUTPUT_FILE_PATH = f"{FILE_PATH}/output.txt"
GAME_INFO_FILE_PATH = f"{FILE_PATH}/game_info.json"

# Size of the board
BOARD_SIZE = 5

# Game board representations
UNOCCUPIED = 0
BLACK = 1
WHITE = 2

# Komi for white player as per instructions
KOMI = 2.5

# TODO: Explainx
# Format: Right, Bottom, Left, Up
X_CHANGES = [1, 0, -1, 0]
Y_CHANGES = [0, 1, 0, -1]

VALID_MOVE_ONE_CAPTURING = "ONE_CAPTURING"
VALID_MOVE_TWO_REGULAR = "TWO_REGULAR"
VALID_MOVE_THREE_SIZE = "THREE_SIZE"

# Player Constants
SEARCH_DEPTH = 4
BRANCHING_FACTOR = 20


#############################################################
### Helper Functions
#############################################################


def load_game_info(previous_board, current_board):
    is_previous_game_a_start = True
    is_current_game_a_start = True

    for row_idx in range(BOARD_SIZE):
        for col_idx in range(BOARD_SIZE):
            if previous_board[row_idx][col_idx] != UNOCCUPIED:
                is_previous_game_a_start = False
                is_current_game_a_start = False
                break
            elif current_board[row_idx][col_idx] != UNOCCUPIED:
                is_current_game_a_start = False

    if is_previous_game_a_start and is_current_game_a_start:
        step = 0
    elif is_previous_game_a_start and not is_current_game_a_start:
        step = 1
    else:
        with open(GAME_INFO_FILE_PATH, mode="r") as game_info_file:
            game_info = json.load(game_info_file)
            step = game_info["step"] + 2

    # Store updated game info
    with open(GAME_INFO_FILE_PATH, "w") as game_info_file:
        game_info = {"step": step}
        json.dump(game_info, game_info_file, ensure_ascii=False)

    return (step,)


#############################################################
### Go Class - Defines the Game Rules
#############################################################
class GO:
    def __init__(self, n, input_file_path):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size: int = n

        self.X_move: bool = True  # X chess plays first
        self.died_pieces: List = list()  # Intialize died pieces to be empty
        self.n_move: int = 0  # Trace the number of moves
        self.max_move: int = n * n - 1  # The max movement of a Go game
        self.komi: float = n / 2  # Komi rule
        self.verbose: bool = False  # Verbose only when there is a manual player

        with open(input_file_path, mode="r") as input_file:
            game_data = [input_file_line.strip() for input_file_line in input_file.readlines()]

            # piece_type = 1 => we are black, piece_type = 2 => we are white
            self.piece_type = int(game_data[0])

            previous_board = np.zeros((self.go.size, self.go.size), dtype=np.int)
            current_board = np.zeros((self.go.size, self.go.size), dtype=np.int)

            for line_num in range(1, 6):
                for col_num in range(len(game_data[line_num])):
                    previous_board[line_num - 1][col_num] = game_data[line_num][col_num]

            for line_num in range(6, 11):
                for col_num in range(len(game_data[line_num])):
                    current_board[line_num - 6][col_num] = game_data[line_num][col_num]

            self.previous_board = previous_board
            self.current_board = current_board

    def store_move(output_file_path, next_move):
        with open(output_file_path, mode="w") as output_file:
            if next_move is None or next_move == (-1, -1):
                output_file.write("PASS")
            else:
                output_file.write(f"{next_move[0]},{next_move[1]}")

    def compare_board(self, board1, board2):
        for i in range(self.size):
            for j in range(self.size):
                if board1[i][j] != board2[i][j]:
                    return False
        return True

    def get_opponent_piece(self, piece_type):
        return WHITE if piece_type == BLACK else BLACK

    def detect_neighbor(self, i, j):
        """
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        """
        board = self.board
        neighbors = []
        # Detect borders and add neighbor coordinates
        if i > 0:
            neighbors.append((i - 1, j))
        if i < len(board) - 1:
            neighbors.append((i + 1, j))
        if j > 0:
            neighbors.append((i, j - 1))
        if j < len(board) - 1:
            neighbors.append((i, j + 1))
        return neighbors

    def detect_neighbor_ally(self, i, j):
        """
        Detect the neighbor allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbored allies row and column (row, column) of position (i, j).
        """
        board = self.board
        neighbors = self.detect_neighbor(i, j)  # Detect neighbors
        group_allies = []
        # Iterate through neighbors
        for piece in neighbors:
            # Add to allies list if having the same color
            if board[piece[0]][piece[1]] == board[i][j]:
                group_allies.append(piece)
        return group_allies

    def ally_dfs(self, i, j):
        """
        Using DFS to search for all allies of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the all allies row and column (row, column) of position (i, j).
        """
        stack = [(i, j)]  # stack for DFS serach
        ally_members = []  # record allies positions during the search
        while stack:
            piece = stack.pop()
            ally_members.append(piece)
            neighbor_allies = self.detect_neighbor_ally(piece[0], piece[1])
            for ally in neighbor_allies:
                if ally not in stack and ally not in ally_members:
                    stack.append(ally)
        return ally_members

    def find_liberty(self, game_state, i, j, side):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            for index in range(len(X_CHANGES)):
                new_i = top_node[0] + X_CHANGES[index]
                new_j = top_node[1] + Y_CHANGES[index]
                if 0 <= new_i < BOARD_SIZE and 0 <= new_j < BOARD_SIZE:
                    if (new_i, new_j) in visited:
                        continue
                    elif game_state[new_i][new_j] == UNOCCUPIED:
                        return True
                    elif game_state[new_i][new_j] == side and (new_i, new_j) not in visited:
                        stack.append((new_i, new_j))
        return False

    def check_for_ko(self, i, j):
        if self.previous_board[i][j] != self.side:
            return False
        new_game_state = deepcopy(self.current_board)
        new_game_state[i][j] = self.side
        opponent_i, opponent_j = self.opponent_move()
        for index in range(len(X_CHANGES)):
            new_i = i + X_CHANGES[index]
            new_j = j + Y_CHANGES[index]
            if new_i == opponent_i and new_j == opponent_j:
                # If opponent group does not have liberty then delete all of them
                if not self.check_for_liberty(new_game_state, new_i, new_j, self.opponent_side):
                    # Delete all of the group from the board and check if we have the same exact board as before
                    self.delete_group(new_game_state, new_i, new_j, self.opponent_side)
        # If opponent's move is not out neighbor then it cannot be KO!
        return np.array_equal(new_game_state, self.previous_board)

    def update_board(self, new_board):
        """
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        """
        self.board = new_board


#############################################################
### My Player Implementation
### MiniMax with Alpha-Beta Pruning
#############################################################
class MyPlayer:
    def __init__(self, go: GO, piece_type: int, previous_board, current_board):
        self.type = "minimax_with_pruning_player"
        self.go: GO = go
        self.piece_type: int = piece_type
        self.opponent_piece_type: int = self.go.get_opponent_piece(self.piece_type)
        self.previous_board: List[List[int]] = previous_board
        self.current_board: List[List[int]] = current_board

    def make_move(self, search_depth, branching_factor, step):
        max_move, max_move_score = self.maxmillians_move(
            self.current_board, self.side, search_depth, 0, branching_factor, -np.inf, np.inf, None, step, False
        )
        self.go.store_move(max_move)

    def maxmillians_move(
        self,
        game_state,
        side,
        search_depth,
        current_depth,
        branching_factor,
        alpha,
        beta,
        last_move,
        step,
        is_second_pass,
    ):
        if search_depth == current_depth or step + current_depth == 24:
            return self.evaluate_game_state(game_state, side)
        if is_second_pass:
            return self.evaluate_game_state(game_state, side)
        is_second_pass = False

        max_move_value = -np.inf
        max_move = None
        valid_moves = self.find_valid_moves(game_state, side)
        valid_moves.append((-1, -1))
        if last_move == (-1, -1):
            is_second_pass = True
        for valid_move in valid_moves[:branching_factor]:
            # Create new game state
            opponent_side = self.get_opponent_side(side)
            if valid_move == (-1, -1):
                new_game_state = deepcopy(game_state)
            else:
                new_game_state = self.move(game_state, side, valid_move)
            min_move_value = self.mindys_move(
                new_game_state,
                opponent_side,
                search_depth,
                current_depth + 1,
                branching_factor,
                alpha,
                beta,
                valid_move,
                step,
                is_second_pass,
            )
            if max_move_value < min_move_value:
                max_move_value = min_move_value
                max_move = valid_move
            if max_move_value >= beta:
                if current_depth == 0:
                    return max_move, max_move_value
                else:
                    return max_move_value
            alpha = max(alpha, max_move_value)
        if current_depth == 0:
            return max_move, max_move_value
        else:
            return max_move_value

    def mindys_move(
        self,
        game_state,
        side,
        search_depth,
        current_depth,
        branching_factor,
        alpha,
        beta,
        last_move,
        step_number,
        is_second_pass,
    ):
        if search_depth == current_depth:
            return self.evaluate_game_state(game_state, side)
        if step_number + current_depth == self.go.max_move or is_second_pass:
            return self.evaluate_game_state(game_state, self.side)
        is_second_pass = False
        min_move_value = np.inf
        valid_moves = self.find_valid_moves(game_state, side)
        valid_moves.append((-1, -1))
        if last_move == (-1, -1):
            is_second_pass = True
        for valid_move in valid_moves[:branching_factor]:
            # Create new game state
            if valid_move == (-1, -1):
                new_game_state = deepcopy(game_state)
            else:
                new_game_state = self.move(game_state, side, valid_move)
            max_move_value = self.max_value(
                new_game_state,
                self.opponent_piece_type,
                search_depth,
                current_depth + 1,
                branching_factor,
                alpha,
                beta,
                valid_move,
                step_number,
                is_second_pass,
            )
            if max_move_value < min_move_value:
                min_move_value = max_move_value
            if min_move_value <= alpha:
                return min_move_value
            beta = min(beta, min_move_value)

        return min_move_value

    def evaluate_game_state(self, game_state, side):
        # Define heuristic here
        # Count number of sides stones - opponent stones
        opponent_piece = self.go.get_opponent_piece(side)
        my_piece_count = 0
        my_piece_liberties = set()
        opponent_piece_count = 0
        opponent_liberties = set()

        for i in range(self.go.size):
            for j in range(self.go.size):
                if game_state[i][j] == side:
                    my_piece_count += 1
                elif game_state[i][j] == opponent_piece:
                    opponent_piece_count += 1
                # This point should be UNOCCUPIED!
                else:
                    for index in range(len(X_CHANGES)):
                        new_i = i + X_CHANGES[index]
                        new_j = j + Y_CHANGES[index]
                        if 0 <= new_i < self.go.size and 0 <= new_j < self.go.size:
                            if game_state[new_i][new_j] == side:
                                my_piece_liberties.add((i, j))
                            elif game_state[new_i][new_j] == opponent_piece:
                                opponent_liberties.add((i, j))

        side_edge_count = 0
        opponent_side_edge_count = 0
        for j in range(self.go.size):
            if game_state[0][j] == side or game_state[self.go.size - 1][j] == side:
                side_edge_count += 1
            if game_state[0][j] == opponent_piece or game_state[self.go.size - 1][j] == opponent_piece:
                opponent_side_edge_count += 1

        for j in range(1, self.go.size - 1):
            if game_state[j][0] == side or game_state[j][self.go.size - 1] == side:
                side_edge_count += 1
            if game_state[j][0] == opponent_piece or game_state[j][self.go.size - 1] == opponent_piece:
                opponent_side_edge_count += 1

        center_unoccupied_count = 0
        for i in range(1, self.go.size - 1):
            for j in range(1, self.go.size - 1):
                if game_state[i][j] == UNOCCUPIED:
                    center_unoccupied_count += 1

        score = (
            min(max((len(my_piece_liberties) - len(opponent_liberties)), -8), 8)
            + (-4 * self.calculate_euler_number(game_state, side))
            + (5 * (my_piece_count - opponent_piece_count))
            - (9 * side_edge_count * (center_unoccupied_count / 9))
        )
        if self.side == WHITE:
            score += KOMI

    def do_move(self, board, side, move):
        new_board = deepcopy(board)

        # We know that the move which is going to be done is definitely valid for this side!
        # We checked for liberty and KO before! So we can do the move!
        new_board[move[0]][move[1]] = side

        # Now we check if we have to delete opponents group or not
        for index in range(len(X_CHANGES)):
            new_i = move[0] + X_CHANGES[index]
            new_j = move[1] + Y_CHANGES[index]

            if 0 <= new_i < self.go.size and 0 <= new_j < self.go.size:
                opponent_side = self.go.get_opponent_side(side)

                if new_board[new_i][new_j] == opponent_side:
                    # DFS!
                    stack = [(new_i, new_j)]
                    visited = set()
                    opponent_group_should_be_deleted = True

                    while stack:
                        top_node = stack.pop()
                        visited.add(top_node)
                        for index in range(len(X_CHANGES)):
                            new_new_i = top_node[0] + X_CHANGES[index]
                            new_new_j = top_node[1] + Y_CHANGES[index]
                            if 0 <= new_new_i < self.go.size and 0 <= new_new_j < self.go.size:
                                if (new_new_i, new_new_j) in visited:
                                    continue
                                elif new_board[new_new_i][new_new_j] == UNOCCUPIED:
                                    opponent_group_should_be_deleted = False
                                    break
                                elif (
                                    new_board[new_new_i][new_new_j] == opponent_side
                                    and (new_new_i, new_new_j) not in visited
                                ):
                                    stack.append((new_new_i, new_new_j))

                    if opponent_group_should_be_deleted:
                        for stone in visited:
                            new_board[stone[0]][stone[1]] = UNOCCUPIED

        return new_board

    def calculate_euler_number(self, game_state, side):
        def count_q1(self, game_sub_state, side):
            if (
                (
                    game_sub_state[0][0] == side
                    and game_sub_state[0][1] != side
                    and game_sub_state[1][0] != side
                    and game_sub_state[1][1] != side
                )
                or (
                    game_sub_state[0][0] != side
                    and game_sub_state[0][1] == side
                    and game_sub_state[1][0] != side
                    and game_sub_state[1][1] != side
                )
                or (
                    game_sub_state[0][0] != side
                    and game_sub_state[0][1] != side
                    and game_sub_state[1][0] == side
                    and game_sub_state[1][1] != side
                )
                or (
                    game_sub_state[0][0] != side
                    and game_sub_state[0][1] != side
                    and game_sub_state[1][0] != side
                    and game_sub_state[1][1] == side
                )
            ):
                return 1
            else:
                return 0

        def count_q2(self, game_sub_state, side):
            if (
                game_sub_state[0][0] == side
                and game_sub_state[0][1] != side
                and game_sub_state[1][0] != side
                and game_sub_state[1][1] == side
            ) or (
                game_sub_state[0][0] != side
                and game_sub_state[0][1] == side
                and game_sub_state[1][0] == side
                and game_sub_state[1][1] != side
            ):
                return 1
            else:
                return 0

        def count_q3(self, game_sub_state, side):
            if (
                (
                    game_sub_state[0][0] == side
                    and game_sub_state[0][1] == side
                    and game_sub_state[1][0] == side
                    and game_sub_state[1][1] != side
                )
                or (
                    game_sub_state[0][0] != side
                    and game_sub_state[0][1] == side
                    and game_sub_state[1][0] == side
                    and game_sub_state[1][1] == side
                )
                or (
                    game_sub_state[0][0] == side
                    and game_sub_state[0][1] != side
                    and game_sub_state[1][0] == side
                    and game_sub_state[1][1] == side
                )
                or (
                    game_sub_state[0][0] != side
                    and game_sub_state[0][1] == side
                    and game_sub_state[1][0] == side
                    and game_sub_state[1][1] == side
                )
            ):
                return 1
            else:
                return 0

        opponent_side = self.get_opponent_side(side)
        new_game_state = np.zeros((self.go.size + 2, self.go.size + 2), dtype=int)
        # First copy the original game_state
        for i in range(self.go.size):
            for j in range(self.go.size):
                new_game_state[i + 1][j + 1] = game_state[i][j]

        q1_side = 0
        q2_side = 0
        q3_side = 0
        q1_opponent_side = 0
        q2_opponent_side = 0
        q3_opponent_side = 0

        for i in range(self.go.size):
            for j in range(self.go.size):
                new_game_sub_state = new_game_state[i : i + 2, j : j + 2]
                q1_side += count_q1(new_game_sub_state, side)
                q2_side += count_q2(new_game_sub_state, side)
                q3_side += count_q3(new_game_sub_state, side)
                q1_opponent_side += count_q1(new_game_sub_state, opponent_side)
                q2_opponent_side += count_q2(new_game_sub_state, opponent_side)
                q3_opponent_side += count_q3(new_game_sub_state, opponent_side)

        return (q1_side - q3_side + 2 * q2_side - (q1_opponent_side - q3_opponent_side + 2 * q2_opponent_side)) / 4

    def find_valid_moves(self, board, side):
        valid_moves_list = {
            VALID_MOVE_ONE_CAPTURING: list(),
            VALID_MOVE_TWO_REGULAR: list(),
            VALID_MOVE_THREE_SIZE: list(),
        }

        for i in range(self.go.size):
            for j in range(self.go.size):
                if board[i][j] == UNOCCUPIED:
                    if self.go.find_liberty(board, i, j, side):
                        # Check for 'KO' rule before validating this move!
                        if not self.go.check_for_ko(i, j):
                            if i == 0 or j == 0 or i == self.go.size - 1 or j == self.go.size - 1:
                                valid_moves_list[VALID_MOVE_THREE_SIZE].append((i, j))
                            else:
                                valid_moves_list.get(VALID_MOVE_TWO_REGULAR).append((i, j))
                    # Check if we are capturing some stones by doing this move
                    else:
                        for index in range(len(X_CHANGES)):
                            new_i = i + X_CHANGES[index]
                            new_j = j + Y_CHANGES[index]
                            if 0 <= new_i < self.go.size and 0 <= new_j < self.go.size:
                                opponent_side = self.go.get_opponent_piece(side)
                                if board[new_i][new_j] == opponent_side:
                                    # If there is a group of opponent_side that has no liberty with our move then we
                                    # can capture them and do this move!
                                    new_game_state = deepcopy(board)
                                    new_game_state[i][j] = side
                                    if not self.go.find_liberty(new_game_state, new_i, new_j, opponent_side):
                                        # Check for 'KO' rule before validating this move!
                                        if not self.go.check_for_ko(i, j):
                                            valid_moves_list[VALID_MOVE_ONE_CAPTURING].append((i, j))
                                        break

                        # If the for loop did not break at all, then all of our neighbors have liberty and we cannot
                        # do this move

        valid_moves_list = [
            *valid_moves_list[VALID_MOVE_ONE_CAPTURING],
            *valid_moves_list[VALID_MOVE_TWO_REGULAR],
            *valid_moves_list[VALID_MOVE_THREE_SIZE],
        ]

        # DEBUG
        # print(valid_moves_list)
        return valid_moves_list


if __name__ == "__main__":

    go = GO(BOARD_SIZE, INPUT_FILE_PATH)
    piece_type, previous_board, current_board = go.piece_type, go.previous_board, go.current_board
    step = load_game_info(previous_board, current_board)

    my_player = MyPlayer(go, piece_type, previous_board, current_board)

    my_player.make_move(SEARCH_DEPTH, BRANCHING_FACTOR, step)
