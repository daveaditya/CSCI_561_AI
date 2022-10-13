import sys
import random
import json
from copy import deepcopy

import numpy as np

from host import GO

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

# TODO: Explain
# Right, Bottom, Left, Up
X_CHANGES = [1, 0, -1, 0]
Y_CHANGES = [0, 1, 0, -1]


#############################################################
### Helper Functions
#############################################################


def load_game_info(previous_game_state, current_game_state):
    is_previous_game_a_start = True
    is_current_game_a_start = True

    for row_idx in range(BOARD_SIZE - 1):
        for col_idx in range(BOARD_SIZE - 1):
            if previous_game_state[row_idx][col_idx] != UNOCCUPIED:
                is_previous_game_a_start = False
                is_current_game_a_start = False
                break
            elif current_game_state[row_idx][col_idx] != UNOCCUPIED:
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
    def __init__(self, n):
        """
        Go game.

        :param n: size of the board n*n
        """
        self.size = n
        # self.previous_board = None # Store the previous board
        self.X_move = True  # X chess plays first
        self.died_pieces = []  # Intialize died pieces to be empty
        self.n_move = 0  # Trace the number of moves
        self.max_move = n * n - 1  # The max movement of a Go game
        self.komi = n / 2  # Komi rule
        self.verbose = False  # Verbose only when there is a manual player

    def load_game(input_file_path=INPUT_FILE_PATH):
        with open(input_file_path, mode="r") as input_file:
            game_data = [input_file_line.strip() for input_file_line in input_file.readlines()]

            # piece_type = 1 => we are black, piece_type = 2 => we are white
            piece_type = int(game_data[0])

            previous_game_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int)
            current_game_state = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int)

            for i in range(1, 6):
                for j in range(len(game_data[i])):
                    previous_game_state[i - 1][j] = game_data[i][j]

            for i in range(6, 11):
                for j in range(len(game_data[i])):
                    current_game_state[i - 6][j] = game_data[i][j]

            return piece_type, previous_game_state, current_game_state

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

    def copy_board(self):
        """
        Copy the current board for potential testing.

        :param: None.
        :return: the copied board instance.
        """
        return deepcopy(self)

    def get_opponent_piece_type(self, piece_type):
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

    def find_liberty(self, i, j):
        """
        Find liberty of a given stone. If a group of allied stones has no liberty, they all die.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: boolean indicating whether the given stone still has liberty.
        """
        board = self.board
        ally_members = self.ally_dfs(i, j)
        for member in ally_members:
            neighbors = self.detect_neighbor(member[0], member[1])
            for piece in neighbors:
                # If there is empty space around a piece, it has liberty
                if board[piece[0]][piece[1]] == 0:
                    return True
        # If none of the pieces in a allied group has an empty space, it has no liberty
        return False

    def find_died_pieces(self, piece_type):
        """
        Find the died stones that has no liberty in the board for a given piece type.

        :param piece_type: 1('X') or 2('O').
        :return: a list containing the dead pieces row and column(row, column).
        """
        board = self.board
        died_pieces = []

        for i in range(len(board)):
            for j in range(len(board)):
                # Check if there is a piece at this position:
                if board[i][j] == piece_type:
                    # The piece die if it has no liberty
                    if not self.find_liberty(i, j):
                        died_pieces.append((i, j))
        return died_pieces

    def remove_died_pieces(self, piece_type):
        """
        Remove the dead stones in the board.

        :param piece_type: 1('X') or 2('O').
        :return: locations of dead pieces.
        """

        died_pieces = self.find_died_pieces(piece_type)
        if not died_pieces:
            return []
        self.remove_certain_pieces(died_pieces)
        return died_pieces

    def remove_certain_pieces(self, positions):
        """
        Remove the stones of certain locations.

        :param positions: a list containing the pieces to be removed row and column(row, column)
        :return: None.
        """
        board = self.board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def place_chess(self, i, j, piece_type):
        """
        Place a chess stone in the board.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the placement is valid.
        """
        board = self.board

        valid_place = self.valid_place_check(i, j, piece_type)
        if not valid_place:
            return False
        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.update_board(board)
        # Remove the following line for HW2 CS561 S2020
        # self.n_move += 1
        return True

    def valid_place_check(self, i, j, piece_type, test_check=False):
        """
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        """
        board = self.board
        verbose = self.verbose
        if test_check:
            verbose = False

        # Check if the place is in the board range
        if not (i >= 0 and i < len(board)):
            if verbose:
                print(("Invalid placement. row should be in the range 1 to {}.").format(len(board) - 1))
            return False
        if not (j >= 0 and j < len(board)):
            if verbose:
                print(("Invalid placement. column should be in the range 1 to {}.").format(len(board) - 1))
            return False

        # Check if the place already has a piece
        if board[i][j] != 0:
            if verbose:
                print("Invalid placement. There is already a chess in this position.")
            return False

        # Copy the board for testing
        test_go = self.copy_board()
        test_board = test_go.board

        # Check if the place has liberty
        test_board[i][j] = piece_type
        test_go.update_board(test_board)
        if test_go.find_liberty(i, j):
            return True

        # If not, remove the died pieces of opponent and check again
        test_go.remove_died_pieces(3 - piece_type)
        if not test_go.find_liberty(i, j):
            if verbose:
                print("Invalid placement. No liberty found in this position.")
            return False

        # Check special case: repeat placement causing the repeat board state (KO rule)
        else:
            if self.died_pieces and self.compare_board(self.previous_board, test_go.board):
                if verbose:
                    print("Invalid placement. A repeat move not permitted by the KO rule.")
                return False
        return True

    def update_board(self, new_board):
        """
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        """
        self.board = new_board

    def game_end(self, piece_type, action="MOVE"):
        """
        Check if the game should end.

        :param piece_type: 1('X') or 2('O').
        :param action: "MOVE" or "PASS".
        :return: boolean indicating whether the game should end.
        """

        # Case 1: max move reached
        if self.n_move >= self.max_move:
            return True
        # Case 2: two players all pass the move.
        if self.compare_board(self.previous_board, self.board) and action == "PASS":
            return True
        return False

    def score(self, piece_type):
        """
        Get score of a player by counting the number of stones.

        :param piece_type: 1('X') or 2('O').
        :return: boolean indicating whether the game should end.
        """

        board = self.board
        cnt = 0
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == piece_type:
                    cnt += 1
        return cnt


#############################################################
### My Player Implementation
### MiniMax with Alpha-Beta Pruning
#############################################################
class MyPlayer:
    def __init__(self, go, piece_type, previous_game_state, current_game_state):
        self.type = "minimax_with_pruning_player"
        self.piece_type = piece_type
        self.opponent_piece_type = go.get_opponent_piece_type(self.piece_type)
        self.previous_game_state = previous_game_state
        self.current_game_state = current_game_state

    def make_move(self, search_depth, branching_factor, step):
        max_move, max_move_score = self.maxmillians_move(
            self.current_game_state, self.side, search_depth, 0, branching_factor, -np.inf, np.inf, None, step, False
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
        if step_number + current_depth == 24 or is_second_pass:
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
        opponent_side = self.get_opponent_side(side)
        side_count = 0
        side_liberty = set()
        opponent_count = 0
        opponent_liberty = set()
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if game_state[i][j] == side:
                    side_count += 1
                elif game_state[i][j] == opponent_side:
                    opponent_count += 1
                # This point should be UNOCCUPIED!
                else:
                    for index in range(len(X_CHANGES)):
                        new_i = i + X_CHANGES[index]
                        new_j = j + Y_CHANGES[index]
                        if 0 <= new_i < BOARD_SIZE and 0 <= new_j < BOARD_SIZE:
                            if game_state[new_i][new_j] == side:
                                side_liberty.add((i, j))
                            elif game_state[new_i][new_j] == opponent_side:
                                opponent_liberty.add((i, j))

        side_edge_count = 0
        opponent_side_edge_count = 0
        for j in range(BOARD_SIZE):
            if game_state[0][j] == side or game_state[BOARD_SIZE - 1][j] == side:
                side_edge_count += 1
            if game_state[0][j] == opponent_side or game_state[BOARD_SIZE - 1][j] == opponent_side:
                opponent_side_edge_count += 1

        for j in range(1, BOARD_SIZE - 1):
            if game_state[j][0] == side or game_state[j][BOARD_SIZE - 1] == side:
                side_edge_count += 1
            if game_state[j][0] == opponent_side or game_state[j][BOARD_SIZE - 1] == opponent_side:
                opponent_side_edge_count += 1

        center_unoccupied_count = 0
        for i in range(1, BOARD_SIZE - 1):
            for j in range(1, BOARD_SIZE - 1):
                if game_state[i][j] == UNOCCUPIED:
                    center_unoccupied_count += 1

        score = min(max((len(side_liberty) - len(opponent_liberty)), -8), 8) + (
                -4 * self.calculate_euler_number(game_state, side)) + (
                        5 * (side_count - opponent_count)) - (9 * side_edge_count * (center_unoccupied_count / 9))
        if self.side == WHITE:
            score += KOMI


    def do_move(self, game_state, side, move):
        new_game_state = deepcopy(game_state)
        # We know that the move which is going to be done is definitely valid for this side!
        # We checked for liberty and KO before! So we can do the move!
        new_game_state[move[0]][move[1]] = side
        # Now we check if we have to delete opponents group or not
        for index in range(len(X_CHANGES)):
            new_i = move[0] + X_CHANGES[index]
            new_j = move[1] + Y_CHANGES[index]
            if 0 <= new_i < BOARD_SIZE and 0 <= new_j < BOARD_SIZE:
                opponent_side = self.get_opponent_side(side)
                if new_game_state[new_i][new_j] == opponent_side:
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
                            if 0 <= new_new_i < BOARD_SIZE and 0 <= new_new_j < BOARD_SIZE:
                                if (new_new_i, new_new_j) in visited:
                                    continue
                                elif new_game_state[new_new_i][new_new_j] == UNOCCUPIED:
                                    opponent_group_should_be_deleted = False
                                    break
                                elif new_game_state[new_new_i][new_new_j] == opponent_side and \
                                        (new_new_i, new_new_j) not in visited:
                                    stack.append((new_new_i, new_new_j))

                    if opponent_group_should_be_deleted:
                        for stone in visited:
                            new_game_state[stone[0]][stone[1]] = UNOCCUPIED
        return new_game_state


    def calculate_euler_number(self, game_state, side):
        opponent_side = self.get_opponent_side(side)
        new_game_state = np.zeros((BOARD_SIZE + 2, BOARD_SIZE + 2), dtype=int)
        # First copy the original game_state
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                new_game_state[i + 1][j + 1] = game_state[i][j]

        q1_side = 0
        q2_side = 0
        q3_side = 0
        q1_opponent_side = 0
        q2_opponent_side = 0
        q3_opponent_side = 0

        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                new_game_sub_state = new_game_state[i: i + 2, j: j + 2]
                q1_side += self.count_q1(new_game_sub_state, side)
                q2_side += self.count_q2(new_game_sub_state, side)
                q3_side += self.count_q3(new_game_sub_state, side)
                q1_opponent_side += self.count_q1(new_game_sub_state, opponent_side)
                q2_opponent_side += self.count_q2(new_game_sub_state, opponent_side)
                q3_opponent_side += self.count_q3(new_game_sub_state, opponent_side)

        return (q1_side - q3_side + 2 * q2_side - (q1_opponent_side - q3_opponent_side + 2 * q2_opponent_side)) / 4

    def count_q1(self, game_sub_state, side):
        if ((game_sub_state[0][0] == side and game_sub_state[0][1] != side
             and game_sub_state[1][0] != side and game_sub_state[1][1] != side)
                or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                    and game_sub_state[1][0] != side and game_sub_state[1][1] != side)
                or (game_sub_state[0][0] != side and game_sub_state[0][1] != side
                    and game_sub_state[1][0] == side and game_sub_state[1][1] != side)
                or (game_sub_state[0][0] != side and game_sub_state[0][1] != side
                    and game_sub_state[1][0] != side and game_sub_state[1][1] == side)):
            return 1
        else:
            return 0

    def count_q2(self, game_sub_state, side):
        if ((game_sub_state[0][0] == side and game_sub_state[0][1] != side
             and game_sub_state[1][0] != side and game_sub_state[1][1] == side)
                or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                    and game_sub_state[1][0] == side and game_sub_state[1][1] != side)):
            return 1
        else:
            return 0

    def count_q3(self, game_sub_state, side):
        if ((game_sub_state[0][0] == side and game_sub_state[0][1] == side
             and game_sub_state[1][0] == side and game_sub_state[1][1] != side)
                or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                    and game_sub_state[1][0] == side and game_sub_state[1][1] == side)
                or (game_sub_state[0][0] == side and game_sub_state[0][1] != side
                    and game_sub_state[1][0] == side and game_sub_state[1][1] == side)
                or (game_sub_state[0][0] != side and game_sub_state[0][1] == side
                    and game_sub_state[1][0] == side and game_sub_state[1][1] == side)):
            return 1
        else:
            return 0


    def find_valid_moves(self, game_state, side):
        valid_moves = {'3side': [], '1capturing': [], '2regular': []}
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if game_state[i][j] == UNOCCUPIED:
                    if self.check_for_liberty(game_state, i, j, side):
                        # Check for 'KO' rule before validating this move!
                        if not self.check_for_ko(i, j):
                            if i == 0 or j == 0 or i == BOARD_SIZE - 1 or j == BOARD_SIZE - 1:
                                valid_moves.get('3side').append((i, j))
                            else:
                                valid_moves.get('2regular').append((i, j))
                    # Check if we are capturing some stones by doing this move
                    else:
                        for index in range(len(X_CHANGES)):
                            new_i = i + X_CHANGES[index]
                            new_j = j + Y_CHANGES[index]
                            if 0 <= new_i < BOARD_SIZE and 0 <= new_j < BOARD_SIZE:
                                opponent_side = self.get_opponent_side(side)
                                if game_state[new_i][new_j] == opponent_side:
                                    # If there is a group of opponent_side that has no liberty with our move then we
                                    # can capture them and do this move!
                                    new_game_state = deepcopy(game_state)
                                    new_game_state[i][j] = side
                                    if not self.check_for_liberty(new_game_state, new_i, new_j,
                                                                  opponent_side):
                                        # Check for 'KO' rule before validating this move!
                                        if not self.check_for_ko(i, j):
                                            valid_moves.get('1capturing').append((i, j))
                                        break

                        # If the for loop did not break at all, then all of our neighbors have liberty and we cannot
                        # do this move

        valid_moves_list = []
        for valid_move in valid_moves.get('1capturing'):
            valid_moves_list.append(valid_move)
        for valid_move in valid_moves.get('2regular'):
            valid_moves_list.append(valid_move)
        for valid_move in valid_moves.get('3side'):
            valid_moves_list.append(valid_move)

        # DEBUG
        # print(valid_moves_list)
        return valid_moves_list


if __name__ == "__main__":

    go = GO(BOARD_SIZE)
    piece_type, previous_game_state, current_game_state = go.load_game()
    step = load_game_info(previous_game_state, current_game_state)

    my_player = MyPlayer(go, piece_type, previous_game_state, current_game_state)

    my_player.make_move(4, 20, step)
