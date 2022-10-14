import json
from copy import deepcopy

import numpy as np

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
BLACK_PIECE = 1
WHITE_PIECE = 2

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

    for row_idx in range(BOARD_SIZE - 1):
        for col_idx in range(BOARD_SIZE - 1):
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

    return step


def store_move(output_file_path, move):
    with open(output_file_path, mode="w") as output_file:
        if move is None or move == (-1, -1):
            output_file.write("PASS")
        else:
            output_file.write(f"{move[0]},{move[1]}")


#############################################################
### Go Class - Defines the Game Rules
#############################################################
class GO:
    def __init__(self, board_size: int, input_file_path: str, representations, x_changes, y_changes):
        self.board_size: int = board_size
        self.input_file_path = input_file_path
        self.BLACK_PIECE = representations["BLACK_PIECE"]
        self.WHITE_PIECE = representations["WHITE_PIECE"]
        self.UNOCCUPIED = representations["UNOCCUPIED"]
        self.X_CHANGES = x_changes
        self.Y_CHANGES = y_changes

        self.max_move: int = self.board_size * self.board_size - 1  # The maximum moves of the Go game

        with open(self.input_file_path, mode="r") as input_file:
            game_info = [input_file_line.strip() for input_file_line in input_file.readlines()]

            # piece = 1 => we are black, piece = 2 => we are white
            piece = int(game_info[0])

            previous_board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
            current_board = np.zeros((self.board_size, self.board_size), dtype=np.int32)

            for line_num in range(1, 6):
                for col_num in range(len(game_info[line_num])):
                    previous_board[line_num - 1][col_num] = game_info[line_num][col_num]

            for line_num in range(6, 11):
                for col_num in range(len(game_info[line_num])):
                    current_board[line_num - 6][col_num] = game_info[line_num][col_num]

            self.piece = piece
            self.previous_board = previous_board
            self.current_board = current_board

    def get_opponent_piece(self, piece):
        return self.WHITE_PIECE if piece == self.BLACK_PIECE else self.BLACK_PIECE



    def find_valid_moves(self, board, piece, my_piece):
        valid_moves_list = {
            VALID_MOVE_ONE_CAPTURING: list(),
            VALID_MOVE_TWO_REGULAR: list(),
            VALID_MOVE_THREE_SIZE: list(),
        }

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == self.UNOCCUPIED:
                    if self.find_liberty(board, i, j, piece):
                        # Check for 'KO' rule before validating this move!
                        if not self.check_for_ko(i, j, my_piece):
                            if i == 0 or j == 0 or i == self.board_size - 1 or j == self.board_size - 1:
                                valid_moves_list[VALID_MOVE_THREE_SIZE].append((i, j))
                            else:
                                valid_moves_list.get(VALID_MOVE_TWO_REGULAR).append((i, j))
                    # Check if we are capturing some stones by doing this move
                    else:
                        for idx in range(len(self.X_CHANGES)):
                            i_prime = i + self.X_CHANGES[idx]
                            j_prime = j + self.Y_CHANGES[idx]
                            if 0 <= i_prime < self.board_size and 0 <= j_prime < self.board_size:
                                opponent_piece = self.get_opponent_piece(piece)
                                if board[i_prime][j_prime] == opponent_piece:
                                    # If there is a group of opponent_side that has no liberty with our move then we
                                    # can capture them and do this move!
                                    new_game_board = deepcopy(board)
                                    new_game_board[i][j] = piece
                                    if not self.find_liberty(new_game_board, i_prime, j_prime, opponent_piece):
                                        # Check for 'KO' rule before validating this move!
                                        if not self.check_for_ko(i, j, my_piece):
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

    def find_liberty(self, board, i, j, piece):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            for idx in range(len(self.X_CHANGES)):
                i_prime = top_node[0] + X_CHANGES[idx]
                j_prime = top_node[1] + Y_CHANGES[idx]
                if 0 <= i_prime < self.board_size and 0 <= j_prime < self.board_size:
                    if (i_prime, j_prime) in visited:
                        continue
                    elif board[i_prime][j_prime] == self.UNOCCUPIED:
                        return True
                    elif board[i_prime][j_prime] == piece and (i_prime, j_prime) not in visited:
                        stack.append((i_prime, j_prime))
        return False


    def check_for_ko(self, i, j, my_piece):
        if self.previous_board[i][j] != my_piece:
            return False
        new_game_board = deepcopy(self.current_board)
        new_game_board[i][j] = my_piece
        opponent_i, opponent_j = self.opponent_move()
        for idx in range(len(X_CHANGES)):
            i_prime = i + X_CHANGES[idx]
            j_prime = j + Y_CHANGES[idx]
            if i_prime == opponent_i and j_prime == opponent_j:
                # If opponent group does not have liberty then delete all of them
                if not self.find_liberty(new_game_board, i_prime, j_prime, self.opponent_piece):
                    # Delete all of the group from the board and check if we have the same exact board as before
                    self.delete_group(new_game_board, i_prime, j_prime, self.opponent_piece)
        # If opponent's move is not out neighbor then it cannot be KO!
        return np.array_equal(new_game_board, self.previous_board)


    def opponent_move(self):
        if np.array_equal(self.current_board, self.previous_board):
            return None
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (
                    self.current_board[i][j] != self.previous_board[i][j]
                    and self.current_board[i][j] != self.UNOCCUPIED
                ):
                    # Just a double check that the difference is a stone that belongs to the opponent!
                    # assert self.current_board[i][j] == self.opponent_piece, "Houston we've got a problem!"
                    return i, j

    def delete_group(self, game_board, i, j, piece):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            game_board[top_node[0]][top_node[1]] = self.UNOCCUPIED
            for idx in range(len(self.X_CHANGES)):
                i_prime = top_node[0] + self.X_CHANGES[idx]
                j_prime = top_node[1] + self.Y_CHANGES[idx]
                if 0 <= i_prime < self.board_size and 0 <= j_prime < self.board_size:
                    if (i_prime, j_prime) in visited:
                        continue
                    elif game_board[i_prime][j_prime] == piece:
                        stack.append((i_prime, j_prime))
        return game_board


#############################################################
### My Player Implementation
### MiniMax with Alpha-Beta Pruning
#############################################################
class MyPlayer:
    def __init__(self, go: GO, piece: int, previous_board, current_board):
        self.type = "minimax_with_pruning_player"
        self.go: GO = go
        self.my_piece: int = piece
        self.opponent_piece: int = self.go.get_opponent_piece(self.my_piece)
        self.previous_board = previous_board
        self.current_board = current_board

    def make_a_move(self, search_depth: int, branching_factor: int, step: int):
        max_move, _ = self.maxmillians_move(
            self.current_board, self.my_piece, search_depth, 0, branching_factor, -np.inf, np.inf, None, step, False
        )
        store_move(OUTPUT_FILE_PATH, max_move)

    def maxmillians_move(
        self,
        game_board,
        piece,
        search_depth,
        current_depth,
        branching_factor,
        alpha,
        beta,
        last_move,
        step,
        is_second_pass,
    ):
        if search_depth == current_depth or step + current_depth == self.go.max_move:
            return self.evaluate_game_board(game_board, piece)
        if is_second_pass:
            return self.evaluate_game_board(game_board, piece)

        is_second_pass = False
        max_move_value = -np.inf
        max_move = None
        valid_moves = self.go.find_valid_moves(game_board, piece, self.my_piece)
        valid_moves.append((-1, -1))

        if last_move == (-1, -1):
            is_second_pass = True

        for valid_move in valid_moves[:branching_factor]:
            # Create new game state
            opponent_piece = self.go.get_opponent_piece(piece)

            if valid_move == (-1, -1):
                new_game_board = deepcopy(game_board)
            else:
                new_game_board = self.do_move(game_board, piece, valid_move)

            min_move_value = self.mindys_move(
                new_game_board,
                opponent_piece,
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
        game_board,
        piece,
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
            return self.evaluate_game_board(game_board, piece)
        if step_number + current_depth == self.go.max_move or is_second_pass:
            return self.evaluate_game_board(game_board, self.my_piece)

        is_second_pass = False
        min_move_value = np.inf
        valid_moves = self.go.find_valid_moves(game_board, piece, self.my_piece)
        valid_moves.append((-1, -1))

        if last_move == (-1, -1):
            is_second_pass = True

        for valid_move in valid_moves[:branching_factor]:
            # Create new game state
            opponent_piece = self.go.get_opponent_piece(piece)

            if valid_move == (-1, -1):
                new_game_board = deepcopy(game_board)
            else:
                new_game_board = self.do_move(game_board, piece, valid_move)

            max_move_value = self.maxmillians_move(
                new_game_board,
                opponent_piece,
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

    def evaluate_game_board(self, game_state, piece):
        # My Heiristics here...
        opponent_piece = self.go.get_opponent_piece(piece)
        piece_count = 0
        piece_liberties = set()
        opponent_piece_count = 0
        opponent_liberties = set()

        for i in range(self.go.board_size):
            for j in range(self.go.board_size):
                if game_state[i][j] == piece:
                    piece_count += 1
                elif game_state[i][j] == opponent_piece:
                    opponent_piece_count += 1
                # This point should be UNOCCUPIED!
                else:
                    for idx in range(len(self.go.X_CHANGES)):
                        i_prime = i + self.go.X_CHANGES[idx]
                        j_prime = j + self.go.Y_CHANGES[idx]
                        if 0 <= i_prime < self.go.board_size and 0 <= j_prime < self.go.board_size:
                            if game_state[i_prime][j_prime] == piece:
                                piece_liberties.add((i, j))
                            elif game_state[i_prime][j_prime] == opponent_piece:
                                opponent_liberties.add((i, j))

        piece_edge_count = 0
        opponent_piece_edge_count = 0
        for j in range(self.go.board_size):
            if game_state[0][j] == piece or game_state[self.go.board_size - 1][j] == piece:
                piece_edge_count += 1
            if game_state[0][j] == opponent_piece or game_state[self.go.board_size - 1][j] == opponent_piece:
                opponent_piece_edge_count += 1

        for j in range(1, self.go.board_size - 1):
            if game_state[j][0] == piece or game_state[j][self.go.board_size - 1] == piece:
                piece_edge_count += 1
            if game_state[j][0] == opponent_piece or game_state[j][self.go.board_size - 1] == opponent_piece:
                opponent_piece_edge_count += 1

        center_unoccupied_count = 0
        for i in range(1, self.go.board_size - 1):
            for j in range(1, self.go.board_size - 1):
                if game_state[i][j] == self.go.UNOCCUPIED:
                    center_unoccupied_count += 1

        score = (
            min(max((len(piece_liberties) - len(opponent_liberties)), -8), 8)
            + (-4 * self.calculate_magic_number(game_state, piece))
            + (5 * (piece_count - opponent_piece_count))
            - (9 * piece_edge_count * (center_unoccupied_count / 9))
        )
        if self.my_piece == self.go.WHITE_PIECE:
            score += KOMI

        return score

    def do_move(self, board, piece, move):
        new_board = deepcopy(board)

        # We know that the move which is going to be done is definitely valid for this side!
        # We checked for liberty and KO before! So we can do the move!
        new_board[move[0]][move[1]] = piece

        # Now we check if we have to delete opponents group or not
        for idx in range(len(self.go.X_CHANGES)):
            i_prime = move[0] + self.go.X_CHANGES[idx]
            j_prime = move[1] + self.go.Y_CHANGES[idx]

            if 0 <= i_prime < self.go.board_size and 0 <= j_prime < self.go.board_size:
                opponent_piece = self.go.get_opponent_piece(piece)

                if new_board[i_prime][j_prime] == opponent_piece:
                    # DFS!
                    stack = [(i_prime, j_prime)]
                    visited = set()
                    should_delete_opponent_group = True

                    while stack:
                        top_node = stack.pop()
                        visited.add(top_node)
                        for idx in range(len(self.go.X_CHANGES)):
                            new_i_prime = top_node[0] + self.go.X_CHANGES[idx]
                            new_j_prime = top_node[1] + self.go.Y_CHANGES[idx]

                            if 0 <= new_i_prime < self.go.board_size and 0 <= new_j_prime < self.go.board_size:
                                if (new_i_prime, new_j_prime) in visited:
                                    continue
                                elif new_board[new_i_prime][new_j_prime] == self.go.UNOCCUPIED:
                                    should_delete_opponent_group = False
                                    break
                                elif (
                                    new_board[new_i_prime][new_j_prime] == opponent_piece
                                    and (new_i_prime, new_j_prime) not in visited
                                ):
                                    stack.append((new_i_prime, new_j_prime))

                    if should_delete_opponent_group:
                        for stone in visited:
                            new_board[stone[0]][stone[1]] = self.go.UNOCCUPIED

        return new_board

    def calculate_magic_number(self, game_board, piece):
        def count_m1(game_sub_board, piece):
            if (
                (
                    game_sub_board[0][0] == piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] != piece
                    and game_sub_board[1][1] != piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] == piece
                    and game_sub_board[1][0] != piece
                    and game_sub_board[1][1] != piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] != piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] != piece
                    and game_sub_board[1][1] == piece
                )
            ):
                return 1
            else:
                return 0

        def count_m2(game_sub_board, piece):
            if (
                game_sub_board[0][0] == piece
                and game_sub_board[0][1] != piece
                and game_sub_board[1][0] != piece
                and game_sub_board[1][1] == piece
            ) or (
                game_sub_board[0][0] != piece
                and game_sub_board[0][1] == piece
                and game_sub_board[1][0] == piece
                and game_sub_board[1][1] != piece
            ):
                return 1
            else:
                return 0

        def count_m3(game_sub_board, piece):
            if (
                (
                    game_sub_board[0][0] == piece
                    and game_sub_board[0][1] == piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] != piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] == piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] == piece
                )
                or (
                    game_sub_board[0][0] == piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] == piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] == piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] == piece
                )
            ):
                return 1
            else:
                return 0

        opponent_piece = self.go.get_opponent_piece(piece)
        new_game_board = np.zeros((self.go.board_size + 2, self.go.board_size + 2), dtype=int)
        # First copy the original game_state
        for i in range(self.go.board_size):
            for j in range(self.go.board_size):
                new_game_board[i + 1][j + 1] = game_board[i][j]

        m1_piece = 0
        m2_piece = 0
        m3_piece = 0
        m1_opponent_piece = 0
        m2_opponent_piece = 0
        m3_opponent_piece = 0

        for i in range(self.go.board_size):
            for j in range(self.go.board_size):
                new_game_sub_board = new_game_board[i : i + 2, j : j + 2]
                m1_piece += count_m1(new_game_sub_board, piece)
                m2_piece += count_m2(new_game_sub_board, piece)
                m3_piece += count_m3(new_game_sub_board, piece)
                m1_opponent_piece += count_m1(new_game_sub_board, opponent_piece)
                m2_opponent_piece += count_m2(new_game_sub_board, opponent_piece)
                m3_opponent_piece += count_m3(new_game_sub_board, opponent_piece)

        return (
            m1_piece - m3_piece + 2 * m2_piece - (m1_opponent_piece - m3_opponent_piece + 2 * m2_opponent_piece)
        ) / 4


if __name__ == "__main__":

    go = GO(BOARD_SIZE, INPUT_FILE_PATH, {"BLACK_PIECE": BLACK_PIECE, "WHITE_PIECE": WHITE_PIECE, "UNOCCUPIED": UNOCCUPIED}, X_CHANGES, Y_CHANGES)
    piece_type, previous_board, current_board = go.piece, go.previous_board, go.current_board
    step = load_game_info(previous_board, current_board)

    my_player = MyPlayer(go, piece_type, previous_board, current_board)

    my_player.make_a_move(SEARCH_DEPTH, BRANCHING_FACTOR, step)
