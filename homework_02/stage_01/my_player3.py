import random

import numpy as np

from constants import *
from helper import *
from my_go import MyGO as GO

random.seed(42)

#############################################################
### My Player Implementation
### MiniMax with Alpha-Beta Pruning
#############################################################
class MyPlayer:
    def __init__(
        self, go: GO, piece: int, previous_board, current_board, step: int, snake_check_step_threshold: int, secondary_heuristics_threshold: int, reward
    ):
        self.type = "minimax_with_pruning_player"
        self.go: GO = go
        self.my_piece: int = piece
        self.opponent_piece: int = self.go.get_opponent_piece(self.my_piece)
        self.previous_board = previous_board
        self.current_board = current_board
        self.step = step
        self.snake_check_step_threshold = snake_check_step_threshold
        self.secondary_heuristics_threshold = secondary_heuristics_threshold

        self.REWARD = reward

    def make_a_move(self, search_depth: int, branching_factor: int, step: int):
        max_move, _ = self.maxmillians_move(
            piece=self.my_piece,
            game_board=self.current_board,
            search_depth=search_depth,
            branching_factor=branching_factor,
            current_depth=0,
            step=step,
            alpha=-np.inf,
            beta=np.inf,
            last_move=None,
            is_second_pass=False,
        )
        store_move(OUTPUT_FILE_PATH, max_move)

    def has_snake_move(self, piece: int, game_board) -> bool:
        # Match `P 0 0 P` or `P 0 0 0 P` horizontally, vertically and return true if it is present
        game_board_transposed = np.transpose(game_board)
        n_rows, n_cols = np.shape(game_board)

        small_snake = np.full((3, 4), piece, dtype=np.int32)
        small_snake[1, 1], small_snake[1, 2] = 0, 0

        big_snake = np.full((3, 5), piece, dtype=np.int32)
        big_snake[1, 1], big_snake[1, 2], big_snake[1, 3] = 0, 0, 0

        snake_counter = 0

        for i in range(n_rows - 3 + 1):
            for j in range(n_cols - 4 + 1):
                if j == 0:
                    if np.all(game_board_transposed[i : i + 3, j : j + 5] == big_snake):
                        snake_counter += 1

                if np.all(game_board[i : i + 3, j : j + 4] == small_snake):
                    snake_counter += 1

        return snake_counter

    def defense_heuristic(self, game_board, i, j):
        defense = 0
        for i_dash, j_dash in self.go.DEFENSE_INDICES:
            i_prime = i + i_dash
            j_prime = j + j_dash
            if (
                0 < i_prime < self.go.game_board_size
                and 0 < j_prime < self.go.game_board_size
                and game_board[i_prime, j_prime] == game_board[i, j]
            ):
                defense += 1
        return defense

    def count_dead(self, piece, game_board):
        dead_count = 0
        dead_indices = ()
        for i in range(self.go.game_board_size):
            for j in range(self.go.game_board_size):
                if game_board[i, j] == piece and (i, j) not in dead_indices:
                    if not self.find_liberty(piece, i, j, game_board):
                        dead_indices.add((i, j))
        return dead_count

    def calculate_score(self, piece_type, game_board, step):
        # My Heuristics
        piece_count = (game_board == piece_type).sum()
        piece_liberties = set()
        opponent_piece_type = self.go.get_opponent_piece(piece_type)
        opponent_piece_count = (game_board == opponent_piece_type).sum()
        opponent_liberties = set()

        # Give points for having piece on game board
        piece_score = 5 * (piece_count - opponent_piece_count)

        for i in range(0, self.go.game_board_size):
            for j in range(0, self.go.game_board_size):
                if game_board[i][j] == self.go.UNOCCUPIED_SYMBOL:
                    for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
                        i_prime = i + i_dash
                        j_prime = j + j_dash
                        if 0 <= i_prime < self.go.game_board_size and 0 <= j_prime < self.go.game_board_size:
                            if game_board[i_prime][j_prime] == piece_type:
                                piece_liberties.add((i, j))
                            elif game_board[i_prime][j_prime] == opponent_piece_type:
                                opponent_liberties.add((i, j))

        piece_liberties_count = len(piece_liberties)
        opponent_liberties_count = len(opponent_liberties)

        # Liberty based heuristics#1
        liberty_score = min(max((piece_liberties_count - opponent_liberties_count), -8), 8)

        # Calculate pieces on edges
        piece_edge_count = np.sum(
            [
                (game_board[0, :] == piece_type),  # First row / Top
                (game_board[:, self.go.game_board_size - 1] == piece_type),  # Last Column / Right
                (game_board[self.go.game_board_size - 1, :] == piece_type),  # Last row / Bottom
                (game_board[:, 0] == piece_type),  # First column / Left
            ]
        )
        middle_unoccupied_count = (game_board[1:-1, 1:-1] == self.go.UNOCCUPIED_SYMBOL).sum()

        # Scoring for middle and edge of the board
        edge_score = 9 * piece_edge_count * (middle_unoccupied_count / 9)

        # Dead count for opponent
        dead_score = 0
        if piece_type == self.my_piece:
            dead_score = 10 * self.count_dead(piece_type, game_board)

        # Check for the no liberty move
        snake_score = 0
        # Check for snake only if more than 8 moves have been played
        if step > self.snake_check_step_threshold:
            # Check if there are 10 or more pieces on board
            if (game_board == self.go.BLACK_PIECE).sum() >= 10 or (game_board == self.go.WHITE_PIECE).sum() >= 10:
                snake_move_count = self.has_snake_move(piece_type, game_board)
                if snake_move_count:
                    snake_score = (
                        MY_SNAKE_SCORE if piece_type == self.my_piece else OPPONENT_SNAKE_SCORE
                    ) * snake_move_count

        # secondary heuristics
        secondary_score = 0
        if step > self.secondary_heuristics_threshold:
            secondary_score = -4 * (self.secondary_heuristics(game_board, piece_type) - self.secondary_heuristics(game_board, opponent_piece_type))

        score = liberty_score + snake_score + piece_score + dead_score + secondary_score - edge_score
        if self.my_piece == self.go.WHITE_PIECE:
            score += KOMI

        return score

    def do_move(self, piece_type, move, board):
        new_board = self.go.copy_board(board)

        # Liberty and KO rules are already checked, and hence it is a valid move for this piece
        new_board[move[0]][move[1]] = piece_type

        # Delete opponent group if required
        for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
            i_prime = move[0] + i_dash
            j_prime = move[1] + j_dash

            if 0 <= i_prime < self.go.game_board_size and 0 <= j_prime < self.go.game_board_size:
                opponent_piece = self.go.get_opponent_piece(piece_type)

                if new_board[i_prime][j_prime] == opponent_piece:
                    # Perform Depth First Search at new location
                    stack = [(i_prime, j_prime)]
                    visited = set()
                    should_delete_opponent_group = True

                    while stack:
                        top_node = stack.pop()
                        visited.add(top_node)
                        for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
                            new_i_prime = top_node[0] + i_dash
                            new_j_prime = top_node[1] + j_dash

                            if (
                                0 <= new_i_prime < self.go.game_board_size
                                and 0 <= new_j_prime < self.go.game_board_size
                            ):
                                if (new_i_prime, new_j_prime) in visited:
                                    continue
                                elif new_board[new_i_prime][new_j_prime] == self.go.UNOCCUPIED_SYMBOL:
                                    should_delete_opponent_group = False
                                    break
                                elif (
                                    new_board[new_i_prime][new_j_prime] == opponent_piece
                                    and (new_i_prime, new_j_prime) not in visited
                                ):
                                    stack.append((new_i_prime, new_j_prime))

                    if should_delete_opponent_group:
                        for stone in visited:
                            new_board[stone[0]][stone[1]] = self.go.UNOCCUPIED_SYMBOL

        return new_board

    def secondary_heuristics(self, game_board, piece):
        def count_single(game_sub_board, piece):
            if (
                (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] != piece
                    and game_sub_board[1][1] == piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] != piece
                )
                or (
                    game_sub_board[0][0] != piece
                    and game_sub_board[0][1] == piece
                    and game_sub_board[1][0] != piece
                    and game_sub_board[1][1] != piece
                )
                or (
                    game_sub_board[0][0] == piece
                    and game_sub_board[0][1] != piece
                    and game_sub_board[1][0] != piece
                    and game_sub_board[1][1] != piece
                )
            ):
                return 1
            else:
                return 0

        def count_group_3(game_sub_board, piece):
            if (
                game_sub_board[0][0] != piece
                and game_sub_board[0][1] == piece
                and game_sub_board[1][0] == piece
                and game_sub_board[1][1] != piece
            ) or (
                game_sub_board[0][0] == piece
                and game_sub_board[0][1] != piece
                and game_sub_board[1][0] != piece
                and game_sub_board[1][1] == piece
            ):
                return 1
            else:
                return 0

        def count_diagonal(game_sub_board, piece):
            if (
                (
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
                or (
                    game_sub_board[0][0] == piece
                    and game_sub_board[0][1] == piece
                    and game_sub_board[1][0] == piece
                    and game_sub_board[1][1] != piece
                )
            ):
                return 1
            else:
                return 0

        # opponent_piece = self.go.get_opponent_piece(piece)

        # Duplicate game board with extra places on each corner
        new_game_board = np.zeros((self.go.game_board_size + 2, self.go.game_board_size + 2), dtype=int)
        new_game_board[1:-1, 1:-1] = game_board

        single_piece = 0
        group_3_piece = 0
        diagonal_piece = 0
        # single_opponent_piece = 0
        # group_3_opponent_piece = 0
        # diagonal_opponent_piece = 0

        for i in range(self.go.game_board_size):
            for j in range(self.go.game_board_size):
                new_game_sub_board = new_game_board[i : i + 2, j : j + 2]
                single_piece += count_single(new_game_sub_board, piece)
                group_3_piece += count_group_3(new_game_sub_board, piece)
                diagonal_piece += count_diagonal(new_game_sub_board, piece)
                # single_opponent_piece += count_single(new_game_sub_board, opponent_piece)
                # group_3_opponent_piece += count_group_3(new_game_sub_board, opponent_piece)
                # diagonal_opponent_piece += count_diagonal(new_game_sub_board, opponent_piece)

        return (
            single_piece - diagonal_piece + 2 * group_3_piece 
        ) / 4 # - (single_opponent_piece - diagonal_opponent_piece + 2 * group_3_opponent_piece)

    def find_valid_moves(self, piece_type, game_board):
        valid_moves_list = {
            VALID_MOVE_ONE_CAPTURING: list(),
            VALID_MOVE_TWO_REGULAR: list(),
            VALID_MOVE_THREE_SIZE: list(),
        }

        for i in range(self.go.game_board_size):
            for j in range(self.go.game_board_size):
                if game_board[i][j] == self.go.UNOCCUPIED_SYMBOL:
                    if self.find_liberty(piece_type, i, j, game_board):
                        # Check `KO` rule
                        if not self.check_for_ko(i, j):
                            # If in corners has high liberty
                            if i == 0 or j == 0 or i == self.go.game_board_size - 1 or j == self.go.game_board_size - 1:
                                valid_moves_list[VALID_MOVE_THREE_SIZE].append((i, j))
                            else:
                                valid_moves_list.get(VALID_MOVE_TWO_REGULAR).append((i, j))
                    # Check for capture
                    else:
                        for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
                            i_prime = i + i_dash
                            j_prime = j + j_dash
                            if 0 <= i_prime < self.go.game_board_size and 0 <= j_prime < self.go.game_board_size:
                                opponent_piece = self.go.get_opponent_piece(piece_type)
                                if game_board[i_prime][j_prime] == opponent_piece:
                                    # Opponent has no liberty after this rule, so we can capture them , hence we can do this move.
                                    new_game_board = self.go.copy_board(game_board)
                                    new_game_board[i][j] = piece_type
                                    if not self.find_liberty(opponent_piece, i_prime, j_prime, new_game_board):
                                        # Check for `KO` rule
                                        if not self.check_for_ko(i, j):
                                            valid_moves_list[VALID_MOVE_ONE_CAPTURING].append((i, j))
                                        break

                        # Reaching this point means all of the neighbors have liberty

        valid_moves_list = [
            *valid_moves_list[VALID_MOVE_ONE_CAPTURING],
            *valid_moves_list[VALID_MOVE_TWO_REGULAR],
            *valid_moves_list[VALID_MOVE_THREE_SIZE],
        ]

        return valid_moves_list

    def find_liberty(self, piece, i, j, board):
        stack = [(i, j)]
        visited = set()
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
                i_prime = top_node[0] + i_dash
                j_prime = top_node[1] + j_dash
                if 0 <= i_prime < self.go.game_board_size and 0 <= j_prime < self.go.game_board_size:
                    if (i_prime, j_prime) in visited:
                        continue
                    elif board[i_prime][j_prime] == self.go.UNOCCUPIED_SYMBOL:
                        return True
                    elif board[i_prime][j_prime] == piece and (i_prime, j_prime) not in visited:
                        stack.append((i_prime, j_prime))
        return False

    def check_for_ko(self, i, j):
        if self.previous_board[i][j] != self.my_piece:
            return False

        new_game_board = self.go.copy_board(self.current_board)
        new_game_board[i][j] = self.my_piece
        opponent_i, opponent_j = self.opponents_move()

        for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
            i_prime = i + i_dash
            j_prime = j + j_dash
            if i_prime == opponent_i and j_prime == opponent_j:
                # If no liberty delete all opponent group
                if not self.find_liberty(self.opponent_piece, i_prime, j_prime, new_game_board):
                    # Delete all of the group
                    self.delete_group(self.opponent_piece, i_prime, j_prime, new_game_board)

        return np.array_equal(new_game_board, self.previous_board)

    def opponents_move(self):
        if np.array_equal(self.current_board, self.previous_board):
            return None
        locations = np.argwhere(
            np.logical_and(self.current_board != self.previous_board, self.current_board != self.go.UNOCCUPIED_SYMBOL)
        )
        return locations[0]

    def delete_group(self, piece, i, j, game_board):
        stack = [(i, j)]
        visited = set()

        # Perform Depth First Search
        while stack:
            top_node = stack.pop()
            visited.add(top_node)
            game_board[top_node[0], top_node[1]] = self.go.UNOCCUPIED_SYMBOL
            for i_dash, j_dash in self.go.NEIGHBOR_INDICES:
                i_prime = top_node[0] + i_dash
                j_prime = top_node[1] + j_dash
                if 0 <= i_prime < self.go.game_board_size and 0 <= j_prime < self.go.game_board_size:
                    if (i_prime, j_prime) in visited:
                        continue
                    elif game_board[i_prime][j_prime] == piece:
                        stack.append((i_prime, j_prime))

        return game_board

    def maxmillians_move(
        self,
        piece,
        game_board,
        search_depth,
        branching_factor,
        current_depth,
        step,
        alpha,
        beta,
        last_move,
        is_second_pass,
    ):
        if is_second_pass:
            return self.calculate_score(piece, game_board, step)
        if search_depth == current_depth or step + current_depth == self.go.max_move:
            return self.calculate_score(piece, game_board, step)

        is_second_pass = False
        max_move_value = -np.inf
        max_move = None
        valid_moves = self.find_valid_moves(piece, game_board)
        valid_moves.append((-1, -1))

        if last_move == (-1, -1):
            is_second_pass = True

        for valid_move in valid_moves[:branching_factor]:
            if valid_move == (-1, -1):
                new_game_board = self.go.copy_board(game_board)
            else:
                new_game_board = self.do_move(piece, valid_move, game_board)

            # Create new game board
            opponent_piece = self.go.get_opponent_piece(piece)
            min_move_value = self.mindys_move(
                piece=opponent_piece,
                game_board=new_game_board,
                search_depth=search_depth,
                branching_factor=branching_factor,
                current_depth=current_depth + 1,
                step=step,
                alpha=alpha,
                beta=beta,
                last_move=valid_move,
                is_second_pass=is_second_pass,
            )

            # Defense heuristic
            # defense_score = self.defense_heuristic(game_board, valid_move[0], valid_move[1])

            # Reward the move
            # reward = self.REWARD[valid_move[0]][valid_move[1]] * 0.1

            # max_move_value += defense_score

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
        piece,
        game_board,
        search_depth,
        branching_factor,
        current_depth,
        step,
        alpha,
        beta,
        last_move,
        is_second_pass,
    ):
        if search_depth == current_depth:
            return self.calculate_score(piece, game_board, step)
        if step + current_depth == self.go.max_move or is_second_pass:
            return self.calculate_score(self.my_piece, game_board, step)

        is_second_pass = False
        min_move_value = np.inf

        valid_moves = self.find_valid_moves(piece, game_board)
        valid_moves.append((-1, -1))

        if last_move == (-1, -1):
            is_second_pass = True

        for valid_move in valid_moves[:branching_factor]:
            if valid_move == (-1, -1):
                new_game_board = self.go.copy_board(game_board)
            else:
                new_game_board = self.do_move(piece, valid_move, game_board)

            # Create new game board
            opponent_piece = self.go.get_opponent_piece(piece)
            max_move_value = self.maxmillians_move(
                piece=opponent_piece,
                game_board=new_game_board,
                search_depth=search_depth,
                branching_factor=branching_factor,
                current_depth=current_depth + 1,
                step=step,
                alpha=alpha,
                beta=beta,
                last_move=valid_move,
                is_second_pass=is_second_pass,
            )

            if max_move_value < min_move_value:
                min_move_value = max_move_value

            if min_move_value <= alpha:
                return min_move_value

            beta = min(beta, min_move_value)

        return min_move_value


if __name__ == "__main__":

    go = GO(
        GAME_BOARD_SIZE,
        INPUT_FILE_PATH,
        {"BLACK_PIECE": BLACK_PIECE, "WHITE_PIECE": WHITE_PIECE, "UNOCCUPIED_SYMBOL": UNOCCUPIED_SYMBOL},
        NEIGHBOR_INDICES,
        DEFENSE_INDICES,
    )
    piece_type, previous_board, current_board = go.piece, go.previous_board, go.current_board
    step = load_game_info(previous_board, current_board)

    my_player = MyPlayer(
        go,
        piece_type,
        previous_board,
        current_board,
        step,
        snake_check_step_threshold=SNAKE_CHECK_STEP_THRESHOLD,
        secondary_heuristics_threshold=SECONDARY_HEURISTICS_THRESHOLD,
        reward=REWARD,
    )

    # Increase the search depth as we go ahead in the game.
    search_depth = SEARCH_DEPTH
    if step < INITIAL_MOVES_THRESHOLD:
        search_depth = INITIAL_SEARCH_DEPTH
    elif INITIAL_MOVES_THRESHOLD < step < MIDDLE_MOVES_THRESHOLD:
        search_depth = MIDDLE_SEARCH_DEPTH
    else:
        search_depth = END_SEARCH_DEPTH

    my_player.make_a_move(SEARCH_DEPTH, BRANCHING_FACTOR, step)
