from copy import deepcopy

import numpy as np

#############################################################
### Go Class - Defines the Game Rules
#############################################################
class MyGO:
    def __init__(
        self, game_board_size: int, input_file_path: str, representations, horizontal_changes, vertical_changes
    ):
        self.game_board_size: int = game_board_size
        self.input_file_path = input_file_path
        self.BLACK_PIECE = representations["BLACK_PIECE"]
        self.WHITE_PIECE = representations["WHITE_PIECE"]
        self.UNOCCUPIED_SYMBOL = representations["UNOCCUPIED_SYMBOL"]
        self.HORIZONTAL_CHANGES = horizontal_changes
        self.VERTICAL_CHANGES = vertical_changes
        self.verbose = True

        self.max_move: int = self.game_board_size * self.game_board_size - 1  # Calculate the maximum number of moves
        self._init_board(self.game_board_size)

    def _init_board(self, n):
        """
        Initialize a board with size n*n.

        :param n: width and height of the board.
        :return: None.
        """
        with open(self.input_file_path, mode="r") as input_file:
            game_info = [input_file_line.strip() for input_file_line in input_file.readlines()]

            # piece = 1 => we are black, piece = 2 => we are white
            piece = int(game_info[0])

            previous_board = np.zeros((self.game_board_size, self.game_board_size), dtype=np.int32)
            current_board = np.zeros((self.game_board_size, self.game_board_size), dtype=np.int32)

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

    def copy_board(self, board):
        return deepcopy(board)

    def detect_neighbor(self, i, j):
        """
        Detect all the neighbors of a given stone.

        :param i: row number of the board.
        :param j: column number of the board.
        :return: a list containing the neighbors row and column (row, column) of position (i, j).
        """
        board = self.current_board
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
        board = self.current_board
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
        board = self.current_board
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
        board = self.current_board
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
        board = self.current_board
        for piece in positions:
            board[piece[0]][piece[1]] = 0
        self.update_board(board)

    def update_board(self, new_board):
        """
        Update the board with new_board

        :param new_board: new board.
        :return: None.
        """
        self.current_board = new_board

    def valid_place_check(self, i, j, piece_type, test_check=False):
        """
        Check whether a placement is valid.

        :param i: row number of the board.
        :param j: column number of the board.
        :param piece_type: 1(white piece) or 2(black piece).
        :param test_check: boolean if it's a test check.
        :return: boolean indicating whether the placement is valid.
        """
        board = self.current_board
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
        test_go = deepcopy(self)
        test_board = test_go.current_board

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
            if self.died_pieces and (self.previous_board == test_go.current_board).sum() != 0:
                if verbose:
                    print("Invalid placement. A repeat move not permitted by the KO rule.")
                return False
        return True
