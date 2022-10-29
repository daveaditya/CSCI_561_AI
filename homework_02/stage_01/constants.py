#############################################################
### Constants
#############################################################

# Game related file paths
# BASE_DIR = "./stage_01/init"
BASE_DIR = "."
INPUT_FILE_PATH = f"{BASE_DIR}/input.txt"
OUTPUT_FILE_PATH = f"{BASE_DIR}/output.txt"
GAME_INFO_FILE_PATH = f"{BASE_DIR}/game_info.json"

# Size of the board
GAME_BOARD_SIZE = 5

# Game board representations
UNOCCUPIED_SYMBOL = 0
BLACK_PIECE = 1
WHITE_PIECE = 2

# Komi for white player as per instructions
KOMI = 2.5

# Format: Right, Bottom, Left, Up
CHANGES = [(1, 0), (0, 1), (-1, 0), (0, -1)]

VALID_MOVE_ONE_CAPTURING = "ONE_CAPTURING"
VALID_MOVE_TWO_REGULAR = "TWO_REGULAR"
VALID_MOVE_THREE_SIZE = "THREE_SIZE"

# Player Constants
SEARCH_DEPTH = 4
BRANCHING_FACTOR = 28

SNAKE_CHECK_STEP_THRESHOLD = 8
INITIAL_MOVES_THRESHOLD = 8
MIDDLE_MOVES_THRESHOLD = 12
END_MOVES_THRESHOLD = 16

INITIAL_SEARCH_DEPTH = 10
MIDDLE_SEARCH_DEPTH = 16
END_SEARCH_DEPTH = 24

MY_SNAKE_SCORE = 15
OPPONENT_SNAKE_SCORE = 15