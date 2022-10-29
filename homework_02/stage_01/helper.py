import json

from constants import *

#############################################################
### Helper Functions
#############################################################


def load_game_info(previous_board, current_board):
    is_previous_game_a_start = True
    is_current_game_a_start = True

    for row_idx in range(GAME_BOARD_SIZE - 1):
        for col_idx in range(GAME_BOARD_SIZE - 1):
            if previous_board[row_idx][col_idx] != UNOCCUPIED_SYMBOL:
                is_previous_game_a_start = False
                is_current_game_a_start = False
                break
            elif current_board[row_idx][col_idx] != UNOCCUPIED_SYMBOL:
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

