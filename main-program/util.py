import numpy as np

class SquareState():
    Empty = 0
    Black = 1
    White = 2

def fix_rotation(rect):
    (x, y) = rect[1][0]- rect[0][0]
    a = np.arctan2(y, x)
    if a > (np.pi * 3 / 4):
        rect = np.roll(rect, -1, axis=0)
    return rect

def has_won(board, player):
    board = board.reshape((3, 3))

    for n in range(3):
        if all(board[n][m] == player for m in range(3)) or all(board[m][n] == player for m in range(3)):
            return True
    # Check diagonals for a win
    if all(board[i][i] == player for i in range(3)) or all(board[i][2-i] == player for i in range(3)):
        return True
    return False


def other_side(state):
    if state == SquareState.Black:
        return SquareState.White
    elif state == SquareState.White:
        return SquareState.Black
    return state
