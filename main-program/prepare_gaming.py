import cv2
import math
from main import Square, Status
from util import SquareState

# squares = []
# squares.append(Square(-41 / 32, 1.5, SquareState.Black))
# squares.append(Square(3 + 40 / 32, 1.5, SquareState.White))
# for i in range(1, 3):
#     squares.append(Square(-41 / 32, 1.5 - i, SquareState.Black))
#     squares.append(Square(3 + 40 / 32, 1.5 - i, SquareState.White))
#     squares.append(Square(-41 / 32, 1.5 + i, SquareState.Black))
#     squares.append(Square(3 + 40 / 32, 1.5 + i, SquareState.White))

def check_game_ready(sm, squares):
    if len(sm.black_pieces) == 5 and len(sm.white_pieces) == 5 and sm.status == Status.PREPARE_GAMING:
        # 棋子数量正确，判断棋子位置
        is_ok = True
        for black in sm.black_pieces:
            b = black.get('board')
            closest_black_square = next((square for square in squares if square.is_close(b) and square.expect == SquareState.Black), None)
            if closest_black_square is None:
                return '仍有黑棋子需要挪到备战区'

        for white in sm.white_pieces:
            b = white.get('board')
            closest_white_square = next((square for square in squares if square.is_close(b) and square.expect == SquareState.White), None)
            if closest_white_square is None:
                return '仍有白棋子需要挪到备战区'

        if is_ok:
            if sm.side == SquareState.Black:
                sm.handle_event('our_turn')
            else:
                sm.handle_event('their_turn')
            return None
        
    return '棋子数量不足'

def prepare_gaming_loop(sm, squares):
    pg = sm.frame.copy()
    sm.draw_point(pg, sm.from_board_to_frame((-41 / 32, 1.5)))
    sm.draw_point(pg, sm.from_board_to_frame((3 + 40 / 32, 1.5)))
    for i in range(1, 3):
        sm.draw_point(pg, sm.from_board_to_frame((-41 / 32, 1.5 - i)))
        sm.draw_point(pg, sm.from_board_to_frame((3 + 40 / 32, 1.5 - i)))
        sm.draw_point(pg, sm.from_board_to_frame((-41 / 32, 1.5 + i)))
        sm.draw_point(pg, sm.from_board_to_frame((3 + 40 / 32, 1.5 + i)))

    cv2.imshow('Prepare Gaming', pg)

    # 把棋子放在备战区
    print(len(sm.black_pieces), len(sm.white_pieces))
    
    check_game_ready(sm, squares)
        # for square in squares:
        #     closest_black_piece = next((piece for piece in sm.black_pieces if (square.x - piece.get('board')[0])**2 + (square.y - piece.get('board')[1])**2 < 0.5), None)
        #     if closest_black_piece is not None:
        #         if square.expect == SquareState.Black:
        #             print('好耶，有黑棋子摆好了')
        #         else:
        #             print('惨了，有黑棋子摆过来了')