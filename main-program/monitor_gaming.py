from main import BoardState, Status
import numpy as np
from util import has_won, other_side, SquareState
from printer import send_gcode_script

def verify_movement(sm):
    if np.array_equal(sm.board_status.squares, sm.prev_board_status.squares):
        print('你还没有放置棋子')
        return '你还没有放置棋子'
    
    diff = np.subtract(sm.prev_board_status.squares, sm.board_status.squares)

    diff_count = np.count_nonzero(diff)
    # 统计非零元素的数量
    if diff_count == 2 and np.sum(sm.board_status.squares) == np.sum(sm.prev_board_status.squares):
        print('移动了棋子')
        diff = np.where(sm.board_status.squares != sm.prev_board_status.squares)[0]
        for i in diff:
            if sm.board_status.squares[i] == SquareState.Empty:
                f = i
            else:
                t = i
        print(f'from {f} to {t}')
        f = sm.from_board_to_frame(np.array([f % 3 + 0.5, f // 3 + 0.5]))
        t = sm.from_board_to_frame(np.array([t % 3 + 0.5, t // 3 + 0.5]))

        sm.handle_event('go_to_grab', position=t, target_position=f)

        # Do something
        return f'你移动了棋子'
    if diff_count >= 2:
        print('你操作了多个棋子')
        #todo: 给它放回去
        return '你操作了多个棋子'
    elif diff_count == 1:
        diff_idx = np.where(diff != 0)[0][0]
        if sm.prev_board_status.squares[diff_idx] != 0:
            print('不可以替换棋子')
            return '不可以替换棋子'
        if sm.side == sm.board_status.squares[diff_idx]:
            print('不能移动对方棋子')
            return '不能移动对方棋子'
        elif sm.board_status.squares[diff_idx] == SquareState.Empty:
            print('不能悔棋')
            return '禁止悔棋'
        else:
            print('可行')
            # cal_game(sm, sm.board_status)
    return None

def cal_game(sm, nbs, turn):
    print(f'处理第 {sm.action_count} 手')

    sm.action_count += 1
    sm.prev_board_status = sm.board_status

    if has_won(sm.board_status.squares, sm.side):
        print('Win')
        sm.handle_event('win')
    elif has_won(sm.board_status.squares, other_side(sm.side)):
        print('Lose')
        sm.handle_event('lose')
    else:
        # 判断结束没有
        if np.count_nonzero(nbs.squares) == 9:
            # 游戏结束
            print('GAME OVER')
            sm.handle_event('draw')
        else:
            if turn == 'our':
                sm.handle_event('their_turn')
            else:
                sm.handle_event('our_turn')

    return None


def monitor_gaming_loop(sm):
    return
    # if np.array_equal(sm.board_status.squares, sm.prev_board_status.squares):
    #     return
    
    # diff = np.subtract(sm.prev_board_status.squares, sm.board_status.squares)

    # diff_count = np.count_nonzero(diff)
    # # 统计非零元素的数量
    # if diff_count > 1:
    #     print('卧槽，一次放多子！')
    #     #todo: 给它放回去
    #     return
    # elif diff_count == 1:
    #     print('更新', sm.board_status.squares)
    #     diff_idx = np.where(diff != 0)[0][0]
    #     if sm.side == sm.board_status.squares[diff_idx]:
    #         print('犯规')
    #     elif sm.board_status.squares[diff_idx] == SquareState.Empty:
    #         print('不能悔棋')
    #     else:
    #         print('可行')
    #         sm.board_status = sm.board_status

    #         sm.handle_event('their_turn_finished')
    #         # cal_game(sm, sm.board_status)
