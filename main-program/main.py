import cv2
import numpy as np
import math
from util import has_won, other_side, SquareState
# from spinning import find_spinning_indicator
from black_pieces import calc_pieces
from frame import calc_frame
from printer import send_gcode_script, standby
import threading
from human_detection import detect_human

class Square:
    def __init__(self, x, y, expect):
        self.x = x
        self.y = y
        self.expect = expect
        self.occupied = None

    def is_close(self, pos):
        return (self.x - pos[0])**2 + (self.y - pos[1])**2 < 0.5

SQUARES = []
SQUARES.append(Square(-28 / 32, 1.5, SquareState.Black))
SQUARES.append(Square(3 + 27 / 32, 1.5, SquareState.White))
for i in range(1, 3):
    SQUARES.append(Square(-28 / 32, 1.5 - i, SquareState.Black))
    SQUARES.append(Square(3 + 27 / 32, 1.5 - i, SquareState.White))
    SQUARES.append(Square(-28 / 32, 1.5 + i, SquareState.Black))
    SQUARES.append(Square(3 + 27 / 32, 1.5 + i, SquareState.White))

print(SQUARES)

class BoardState:
    def __init__(self):
        self.squares = np.full(9, SquareState.Empty)
    
bs = BoardState()

class Status():
    IDLE = 1
    PREPARE_GAMING = 2
    TheirTurn = 3
    OurTurn = 4

    WeWin = 5
    WeLose = 6
    Draw = 7

    STOPPED = -1

    
    GAMING = 7
    PLACING_PIECE = 8
    MONITOR_GAMING = 9

def computer_move(board, side):
    oside = other_side(side)

    # If it can win, take the winning move
    for idx in sm.get_empty_cells(board):
        board[idx] = side
        if has_won(board, side):
            print('take it because we can win')
            return idx
        board[idx] = SquareState.Empty

    # If the opponent can win, block them
    for idx in sm.get_empty_cells(board):
        board[idx] = oside
        if has_won(board, oside):
            board[idx] = side
            print('take it because we need to block them')
            return idx
        board[idx] = SquareState.Empty

    # If the center is free, take it
    if board[4] == SquareState.Empty:
        board[4] = side
        return 4

    # Otherwise, take a corner
    corners = [0, 2, 6, 8]
    for idx in corners:
        if board[idx] == SquareState.Empty:
            board[idx] = side
            print('take corner')
            return idx

    # If no corner is available, take an edge
    edges = [3, 1, 7, 5]
    for idx in edges:
        if board[idx] == SquareState.Empty:
            board[idx] = side
            print('take edge')
            return idx
        
def from_board_to_phy(position):
    x_scale = 0.9935
    y_scale = 0.9935
    x_offset = 0.5
    y_offset = 7.5

    frame_size = sm.frame.shape[0]
    phy_x = 350 * x_scale * position[0] / frame_size
    phy_y = 350 - y_scale * 350 * position[1] / frame_size

    phy_x += x_offset
    phy_y += y_offset

    if phy_y > 350:
        phy_y = 350
    if phy_x > 350:
        phy_x = 350
    if phy_y < 0:
        phy_y = 0
    if phy_x < 0:
        phy_x = 0

    return (phy_x, phy_y)
        
def execute_code_in_thread(frame_size, position, target_position, sm, end_event, end_status):
    (phy_x, phy_y) = from_board_to_phy(position)
    print('printer go to', phy_x, phy_y)

    target = from_board_to_phy(target_position)
    print(target)

    send_gcode_script("G90")
    send_gcode_script(f"G0 X{round(phy_x)} Y{round(phy_y)} Z30 F30000")
    # send_gcode_script(f"G0 X{round(phy_x)} Y{round(phy_y)} Z30 F10000")
    send_gcode_script(f"G0 Z8 F10000")
    # 使能电磁铁
    send_gcode_script(f"M106")
    # 回家喽
    # standby()
    send_gcode_script(f"G0 Z30 F10000")
    send_gcode_script(f"G0 X{round(target[0])} Y{round(target[1])} Z30 F10000")
    send_gcode_script(f"G0 Z8 F10000")
    # 失能电磁铁
    send_gcode_script(f"M107")
    standby()

    send_gcode_script(f"M400")

    sm.grabbing = False

    if end_status is not None:
        sm.status = end_status
    if end_event is not None:
        sm.handle_event(end_event)

class StateMachine:
    def __init__(self):
        self.status = Status.IDLE
        self.target_position = None  # 用于存储目标位置

        self.human = False

        self.white_pieces = np.empty(shape=2)
        self.black_pieces = np.empty(shape=2)

        self.M = None # 映射
        self.N = None # 映射
        self.MN = None # 映射

        self.img = None # 原始图像
        self.frame = None # 框图
        self.board = None # 棋盘图

        self.grabbing = False

        self.side = SquareState.Black

        self.board_status = bs
        self.prev_board_status = self.board_status

        self.action_count = 0

        self.black = True

    def update_board_status(self):
        # 更新 board_state
        nbs = BoardState()
        # 检查棋盘上的黑棋子以及白棋子，更新 bs
        for black in sm.black_pieces:
            (x, y) = np.floor(black.get('board')).astype(np.int32)
            if 0 <= x < 3 and 0 <= y < 3:
                i = 3 * y + x
                nbs.squares[i] = SquareState.Black
        for black in sm.white_pieces:
            (x, y) = np.floor(black.get('board')).astype(np.int32)
            if 0 <= x < 3 and 0 <= y < 3:
                i = 3 * y + x
                nbs.squares[i] = SquareState.White
        sm.board_status = nbs

    def from_frame_to_img(self, point):
        return np.dot(np.linalg.inv(self.M), np.append(point, 1))[:2]
    
    def from_board_to_frame(self, point):
        # X 大右 Y大下
        offset = (-5, -5.5)

        (width, height) = self.board.shape[:2]
        def pos(x, y):
            return (((32 * x + 56) * width / 206), (32 * y + 56) * height / 206)
        return np.dot(np.linalg.inv(self.N), np.append(pos(point[0], point[1]), 1))[:2] + offset
    
    def from_board_to_img(self, point):
        return self.from_frame_to_img(self.from_board_to_frame(point))

    def draw_point(self, target, point):
        cv2.circle(target, np.round(point).astype(np.int32), 5, (255, 255, 255), -1)

    # def get_one_free_piece_not_in_pending(self, side=None):
    #     if side == None:
    #         side = self.side
    #     for piece in sm.black_pieces if side == SquareState.Black else sm.white_pieces:
    #         b = piece.get('board')
    #         closest_black_square = next((square for square in SQUARES if square.is_close(b) and square.expect == side), None)
    #         if closest_black_square is not None:
    #             return piece
    #     return None
    
    def get_empty_cells(self, board):
        return [i for i in range(9) if board[i] == SquareState.Empty]

    def get_one_free_piece_not_in_board(self, side=None):
        if side == None:
            side = self.side
        if side == SquareState.Black:
            pieces = sm.black_pieces
        else:
            pieces = sm.white_pieces

        for piece in pieces:
            (x, y) = np.floor(piece.get('board')).astype(np.int32)
            if x < 0 or x >= 3 or y < 0 or y >= 3:
                return piece
        
        return None
    
    def should_update(self):
        return not self.human and (self.status is Status.IDLE or self.status is Status.PREPARE_GAMING)

    def handle_event(self, event, position=None, target_position=None, end_event = None, idx=None, side=None, end_status=None):
        print('event', event)
        if event == 'reset':
            self.handle_event('start')
        if event == 'start':
            # if self.status == Status.IDLE:
            self.status = Status.IDLE
            # if self.status == Status.IDLE:
            #     self.status = Status.IDLE
            # elif self.status == Status.STOPPED:
            #     self.status = Status.IDLE
            # else:
            #     print("Invalid state transition.")
        elif event == 'our_turn':
            self.status = Status.OurTurn
            send_gcode_script('SET_LED LED="sb_leds" INDEX=1 RED=0 GREEN=0 BLUE=0 WHITE=0 SYNC=1')
            # 下棋
            piece = sm.get_one_free_piece_not_in_board()    
            if piece:
                print(sm.board_status.squares)
                if sm.next_square is not None:
                    idx = sm.next_square
                    sm.next_square = None
                else:
                    idx = computer_move(sm.board_status.squares, self.side)
                print('我是电脑，我下', idx)
                phy = sm.from_board_to_frame(np.array([idx % 3 + 0.5, idx // 3 + 0.5]))
                sm.draw_point(sm.frame, phy)
                sm.board_status.squares[idx] = self.side

                self.status = Status.PLACING_PIECE

                sm.handle_event('go_to_grab', position=piece.get('frame'), target_position=phy, end_event='our_turn_finished')
            else:
                print('没棋子用了')
        elif event == 'our_turn_finished':
            send_gcode_script('SET_LED LED="sb_leds" INDEX=1 RED=0 GREEN=1 BLUE=1 WHITE=0 SYNC=1')
            from monitor_gaming import cal_game
            cal_game(self, sm.board_status, 'our')
        elif event == 'their_turn':
            self.status = Status.TheirTurn
        elif event == 'their_turn_finished':
            from monitor_gaming import cal_game
            cal_game(self, sm.board_status, 'their')
        elif event == 'put':
            if idx is not None and self.status != Status.PREPARE_GAMING:
                # 代放棋子
                phy = sm.from_board_to_frame(np.array([idx % 3 + 0.5, idx // 3 + 0.5]))

                if side is None:
                    side = self.side

                piece = sm.get_one_free_piece_not_in_board(side)
                if piece:
                    sm.handle_event('go_to_grab', position=piece.get('frame'), target_position=phy)
                else:
                    print('无棋可下')
        elif event == 'stop':
            self.status = Status.STOPPED
        elif event == 'error':
            self.status = Status.ERROR
        elif event == 'win':
            self.prev_board_status = self.board_status = BoardState()
            self.status = Status.WeWin
        elif event == 'lose':
            self.prev_board_status = self.board_status = BoardState()
            self.status = Status.WeLose
        elif event == 'draw':
            self.prev_board_status = self.board_status = BoardState()
            self.status = Status.Draw
        elif event == 'go_to_grab':  # 新增的事件
            if not self.grabbing:
                self.grabbing = True
                prev_status = self.status
                self.status = Status.PLACING_PIECE
                self.target_position = target_position  # 记录棋子的位置
                self.position = position  # 记录棋子的位置

                frame_size = sm.frame.shape[0]

                # 创建一个线程
                thread = threading.Thread(target=execute_code_in_thread, args=(frame_size, position, target_position, sm, end_event, prev_status))

                # 启动线程
                thread.start()

        else:
            print("Unknown event.")

    def get_status(self):
        return self.status

    def get_target_position(self):
        return self.target_position
    
# 示例用法
sm = StateMachine()

def is_circle(contour):
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    if area < 300:
        return False
    
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return False
    
    # 计算圆形度
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    
    # 使用轮廓近似来减少轮廓上的点数
    epsilon = 0.01 * perimeter  # 调整这个值以控制近似程度
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 计算近似轮廓的顶点数
    num_vertices = len(approx)
    
    # 检查近似轮廓是否接近圆形
    if num_vertices > 10:  # 如果顶点数大于6，可能接近圆形
        # 检查长宽比
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # 计算最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        # 计算长宽比
        width, height = rect[1]
        aspect_ratio = min(width, height) / max(width, height)
        
        # 如果长宽比接近 1，且圆形度足够接近 1，则认为是圆形
        if 0.8 < aspect_ratio < 1.2 and 0.5 < circularity < 1.5:
            return True
    
    return False


def show_video_stream(stream_url, x, y):
    sm.handle_event('start')
    # sm.status = Status.PREPARE_GAMING
    sm.x = x
    sm.y = y
    # 创建 VideoCapture 对象
    cap = cv2.VideoCapture(stream_url)

    # 检查是否成功打开摄像头
    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    while sm.status != Status.STOPPED:
        # 读取一帧
        ret, img = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        sm.img = img

        sm.human = detect_human(img)

        frame = calc_frame(img, sm)
        if frame is not None:
            sm.frame = frame
            handle_frame()

        cv2.imshow('Original', img)

        if not sm.human and sm.frame is not None and sm.M is not None and sm.N is not None and sm.board is not None:
            if not sm.grabbing:
                sm.update_board_status()

            # if sm.status == Status.PREPARE_GAMING:
            #     # from prepare_gaming import prepare_gaming_loop
            #     # prepare_gaming_loop(sm, SQUARES)
            #     pass
            # elif sm.status == Status.MONITOR_GAMING:
            #     from monitor_gaming import monitor_gaming_loop
            #     monitor_gaming_loop(sm)


        # if board is None:
        #     continue

        # # 计算四个对角线的长度
        # diagonal1 = ((board[0][0][0] - board[1][0][0]) ** 2) + ((board[0][0][1] - board[1][0][1]) ** 2)
        # diagonal2 = ((board[2][0][0] - board[3][0][0]) ** 2) + ((board[2][0][1] - board[3][0][1]) ** 2)
        # diagonal3 = ((board[0][0][0] - board[3][0][0]) ** 2) + ((board[0][0][1] - board[3][0][1]) ** 2)
        # diagonal4 = ((board[1][0][0] - board[2][0][0]) ** 2) + ((board[1][0][1] - board[2][0][1]) ** 2)

        # # 找到最大的对角线长度作为正方形的边长
        # maxDiagonal = int(np.sqrt(max(diagonal1, diagonal2, diagonal3, diagonal4)))

        # # 定义目标矩形的四个顶点
        # dstQuad = np.float32([[0, 0], [0, maxDiagonal - 1], [maxDiagonal - 1, maxDiagonal - 1], [maxDiagonal - 1, 0]])

        # # 计算透视变换矩阵
        # M = cv2.getPerspectiveTransform(board.astype(np.float32), dstQuad)

        # # 应用透视变换
        # board = cv2.warpPerspective(img, M, (maxDiagonal, maxDiagonal))

        # cv2.imshow('Board', board)

        # # print(width, height)

        # black_pieces = calc_black_pieces(img)
        # target = board.copy()
        # height, width = board.shape[:2]
        # def pos(x, y):
        #     return (((32 * (x + 0.5) + 56) * width / 206), (32 * (y + 0.5) + 56) * height / 206)
        # target_point = pos(0, 0)
        # cv2.circle(target, np.round(target_point).astype(np.int32), 5, (255, 255, 255), -1)
        # cv2.imshow('Target', target)
        # mapped = np.dot(np.linalg.inv(M), np.append(target_point, 1))[:2]

        # cv2.circle(img, np.round(mapped).astype(np.int32), 5, (255, 255, 255), -1)
        # cv2.imshow('Frame', img)
        # frame_height, frame_width = frame.shape[:2]
        # if len(black_pieces) > 0:
        #     if sm.get_status() != Status.GOING_TO_GRAB:
        #         print(black_pieces[0])
        #         sm.handle_event('go_to_grab', position=black_pieces[0], board_size=frame_width, target_position=mapped)
        
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)
        
#         (anchor, T, angle, M) = find_spinning_indicator(frame)

#         # 获取图像尺寸

#         # 计算旋转矩阵
#         # center = (width // 2, height // 2)
#         # angle = angle / np.pi * 180  # 旋转角度
#         # scale = 1.0  # 缩放比例
#         # M = cv2.getRotationMatrix2D(center, angle, scale)

#         rotated_frame = cv2.warpAffine(frame, M, (width, height))
#         cv2.imshow('Rotated Frame', rotated_frame)
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         edged = cv2.Canny(gray, 25, 300)
        
#         cv2.imshow('Edged', edged)

#         # kernel_size = 10
#         # kernel = np.ones((kernel_size, kernel_size), np.uint8)
#         # dilated_edges = cv2.dilate(edged, kernel, iterations=1)

#         # cv2.imshow('dilated_edges', dilated_edges)

#         # edged = dilated_edges
        
#         marker_frame = frame.copy()
        
#         # 预处理
# #         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
#         # 旋转图像
#         # rotated = cv2.warpAffine(frame, M, (width, height))
        
#         contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#         # contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         for cnt in contours:
#             # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

#             if is_square(cnt):
#                 area = cv2.contourArea(cnt)
#                 if area > 200000:
#                     cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)

#             if is_circle(cnt):
#                 center = cv2.moments(cnt)
#                 if center["m00"] != 0:
#                     x = int(center["m10"] / center["m00"])
#                     y = int(center["m01"] / center["m00"])
#                     # [x, y] = np.dot(M, [x, y, 1])
#                     center = (round(x), round(y))
#                     cv2.circle(frame, center, 5, (255, 255, 255), -1)

#                     (x, y) = np.dot(T, (x - anchor[0], y - anchor[1]))
#                     x = (x - (1/8)) / (2/8)
#                     text = f"({x:.3f}, {y:.3f})"
#                     position = (center[0] + 10, center[1] + 20)  # 文本位置
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     font_scale = 0.5
# #                     color = is_circle(cnt)
#                     color = (255, 255, 255)  # 白色
#                     thickness = 1
#                     line_type = cv2.LINE_AA

#                     cv2.putText(frame, text, position, font, font_scale, color, thickness, line_type)

#                     cv2.drawContours(marker_frame, cnt, -1, (0, 0, 255), 2)
                
#         cv2.imshow('Marker', marker_frame)

#         cv2.imshow('Rotation-Invariant Frame', frame)
            
#         # 180度旋转
#         # rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
#         rotated_frame = frame

        # 按 'q' 键退出循环
        if cv2.waitKey(1) == ord('q'):
            break

    send_gcode_script("M400")
    # 释放资源并关闭窗口
    cap.release()
    cv2.destroyAllWindows()

# 使用示例
# send_gcode_script("G28")

send_gcode_script("M107")
send_gcode_script("G90")
send_gcode_script("G0 X0 Y350 Z100 F20000")
send_gcode_script("M400")

def handle_frame():
    from board import find_board
    board = find_board(sm.frame, sm)
    if board is not None and sm.M is not None and sm.N is not None:
        sm.board = board
        handle_board()

    cv2.imshow('Frame', sm.frame)

def handle_board():
    target_point = np.array((sm.x, sm.y)) + 0.5
    cv2.circle(sm.board, np.round(target_point).astype(np.int32), 5, (255, 255, 255), -1)
    phy = sm.from_board_to_img(target_point)
    sm.draw_point(sm.img, phy)
    phy = sm.from_board_to_frame(target_point)
    sm.draw_point(sm.frame, phy)
    cv2.imshow('board', sm.board)

    black_pieces = sm.black_pieces = calc_pieces(sm.frame, sm, black=True)
    sm.white_pieces = calc_pieces(sm.frame, sm, black=False)
    # if not sm.human and len(black_pieces) > 0:
    #     nbs = BoardState()
    #     for black in black_pieces:
    #         (x, y) = black.get('board').astype(np.int32)
    #         if 0 <= x < 3 and 0 <= y < 3:
    #             i = 3 * y + x
    #             nbs.squares[i] = SquareState.Black

    #     bs = nbs
    #     # print(nbs.squares)

    #     for black in black_pieces:
    #         (x, y) = black.get('board')
    #         if x < 0 or x >= 3 or y < 0 or y >= 3:
    #             # black 是在外面的黑棋子
    #             # 寻找棋盘内空闲的格子
    #             empty_idx = np.where(bs.squares == SquareState.Empty)[0]
    #             if len(empty_idx) > 0:
    #                 idx = empty_idx[0]
    #                 phy = sm.from_board_to_frame(np.array([idx % 3 + 0.5, idx // 3 + 0.5]))
    #                 sm.draw_point(sm.frame, phy)

    #                 if sm.status != Status.IDLE:
    #                     break
                    
    #                 sm.handle_event('go_to_grab', position=black.get('frame'), target_position=phy)
    #                 break
    # if len(black_pieces) > 0:
    #     if sm.get_status() != Status.GOING_TO_GRAB:
            # print(black_pieces[0])
            # sm.handle_event('go_to_grab', position=black_pieces[0], target_position=phy)

if __name__ == '__main__':
    stream_url = "http://192.168.1.119/webcam/?action=stream"
    show_video_stream(stream_url, 0, 1)
    sm.status = Status.GAMING