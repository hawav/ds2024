import cv2
import numpy as np
import math
import util

# def nothing(x):
#     pass

# 创建窗口
# cv2.namedWindow('Thresholding')

# 创建滑动条
# cv2.createTrackbar('H Min', 'Thresholding', 0, 179, nothing)
# cv2.createTrackbar('H Max', 'Thresholding', 179, 179, nothing)
# cv2.createTrackbar('S Min', 'Thresholding', 0, 255, nothing)
# cv2.createTrackbar('S Max', 'Thresholding', 255, 255, nothing)
# cv2.createTrackbar('V Min', 'Thresholding', 0, 255, nothing)
# cv2.createTrackbar('V Max', 'Thresholding', 255, 255, nothing)

# cv2.setTrackbarPos('H Min', 'Thresholding', 10)
# cv2.setTrackbarPos('H Max', 'Thresholding', 30)
# cv2.setTrackbarPos('S Min', 'Thresholding', 140)

# angle = 0.
# center = (0, 0)
# scale = 1.0  # 缩放比例

board = None
img = None
maxDiagonal = None

def find_board(frame, sm):
    global maxDiagonal
    # h_min = cv2.getTrackbarPos('H Min', 'Thresholding')
    # h_max = cv2.getTrackbarPos('H Max', 'Thresholding')
    # s_min = cv2.getTrackbarPos('S Min', 'Thresholding')
    # s_max = cv2.getTrackbarPos('S Max', 'Thresholding')
    # v_min = cv2.getTrackbarPos('V Min', 'Thresholding')
    # v_max = cv2.getTrackbarPos('V Max', 'Thresholding')

    # # 定义HSV阈值范围
    # lower_green = np.array([h_min, s_min, v_min])
    # upper_green = np.array([h_max, s_max, v_max])

    if sm.should_update():
        lower_green = np.array([168, 128  , 100])
        upper_green = np.array([179, 255, 250])

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建掩模
        mask = cv2.inRange(hsv, lower_green, upper_green)
        cv2.imshow('mask', mask)

        spinning_capture = cv2.bitwise_and(frame, frame, mask=mask)

        # 寻找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
        # 找到长方形轮廓
        # indicators = []
        for cnt in contours:
            indicator = cv2.approxPolyDP(cnt, 0.07 * cv2.arcLength(cnt, True), True)
            area = cv2.contourArea(indicator)
            cv2.drawContours(spinning_capture, [indicator], 0, (0, 255, 0), 2)
            if len(indicator) == 4 and area > 1500:
                global board
                board = indicator
                break
                indicators.append(indicator)
                cv2.drawContours(spinning_capture, [indicator], 0, (0, 255, 0), 2)
        # indicators = sorted(indicators[:2], key=lambda x: x[0][0][0])
        cv2.imshow('Spinning Capture Status', spinning_capture)
        if board is None:
            return None
        
        board = util.fix_rotation(board)

        sm.board_rect = board

        # 计算四个对角线的长度
        diagonal1 = ((board[0][0][0] - board[1][0][0]) ** 2) + ((board[0][0][1] - board[1][0][1]) ** 2)
        diagonal2 = ((board[2][0][0] - board[3][0][0]) ** 2) + ((board[2][0][1] - board[3][0][1]) ** 2)
        diagonal3 = ((board[0][0][0] - board[3][0][0]) ** 2) + ((board[0][0][1] - board[3][0][1]) ** 2)
        diagonal4 = ((board[1][0][0] - board[2][0][0]) ** 2) + ((board[1][0][1] - board[2][0][1]) ** 2)

        # 找到最大的对角线长度作为正方形的边长
        maxDiagonal = int(np.sqrt(max(diagonal1, diagonal2, diagonal3, diagonal4)))

        # 定义目标矩形的四个顶点
        dstQuad = np.float32([[0, 0], [0, maxDiagonal - 1], [maxDiagonal - 1, maxDiagonal - 1], [maxDiagonal - 1, 0]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(board.astype(np.float32), dstQuad)

        sm.MN = np.dot(sm.M, M)
        sm.N = M

    if not maxDiagonal:
        return None

    # 应用透视变换
    img = cv2.warpPerspective(frame, sm.N, (maxDiagonal, maxDiagonal))

    return img

    # 获取图像尺寸
    height, width = frame.shape[:2]

    T = np.ones((2, 2))
    TINV = np.ones((2, 2))
    anchor = (0, 0)

    if len(indicators) >= 2:
        largest_triangle1 = indicators[0]
        largest_triangle2 = indicators[1]

        M1 = cv2.moments(largest_triangle1)
        M2 = cv2.moments(largest_triangle2)

        cnt = np.vstack([largest_triangle1, largest_triangle2])
        indicator = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), False)
        cv2.drawContours(spinning_capture, [indicator], 0, (255, 255, 255), 2)

        if M1["m00"] != 0 and M2["m00"] != 0:
            # 计算左边定位长方形的质心
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])
            anchor = (cX1, cY1)

            # 计算右边定位长方形的质心
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])

            ux = (cX2 - cX1, cY2 - cY1)
            uy = (-(cY2 - cY1), cX2 - cX1)

            T = np.column_stack((ux, uy))
            TINV = np.linalg.inv(T)

            # 在原图上绘制最大的两个三角形轮廓
            cv2.drawContours(spinning_capture, [largest_triangle1], 0, (0, 0, 255), 2)
            cv2.drawContours(spinning_capture, [largest_triangle2], 0, (0, 0, 255), 2)

            # 在图像上绘制质心
            cv2.circle(spinning_capture, (cX1, cY1), 5, (255, 0, 0), -1)
            cv2.circle(spinning_capture, (cX2, cY2), 5, (255, 0, 0), -1)
            global angle
            global center
            global scale
            angle = math.atan2(cY2 - cY1, cX2 - cX1)
            if not (angle > -np.pi / 2 and angle < np.pi / 2):
                angle = angle + np.pi

            # 计算旋转矩阵
            center = ((cX2 + cX1) // 2, (cY2 + cY1) // 2)
            angle = angle / np.pi * 180  # 旋转角度
            scale = 1.0  # 缩放比例
    
    cv2.imshow('Spinning Capture Status', spinning_capture)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    return (indicator, center, anchor, TINV, angle, M)