import cv2
import numpy as np
from pieces import is_piece

cv2.namedWindow('Black Pieces Thresholding')

def nothing(x):
    pass

# 创建滑动条
cv2.createTrackbar('H Min', 'Black Pieces Thresholding', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Black Pieces Thresholding', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Black Pieces Thresholding', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Black Pieces Thresholding', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Black Pieces Thresholding', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Black Pieces Thresholding', 255, 255, nothing)

cv2.setTrackbarPos('H Min', 'Black Pieces Thresholding', 0)
cv2.setTrackbarPos('H Max', 'Black Pieces Thresholding', 179)
cv2.setTrackbarPos('S Min', 'Black Pieces Thresholding', 0)
cv2.setTrackbarPos('S Max', 'Black Pieces Thresholding', 110)
cv2.setTrackbarPos('V Min', 'Black Pieces Thresholding', 0)
cv2.setTrackbarPos('V Max', 'Black Pieces Thresholding', 120)

# DEBUG = True
DEBUG = False

def calc_pieces(frame, sm, black=True):
    if black:
        h_min = cv2.getTrackbarPos('H Min', 'Black Pieces Thresholding')
        h_max = cv2.getTrackbarPos('H Max', 'Black Pieces Thresholding')
        s_min = cv2.getTrackbarPos('S Min', 'Black Pieces Thresholding')
        s_max = cv2.getTrackbarPos('S Max', 'Black Pieces Thresholding')
        v_min = cv2.getTrackbarPos('V Min', 'Black Pieces Thresholding')
        v_max = cv2.getTrackbarPos('V Max', 'Black Pieces Thresholding')

        # 定义HSV阈值范围
        lower_green = np.array([h_min, s_min, v_min])
        upper_green = np.array([h_max, s_max, v_max])
        # pass
        # lower_green = np.array([0, 0, 0])
        # upper_green = np.array([179, 110, 120])
    else:
        lower_green = np.array([0, 0, 160])
        upper_green = np.array([179, 70, 255])

    marker_frame = frame.copy()

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.bitwise_not(frame)

    # 创建掩模
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # 创建一个与图像大小相匹配的掩码
    mask = cv2.bitwise_and(np.zeros(hsv.shape[:2], dtype=np.uint8), mask)

    # 在掩码上绘制矩形
    cv2.fillPoly(mask, [sm.board_rect], color=255)

    cv2.imshow(f'Pieces MASK {black}', mask)

    capture = cv2.bitwise_and(frame, frame, mask=mask)

    # gray_image = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)

    # circles = cv2.HoughCircles(
    #     gray_image,
    #     cv2.HOUGH_GRADIENT,
    #     dp=1,
    #     minDist=20,
    #     param1=5,
    #     param2=3,
    #     minRadius=1,
    #     maxRadius=10
    # )

    # # 确保找到了一些圆形
    # if circles is not None:
    #     # 将检测到的圆形坐标转换为整数
    #     circles = np.round(circles[0, :]).astype("int")

    #     # 在原图上绘制圆形
    #     for (x, y, r) in circles:
    #         # 绘制外圆
    #         cv2.circle(capture, (x, y), r, (0, 255, 0), 2)
    #         # 绘制圆心
    #         cv2.circle(capture, (x, y), 2, (0, 0, 255), 3)


    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    black_pieces = []

    for cnt in contours:
        cv2.drawContours(capture, [cnt], -1, (0, 255, 0), 2)

        if DEBUG:
            center = cv2.moments(cnt)
            if center["m00"] != 0:
                x = int(center["m10"] / center["m00"])
                y = int(center["m01"] / center["m00"])
                # [x, y] = np.dot(M, [x, y, 1])
                center = (round(x), round(y))

                color = is_piece(cnt)
                if color != (0, 0, 0):
                    area = cv2.contourArea(cnt)

                    text = f"({x:.3f}, {y:.3f}) A={area:.3f}"
                    position = (center[0] + 10, center[1] + 20)  # 文本位置
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    line_type = cv2.LINE_AA

                    cv2.putText(marker_frame, text, position, font, font_scale, color, thickness, line_type)
                    cv2.drawContours(marker_frame, cnt, -1, (0, 0, 255), 2)
        elif is_piece(cnt) == (255, 255, 255):
            center = cv2.moments(cnt)
            if center["m00"] != 0:
                x = center["m10"] / center["m00"]
                y = center["m01"] / center["m00"]
                # [x, y] = np.dot(M, [x, y, 1])
                center = np.array([x, y])

                area = cv2.contourArea(cnt)

                p = cv2.perspectiveTransform(np.array([[[x, y]]]), sm.N)[0][0]
                board_size = sm.board.shape[0]
                p = (p  * 206 / board_size - 56) / 32
                black_pieces.append({'frame': center, 'board': p})
                text = f"({p[0]:.3f}, {p[1]:.3f}) A={area:.3f}"
                position = (round(center[0] + 10), round(center[1] + 20))  # 文本位置
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
#                     color = is_circle(cnt)
                color = (255, 255, 255)  # 白色
                thickness = 1
                line_type = cv2.LINE_AA

                if 0 < p[0] < 3 and 0 < p[1] < 3:
                    cv2.circle(marker_frame, center.astype(np.int32), 5, (0, 255, 0), -1)
                else:
                    cv2.circle(marker_frame, center.astype(np.int32), 5, (0, 0, 255), -1)

                cv2.putText(marker_frame, text, position, font, font_scale, color, thickness, line_type)

                cv2.drawContours(marker_frame, cnt, -1, (0, 0, 255), 2)

    cv2.imshow(f'Camera View Black Pieces Mask {black}', capture)
    cv2.imshow(f'Camera View Black Pieces Status {black}', marker_frame)

    return black_pieces
