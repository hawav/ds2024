import cv2
import numpy as np
from pieces import is_piece
from printer import send_gcode_script

cv2.namedWindow('Cam')

def nothing(x):
    pass

def is_piece(contour):
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    if area > 10000:
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return (0, 0, 255)
        
        # 计算圆形度
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # 使用轮廓近似来减少轮廓上的点数
        epsilon = 0.002 * perimeter  # 调整这个值以控制近似程度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 计算近似轮廓的顶点数
        num_vertices = len(approx)
        
        # 检查近似轮廓是否接近圆形
        if num_vertices > 5:  # 如果顶点数大于6，可能接近圆形
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
            if 0.9 < aspect_ratio < 1.1:
                if 0 < circularity < 2.0:
                    return (255, 255, 255)
                print(circularity)
                return (255, 255, 0)
            return (0, 255, 255)
    
        return (255, 0, 0)
    return (0, 0, 0)

# 创建滑动条
cv2.createTrackbar('H Min', 'Cam', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Cam', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Cam', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Cam', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Cam', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Cam', 255, 255, nothing)

cv2.setTrackbarPos('H Min', 'Cam', 50)
cv2.setTrackbarPos('H Max', 'Cam', 140)
cv2.setTrackbarPos('S Min', 'Cam', 0)
cv2.setTrackbarPos('S Max', 'Cam', 255)
cv2.setTrackbarPos('V Min', 'Cam', 0)
cv2.setTrackbarPos('V Max', 'Cam', 255)

prev_piece_center = None

class Status:
    Idle = 0,
    WaitingMovement = 1,
    WaitingStable = 2,
    MovementCompleted = 3,

class StateMachine:
    def __init__(self) -> None:
        self.status = Status.Idle

    def update(self, is_moving):
        if self.status == Status.WaitingMovement and is_moving:
            self.status = Status.WaitingStable
        elif self.status == Status.WaitingStable and not is_moving:
            self.status = Status.MovementCompleted

sm = StateMachine()

cold_down = 10

def handle_img(img):
    cv2.imshow('img', img)

    # lower_green = np.array([50, 0, 0])
    # upper_green = np.array([140, 255, 255])

    h_min = cv2.getTrackbarPos('H Min', 'Cam')
    h_max = cv2.getTrackbarPos('H Max', 'Cam')
    s_min = cv2.getTrackbarPos('S Min', 'Cam')
    s_max = cv2.getTrackbarPos('S Max', 'Cam')
    v_min = cv2.getTrackbarPos('V Min', 'Cam')
    v_max = cv2.getTrackbarPos('V Max', 'Cam')

    lower_green = np.array([h_min, s_min, v_min])
    upper_green = np.array([h_max, s_max, v_max])

    # 高斯模糊的核大小，必须是奇数
    kernel_size = (3, 3)

    # 应用高斯模糊
    blurred_image = cv2.GaussianBlur(img, kernel_size, 0)

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # 创建掩模
    mask = cv2.inRange(hsv, lower_green, upper_green)

    cv2.imshow('mask', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    capture = blurred_image.copy()

    piece_center = None

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)  # 调整这个值以控制近似程度
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(capture, [approx], -1, is_piece(contour), 2)
        center = cv2.moments(contour)
        if is_piece(contour) == (255, 255, 255) and center["m00"] != 0:
            x = int(center["m10"] / center["m00"])
            y = int(center["m01"] / center["m00"])
            piece_center = np.array([x, y])
            break

    center = np.array((capture.shape[1], capture.shape[0])) / 2

    cv2.circle(capture, np.round(center).astype(np.int32), 5, (255, 255, 255), -1)

    if piece_center is not None:
        global prev_piece_center
        if prev_piece_center is None:
            prev_piece_center = piece_center
        
        is_moving = np.linalg.norm(piece_center - prev_piece_center) > 4

        if is_moving:
            prev_piece_center = piece_center
            print('运动！')
        else:
            print('恒定')

        sm.update(is_moving)

        if sm.status == Status.MovementCompleted:
            print('OK，一次移动完成')
            sm.status = Status.Idle

        if sm.status == Status.Idle:
            cv2.circle(capture, piece_center.astype(np.int32), 5, (255, 255, 255), -1)
            err = (piece_center - center) * (1, -1)
            print('误差', err)
            if np.linalg.norm(err) > 4:
                global cold_down
                if cold_down == 0:
                    cold_down = 5
                    fixup = err * 0.02
                    # fixup = (err / (56 / 3, 18.2)) * 0.8
                    print('修正量', fixup)
                    sm.status = Status.WaitingMovement
                    send_gcode_script(f'G0 X{fixup[0]} Y{fixup[1]}')
                else:
                    cold_down -= 1
            else:
                print('误差良好')
                return

    cv2.imshow('capture', capture)
#     for cnt in contours:
        

#         if DEBUG:
#             center = cv2.moments(cnt)
#             if center["m00"] != 0:
#                 x = int(center["m10"] / center["m00"])
#                 y = int(center["m01"] / center["m00"])
#                 # [x, y] = np.dot(M, [x, y, 1])
#                 center = (round(x), round(y))

#                 color = is_piece(cnt)
#                 if color != (0, 0, 0):
#                     area = cv2.contourArea(cnt)

#                     text = f"({x:.3f}, {y:.3f}) A={area:.3f}"
#                     position = (center[0] + 10, center[1] + 20)  # 文本位置
#                     font = cv2.FONT_HERSHEY_SIMPLEX
#                     font_scale = 0.5
#                     thickness = 1
#                     line_type = cv2.LINE_AA

#                     cv2.putText(marker_frame, text, position, font, font_scale, color, thickness, line_type)
#                     cv2.drawContours(marker_frame, cnt, -1, (0, 0, 255), 2)
#         elif is_piece(cnt) == (255, 255, 255):
#             center = cv2.moments(cnt)
#             if center["m00"] != 0:
#                 x = center["m10"] / center["m00"]
#                 y = center["m01"] / center["m00"]
#                 # [x, y] = np.dot(M, [x, y, 1])
#                 center = np.array([x, y])

#                 area = cv2.contourArea(cnt)

#                 p = cv2.perspectiveTransform(np.array([[[x, y]]]), sm.N)[0][0]
#                 board_size = sm.board.shape[0]
#                 p = (p  * 206 / board_size - 56) / 32
#                 black_pieces.append({'frame': center, 'board': p})
#                 text = f"({p[0]:.3f}, {p[1]:.3f}) A={area:.3f}"
#                 position = (round(center[0] + 10), round(center[1] + 20))  # 文本位置
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 0.5
# #                     color = is_circle(cnt)
#                 color = (255, 255, 255)  # 白色
#                 thickness = 1
#                 line_type = cv2.LINE_AA

#                 if 0 < p[0] < 3 and 0 < p[1] < 3:
#                     cv2.circle(marker_frame, center.astype(np.int32), 5, (0, 255, 0), -1)
#                 else:
#                     cv2.circle(marker_frame, center.astype(np.int32), 5, (0, 0, 255), -1)

#                 cv2.putText(marker_frame, text, position, font, font_scale, color, thickness, line_type)

#                 cv2.drawContours(marker_frame, cnt, -1, (0, 0, 255), 2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def open_near_cam(stream_url):
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    while True:
        ret, img = cap.read()

        handle_img(img)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

stream_url = "http://192.168.1.119/webcam2/?action=stream"
send_gcode_script('G91')
open_near_cam(stream_url)

# handle_img(cv2.imread('C:/Users/hawav/w/ds2024/main-program/img6.png'))
