import cv2
import numpy as np

# cv2.namedWindow('Human Detection Thresholding')

# def nothing(x):
#     pass

# # 创建滑动条
# cv2.createTrackbar('H Min', 'Human Detection Thresholding', 0, 179, nothing)
# cv2.createTrackbar('H Max', 'Human Detection Thresholding', 179, 179, nothing)
# cv2.createTrackbar('S Min', 'Human Detection Thresholding', 0, 255, nothing)
# cv2.createTrackbar('S Max', 'Human Detection Thresholding', 255, 255, nothing)
# cv2.createTrackbar('V Min', 'Human Detection Thresholding', 0, 255, nothing)
# cv2.createTrackbar('V Max', 'Human Detection Thresholding', 255, 255, nothing)

# cv2.setTrackbarPos('S Max', 'Human Detection Thresholding', 30)
# cv2.setTrackbarPos('V Min', 'Human Detection Thresholding', 100)

# cold_down = 3

def detect_human(frame):
#     h_min = cv2.getTrackbarPos('H Min', 'Human Detection Thresholding')
#     h_max = cv2.getTrackbarPos('H Max', 'Human Detection Thresholding')
#     s_min = cv2.getTrackbarPos('S Min', 'Human Detection Thresholding')
#     s_max = cv2.getTrackbarPos('S Max', 'Human Detection Thresholding')
#     v_min = cv2.getTrackbarPos('V Min', 'Human Detection Thresholding')
#     v_max = cv2.getTrackbarPos('V Max', 'Human Detection Thresholding')

    # 定义HSV阈值范围
    # lower_green = np.array([h_min, s_min, v_min])
    # upper_green = np.array([h_max, s_max, v_max])

    lower_green = np.array([0, 50, 116])
    upper_green = np.array([21, 130, 250])

    # 转换到HSV颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建掩模
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # cv2.imshow('Human MASK', mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    global cold_down
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cold_down = 3
            return True
        
    # if cold_down > 0:
    #     cold_down -= 1
    #     return True

    return False