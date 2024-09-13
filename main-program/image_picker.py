import cv2
import numpy as np

cv2.namedWindow('Cam')

def nothing(x):
    pass

# 创建滑动条
cv2.createTrackbar('H Min', 'Cam', 0, 179, nothing)
cv2.createTrackbar('H Max', 'Cam', 179, 179, nothing)
cv2.createTrackbar('S Min', 'Cam', 0, 255, nothing)
cv2.createTrackbar('S Max', 'Cam', 255, 255, nothing)
cv2.createTrackbar('V Min', 'Cam', 0, 255, nothing)
cv2.createTrackbar('V Max', 'Cam', 255, 255, nothing)

cv2.setTrackbarPos('H Min', 'Cam', 30)
cv2.setTrackbarPos('H Max', 'Cam', 160)
cv2.setTrackbarPos('S Min', 'Cam', 0)
cv2.setTrackbarPos('S Max', 'Cam', 255)
cv2.setTrackbarPos('V Min', 'Cam', 0)
cv2.setTrackbarPos('V Max', 'Cam', 255)

def handle_img(img):
    cv2.imshow('img', img)

    while True:

        h_min = cv2.getTrackbarPos('H Min', 'Cam')
        h_max = cv2.getTrackbarPos('H Max', 'Cam')
        s_min = cv2.getTrackbarPos('S Min', 'Cam')
        s_max = cv2.getTrackbarPos('S Max', 'Cam')
        v_min = cv2.getTrackbarPos('V Min', 'Cam')
        v_max = cv2.getTrackbarPos('V Max', 'Cam')

        lower_green = np.array([h_min, s_min, v_min])
        upper_green = np.array([h_max, s_max, v_max])

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 创建掩模
        mask = cv2.inRange(hsv, lower_green, upper_green)

        cv2.imshow('mask', mask)

    # contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

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

# stream_url = "http://192.168.1.119/webcam2/?action=stream"
# open_near_cam(stream_url)

handle_img(cv2.imread('img3.png'))