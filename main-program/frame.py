import cv2
import numpy as np
import util

rect = None

maxDiagonal = None

def calc_frame(img, sm):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edged = cv2.Canny(gray, 25, 300)
    
    cv2.imshow('Edged', edged)

    # kernel_size = 10
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # dilated_edges = cv2.dilate(edged, kernel, iterations=1)

    # # cv2.imshow('dilated_edges', dilated_edges)

    # edged = dilated_edges
    
    marker_frame = img.copy()
    
    # 预处理
#         _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 旋转图像
    # rotated = cv2.warpAffine(frame, M, (width, height))
    
    global maxDiagonal
    if sm.should_update():
        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(is_square, contours)
        # contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        global rect
        if len(contours) > 0:
            max_change_rate = 1  # 设定最大变化率

            now = cv2.approxPolyDP(contours[0], 0.05 * cv2.arcLength(contours[0], True), True)
            def fix_rotation(rect):
                norms = np.linalg.norm(rect.reshape((4, 2)), axis=1)
                pos = np.argmin(norms)
                # (x, y) = rect[1][0]- rect[0][0]
                # a = np.arctan2(y, x)
                # print(a)
                if pos != 0:
                    rect = np.roll(rect, -pos, axis=0)
                return rect
            now = fix_rotation(now)
            # 应用平滑处理
            if rect is not None:
                rect = apply_smoothing(now, rect, max_change_rate)
            else:
                rect = now

            cv2.drawContours(marker_frame, [rect], 0, (0, 255, 0), 2)
        # for cnt in contours:
        #     # approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

        #     if is_square(cnt):
        #         cv2.drawContours(marker_frame, [cnt], 0, (0, 255, 0), 2)
        cv2.imshow('CheckBoard Status', marker_frame)
        
        if rect is None:
            return None

        # 计算四个对角线的长度
        diagonal1 = ((rect[0][0][0] - rect[1][0][0]) ** 2) + ((rect[0][0][1] - rect[1][0][1]) ** 2)
        diagonal2 = ((rect[2][0][0] - rect[3][0][0]) ** 2) + ((rect[2][0][1] - rect[3][0][1]) ** 2)
        diagonal3 = ((rect[0][0][0] - rect[3][0][0]) ** 2) + ((rect[0][0][1] - rect[3][0][1]) ** 2)
        diagonal4 = ((rect[1][0][0] - rect[2][0][0]) ** 2) + ((rect[1][0][1] - rect[2][0][1]) ** 2)

        # 找到最大的对角线长度作为正方形的边长
        maxDiagonal = int(np.sqrt(max(diagonal1, diagonal2, diagonal3, diagonal4)))

        # 定义目标矩形的四个顶点
        dstQuad = np.float32([[0, 0], [0, maxDiagonal - 1], [maxDiagonal - 1, maxDiagonal - 1], [maxDiagonal - 1, 0]])

        # 计算透视变换矩阵
        sm.M = cv2.getPerspectiveTransform(rect.astype(np.float32), dstQuad)
    
    if not maxDiagonal:
        return None

    # 应用透视变换
    img = cv2.warpPerspective(img, sm.M, (maxDiagonal, maxDiagonal))

    return img

def is_square(contour):
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    if area < 100000:
        return False
        
    # 计算轮廓的周长
    perimeter = cv2.arcLength(contour, True)
    
    if perimeter == 0:
        return False
    
    # 使用轮廓近似来减少轮廓上的点数
    epsilon = 0.05 * perimeter  # 调整这个值以控制近似程度
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 计算近似轮廓的顶点数
    num_vertices = len(approx)
    
    # 检查近似轮廓是否接近圆形
    if num_vertices == 4:  # 如果顶点数4可能是矩形或正方形
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
        
        # 如果长宽比接近 1，且圆形度足够接近 1，则认为是方形
        if 0.8 < aspect_ratio < 1.2:
            return True
    
    return False

def apply_smoothing(new_rect, prev_rect, max_change_rate):
    """
    Apply smoothing to the new rectangle coordinates based on the previous rectangle and a maximum change rate.
    
    Parameters:
    - new_rect: New rectangle coordinates as a 3D NumPy array with shape (4, 1, 2).
    - prev_rect: Previous rectangle coordinates as a 3D NumPy array with shape (4, 1, 2).
    - max_change_rate: Maximum allowed change rate for each coordinate.
    
    Returns:
    - Smoothed rectangle coordinates as a 3D NumPy array.
    """
    # Flatten the arrays for easier manipulation
    new_rect_flat = new_rect.reshape(-1, 2)
    prev_rect_flat = prev_rect.reshape(-1, 2)
    
    # Calculate the differences between the new and previous rectangles
    diff = new_rect_flat - prev_rect_flat
    
    # Apply the maximum change rate
    diff_clipped = np.clip(diff, -max_change_rate, max_change_rate)
    
    # Update the new rectangle with the clipped differences
    smoothed_rect = prev_rect_flat + diff_clipped
    
    # Return the smoothed rectangle in its original shape
    return smoothed_rect.reshape(new_rect.shape)
