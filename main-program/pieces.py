import cv2
import numpy as np

def is_piece(contour):
    # 计算轮廓的面积
    area = cv2.contourArea(contour)
    if 600 < area < 10000:
        # 计算轮廓的周长
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter == 0:
            return (0, 0, 255)
        
        # 计算圆形度
        circularity = 4 * np.pi * (area / (perimeter ** 2))
        
        # 使用轮廓近似来减少轮廓上的点数
        epsilon = 0.03 * perimeter  # 调整这个值以控制近似程度
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
            if 0.5 < aspect_ratio < 1.5:
                if 0.2 < circularity < 2.0:
                    return (255, 255, 255)
                return (255, 255, 0)
            return (0, 255, 255)
    
        return (255, 0, 0)
    return (0, 0, 0)