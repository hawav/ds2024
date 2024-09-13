x, y = 225, 397
board_size = 715
phy_x = 350 * x / board_size
phy_y = 350 - 350 * y / board_size + 28
print(phy_x, phy_y)

# 移动到 (pyh_x, pyh_y, 20) 然后切换到近场摄像头微调定位

# 摄像头定位OK后，phy_y - 19 对齐电磁铁

# 高度降到 6