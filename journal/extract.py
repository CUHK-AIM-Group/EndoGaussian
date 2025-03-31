import cv2
import os

# 读取视频文件
video = cv2.VideoCapture('0215.mp4')

# 创建输出目录
os.makedirs('colmap_0215', exist_ok=True)

# 遍历视频帧并保存为图片
frame_count = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    cv2.imwrite(f'colmap_0215/output_{frame_count:04d}.png', frame)
    frame_count += 1

# 释放资源
video.release()
