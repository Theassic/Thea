import cv2
import matplotlib.pyplot as plt
import dlib

# 2 读取一张图片
image = cv2.imread("C:/Users/16376/Desktop/Diploma_Project/Frame/frame5.jpg")

# 3 调用人脸检测器
detector = dlib.get_frontal_face_detector()

# 4 加载预测关键点模型（68个关键点）
# 人脸关键点模型，下载地址：
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor = dlib.shape_predictor("C:/Users/16376/Desktop/Diploma_Project/shape_predictor_68_face_landmarks.dat")

# 5 灰度转换
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 6 人脸检测
faces = detector(gray, 1)

# 7 循环，遍历每一张人脸，给人脸绘制矩形框和关键点
# for face in faces:  # (x, y, w, h)
face = faces[0]

# 8 绘制矩形框
# cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)

# 9 预测关键点
shape = predictor(image, face)

# 10 获取到关键点坐标
for pt in shape.parts():
    # 获取横纵坐标
    pt_position = (pt.x, pt.y)
    # 11 绘制关键点坐标
    cv2.circle(image, pt_position, 3, (255, 0, 0), -1)  # -1填充，2表示大小

roi_a = [shape.part(36).x, shape.part(28).y, shape.part(39).x, shape.part(33).y]
roi_b = [shape.part(42).x, shape.part(28).y, shape.part(45).x, shape.part(33).y]
roi_c = [shape.part(19).x, face.top(), shape.part(24).x, max(shape.part(19).y, shape.part(24).y)]

cv2.rectangle(image, (roi_a[0], roi_a[1]), (roi_a[2], roi_a[3]), (0, 255, 0), 2)
cv2.rectangle(image, (roi_b[0], roi_b[1]), (roi_b[2], roi_b[3]), (0, 255, 0), 2)
cv2.rectangle(image, (roi_c[0], roi_c[1]), (roi_c[2], roi_c[3]), (0, 255, 0), 2)

# 12 显示整个效果图
# plt.imshow(image)
# plt.axis("off")
# plt.show()
cv2.namedWindow('test', 0)
cv2.resizeWindow('test', 360, 640)
cv2.imshow('test', image)
cv2.waitKey(0)
cv2.imencode('.jpg', image)[1].tofile("C:/Users/16376/Desktop/Diploma_Project/facedetectpici.jpg")