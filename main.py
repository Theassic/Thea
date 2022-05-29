import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from scipy import signal

save_path = r"C:\Users\16376\Desktop\Diploma_Project\Frame"
end_count = 20
fps = 3
MIN_HZ = 0.5
MAX_HZ = 3.33

detector = dlib.get_frontal_face_detector()         # 调用人脸检测器
# 加载预测关键点模型（68个关键点）
predictor = dlib.shape_predictor("C:/Users/16376/Desktop/Diploma_Project/shape_predictor_68_face_landmarks.dat")

Time_begin = 0
Time_end = 0
bpm = 68

def get_file():
    root = tk.Tk()
    root.withdraw()
    if messagebox.askokcancel('视频脉搏提取系统', '请选择视频文件'):
        file_path = filedialog.askopenfilename()
        print(file_path)
        return file_path
    else:
        return -1

# main.py
def video2frame(videos_path):
    vidcap = cv2.VideoCapture(videos_path)     # 按帧读取视频文件
    success, image = vidcap.read()
    count = 0
    G_mean = []
    while success:
        count += 1
        if count % fps == 0:
            # G_mean.append(detectFaces('', image))
            G_mean.append(dlibDetect(image, count))
            # cv2.imencode('.jpg', image)[1].tofile(save_path + "/frame%d.jpg" % count)
        # if count == end_count:
        #      break
        success, image = vidcap.read()

    return [int(count/fps), G_mean]

def getRGB(img, roi):
    num = 0
    if roi == '':
        # B = img[:, :, 0]
        G = img[:, :, 1]
        # R = img[:, :, 2]
    else:
        # B = img[roi[1]:roi[3], roi[0]:roi[2], 0]
        G = img[roi[1]:roi[3], roi[0]:roi[2], 1]
        # R = img[roi[1]:roi[3], roi[0]:roi[2], 2]
    Time_end = cv2.getTickCount()
    # print(roi)
    # cv2.namedWindow('test', 0)
    # cv2.resizeWindow('test', 360, 640)
    # cv2.imshow('test', G)
    # cv2.waitKey(0)

    # fp = [1.0, 2.0];        # 通带
    # fst = [0.5, 3.33];      # 阻带
    # wp = 2 * fp / fs;     # 设置通带频率
    # ws = 2 * fst / fs;    # 设置阻带频率
    # ap = 3; # 设置通带波纹系数
    # as = 10; # 设置阻带波纹系数
    # N = math.log(np.sqrt(10**(as/10)-1),10)/math.log(ls,10) # 计算滤波器阶数
    # [b0, a0] = signal.butter(N, [wp, ws], btype='bandpass');    #计算巴特沃夫滤波器参数
    # [w, h] = signal.freqz(b0, a0);

    # print('GetRGB:', np.sum(G))

    # r_mean = np.sum(R) / num
    # g_mean = np.sum(G) / num
    # b_mean = np.sum(B) / num
    # print(g_mean)
    return np.sum(G)

def dlibDetect(image, num):
    global predictor
    global detector

    img = image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      # 灰度转换
    faces = detector(gray, 1)       # 人脸检测
    face = faces[0]

    if num == 1:
        cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        cv2.imencode('.jpg', img)[1].tofile(save_path + "showbg.jpg")

    shape = predictor(image, face)
    # for pt in shape.parts():      # 获取关键点坐标
    #     pt_position = (pt.x, pt.y)
    #     cv2.circle(image, pt_position, 3, (255, 0, 0), -1)  # 绘制关键点坐标

    roi_a = [shape.part(36).x, shape.part(28).y, shape.part(39).x, shape.part(33).y]
    roi_b = [shape.part(42).x, shape.part(28).y, shape.part(45).x, shape.part(33).y]
    roi_c = [shape.part(19).x, face.top(), shape.part(24).x, max(shape.part(19).y, shape.part(24).y)]

    # cv2.rectangle(image, (roi_a[0], roi_a[1]), (roi_a[2], roi_a[3]), (0, 255, 0), 2)
    # cv2.rectangle(image, (roi_b[0], roi_b[1]), (roi_b[2], roi_b[3]), (0, 255, 0), 2)
    # cv2.rectangle(image, (roi_c[0], roi_c[1]), (roi_c[2], roi_c[3]), (0, 255, 0), 2)

    area_a = (roi_a[2] - roi_a[0]) * (roi_a[3] - roi_a[1])
    area_b = (roi_b[2] - roi_b[0]) * (roi_b[3] - roi_b[1])
    area_c = (roi_c[2] - roi_c[0]) * (roi_c[3] - roi_c[1])

    # cv2.namedWindow('test', 0)
    # cv2.resizeWindow('test', 360, 640)
    # cv2.imshow('test', image)
    # cv2.waitKey(0)

    # 计算当前帧图像ROI区域内绿色通道均值
    RGB = (getRGB(image,roi_a) + getRGB(image,roi_b) + getRGB(image,roi_c)) / (area_a + area_b + area_c)
    print('RGB: ', RGB)

    return RGB

def detectFaces(image_name, img):
    global Time_end
    global Time_begin

    # img = cv2.imread(image_name)
    oriimg = img
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.2, 5)

    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
        img_result = cv2.rectangle(img, (x,y), (x+width, y+height), (255,0,0), 2)
        pts = np.array([[x + width*0.15, y + height*0.2],
                       [x + width*0.2, y + height*0.8],
                       [x + width*0.5, y + height],
                       [x + width*0.8, y + height*0.8],
                       [x + width*0.85, y + height*0.2]],
                       np.int32)
        pts = pts.reshape(-1,1,2)
        break
    mask = np.zeros(img.shape, np.uint8)
    mask = cv2.polylines(mask, [pts], True, (0,255,255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    img = cv2.erode(img, kernel)  # 腐蚀
    img = cv2.dilate(img, kernel)  # 膨胀
    mask = cv2.fillPoly(mask, [pts], (255,255,255))
    # mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ROI = cv2.bitwise_and(mask, oriimg)
    Time_end = cv2.getTickCount()
    print('detectfacesend:', Time_end - Time_begin)
    Time_begin = Time_end

    # cv2.namedWindow('test', 0)
    # cv2.resizeWindow('test', 360, 640)

    # cv2.imshow('test', ROI)
    # cv2.imshow('test', img_result)
    # cv2.waitKey(0)

    RGB = getRGB(ROI)/(width*height)
    return RGB

def getRGB_list(frame_path, img_num, face_point):
    # img_list = dir([frame_path, '*.jpg'])
    # img_num = len(img_list)
    if img_num > 0:
        for i in range(img_num):
            order_num = i
            total_path = (frame_path + '%d.jpg' % order_num*5)
            image = cv2.imread(total_path)
            Y = image[face_point[1]:face_point[3], face_point[2]:face_point[4]]
            R = Y[:,:,1]
            G = Y[:,:,2]
            B = Y[:,:,3]
            # r[i] = np.mean(R)
            # g[i] = np.mean(G)
            # b[i] = np.mean(B)
        # print(r)
    return 1

def get_max_abs(lst):
    return max(max(lst), -min(lst))

def draw_graph(signal_values, graph_width, graph_height):
    MAX_VALUES_TO_GRAPH = 200
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH
    max_abs = get_max_abs(signal_values)
    print(max_abs)
    scale_factor_y = (float(graph_height) / 2.0) / max_abs
    midpoint_y = graph_height / 2
    print(midpoint_y)
    for i in range(0, len(signal_values) - 1):
        curr_x = int(i * scale_factor_x)
        curr_y = int(midpoint_y + signal_values[i] * scale_factor_y)
        next_x = int((i + 1) * scale_factor_x)
        next_y = int(midpoint_y + signal_values[i + 1] * scale_factor_y)
        cv2.line(graph, (curr_x, curr_y), (next_x, next_y), color=(0, 255, 0), thickness=1)
    cv2.namedWindow('GREEN_MEAN', 0)
    cv2.resizeWindow('GREEN_MEAN', graph_width, graph_height)
    cv2.imshow('GREEN_MEAN', graph)
    cv2.waitKey(0)
    return graph

def draw_graphM(signal_values):
    temp = 1
    MAX_t = len(signal_values)
    x = []
    y = []
    min_y = min(signal_values)
    while temp <= MAX_t:
        x.append(temp*15/MAX_t)
        y.append(signal_values[temp-1])
        temp += 1
    plt.plot(x, y, '-', 'g')
    plt.legend(loc='best', shadow=False, scatterpoints=1, labels=['1'])
    plt.xlabel('Time/s')
    plt.ylabel('Mean of Green Channel')
    plt.show()


# Calculate the pulse in beats per minute (BPM)
def compute_bpm(values, fps, buffer_size, last_bpm):
    print('compute')
    fft = np.abs(np.fft.rfft(values))   # 快速傅里叶变换
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)
    # draw_graphM(freqs)
    while True:
        print('for')
        max_x = fft.argmax()        # 峰值检测
        bps = freqs[max_x]
        if bps < MIN_HZ or bps > MAX_HZ:
            fft[max_x] = 0
        else:
            bpm = bps * 60.0
            break

    if last_bpm > 0:
        bpm = (last_bpm * 0.9) + (bpm * 0.1)
    return bpm

def main():
    global bpm
    video_path = get_file()
    #video_path = r'C:\Users\16376\Desktop\Diploma_Project\Video\TestVideo_15s.mp4'
    [img_num, G_mean] = video2frame(video_path)

    last_bpm = 0
    # draw_graphM(G_mean)

    bpm = compute_bpm(G_mean, fps, img_num, last_bpm)
    messagebox.showinfo('提示', '您当前的脉搏为：%d' % bpm)

if __name__ == '__main__':
    main()

