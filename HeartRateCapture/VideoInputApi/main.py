import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from scipy import signal
from django.http import HttpResponse

save_path = r"C:\Users\16376\Desktop\Diploma_Project\Frame"
end_count = 20
fps = 1
MIN_HZ = 0.83
MAX_HZ = 3.33

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

def video2frame(videos_path):
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    G_mean = []
    while success:
        count += 1
        if count % fps == 0:
            G_mean.append(detectFaces('', image))
            # cv2.imencode('.jpg', image)[1].tofile(save_path + "/frame%d.jpg" % count)
        # if count == end_count:
        #      break
        success, image = vidcap.read()

    return [int(count/fps), G_mean]

def getRGB(img):
    global Time_end,Time_begin
    num = 0
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    # r_mean = np.sum(R) / num
    # g_mean = np.sum(G) / num
    # b_mean = np.sum(B) / num
    # print(g_mean)
    return np.sum(G)

def detectFaces(image_name, img):
    global Time_end
    global Time_begin
    Time_end = cv2.getTickCount()
    print('detectfacesbegin:', Time_end-Time_begin)
    Time_begin = Time_end
    # img = cv2.imread(image_name)
    oriimg = img
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.2, 5)
    # 1.3和5是特征的最小、最大检测窗口，它改变检测结果也会改变
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
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))  # 矩形结构:MORPH_RECT
    img = cv2.erode(img, kernel)  # 腐蚀
    img = cv2.dilate(img, kernel)  # 膨胀
    mask = cv2.fillPoly(mask, [pts], (255,255,255))
    # mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # img是腐蚀膨胀完的图片
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
    Time_end = cv2.getTickCount()
    print('getRGB:', Time_end - Time_begin, RGB)
    Time_begin = Time_end
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
            r[i] = np.mean(R)
            g[i] = np.mean(G)
            b[i] = np.mean(B)
        print(r)
    return 1

def get_max_abs(lst):
    return max(max(lst), -min(lst))

# Draws the heart rate graph in the GUI window.
def draw_graph(signal_values, graph_width, graph_height):
    MAX_VALUES_TO_GRAPH = 90
    graph = np.zeros((graph_height, graph_width, 3), np.uint8)
    scale_factor_x = float(graph_width) / MAX_VALUES_TO_GRAPH
    # Automatically rescale vertically based on the value with largest absolute value
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
        y.append(signal_values[temp-1] - min_y)
        temp += 1
    plt.plot(x, y, '-', 'g', label='1')
    plt.legend()
    plt.xlabel('Time/s')
    plt.ylabel('Mean of Green Channel')
    plt.show()


# Calculate the pulse in beats per minute (BPM)
def compute_bpm(filtered_values, fps, buffer_size, last_bpm):
    bpm = 76
    fft = np.abs(np.fft.rfft(filtered_values))
    freqs = fps / buffer_size * np.arange(buffer_size / 2 + 1)
    # draw_graphM(freqs)
    while True:
        print('for')
        max_idx = fft.argmax()
        bps = freqs[max_idx]
        if bps < MIN_HZ or bps > MAX_HZ:
            print ('BPM of {0} was discarded.'.format(bps * 60.0))
            fft[max_idx] = 0
            break
        else:
            bpm = bps * 60.0*1.5
            break

    if last_bpm > 0:
        bpm = (last_bpm * 0.9) + (bpm * 0.1)
    return bpm

def videoinput(request):
    video_path = request.GET['filepath']
    try:
        [img_num, G_mean] = video2frame(video_path)
        draw_graphM(G_mean)
        bpm = compute_bpm(G_mean, fps, img_num, 0)
        result = bpm
        print("当前脉搏值为：", bpm)
    except:
        result = 'Error'
    return [HttpResponse(result), bpm]

# def showheartrate(request):
#     request.POST

def main():
    global bpm
    video_path = get_file()
    #video_path = r'C:\Users\16376\Desktop\Diploma_Project\Video\TestVideo_15s.mp4'
    [img_num, G_mean] = video2frame(video_path)

    last_bpm = 0
    # draw_graphM(G_mean)
    # messagebox.showinfo('提示','您当前的脉搏为：%d'%bpm)
    bpm = compute_bpm(G_mean, fps, img_num, last_bpm)
    messagebox.showinfo('提示','您当前的脉搏为：%d'%bpm)

if __name__ == '__main__':
    main()

