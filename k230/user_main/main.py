import time, math, os, gc
from media.sensor import *
from media.display import *
from media.media import *
from libs.PipeLine import PipeLine
import cv2 as cv
from ulab import numpy as np
from ybUtils.YbUart import YbUart
import sys
APP_DIR = "/sdcard/user_main"
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
sys.path.append("/sdcard/user_main")
from yolo_benchmark import YoloTopKBenchmark
from wifi_transmit import wifi_transmit
from user_uart import uart_send
MODEL_PATH = "/data/yolov8n_roi_320_30_topk.kmodel"
AI_SIZE = [640, 480]
MODEL_SIZE = [320, 32]
TOPK_CANDIDATES = 16
DETECT_AREA = [0, 210, 640, 60]
sensor = Sensor(width=1280, height=720, fps=120)
pl = None
rtsp = None
detector = None
center_dot = (DETECT_AREA[0]+DETECT_AREA[2]//2,DETECT_AREA[1]+DETECT_AREA[3]//2)
k230_uart = YbUart(baudrate=115200)
fps = time.clock()
fps_value = 0.0
try:
    pl = PipeLine(rgb888p_size=AI_SIZE,display_mode="lcd",display_size=[640, 480])
    pl.create(sensor=sensor,dont_init=True,ch1_frame_size=[640,480])
    pl.sensor.set_pixformat(PIXEL_FORMAT_YUV_SEMIPLANAR_420,chn=CAM_CHN_ID_1)
    detector = YoloTopKBenchmark(
    MODEL_PATH,
    MODEL_SIZE,
    AI_SIZE,
    TOPK_CANDIDATES,
    crop_region=DETECT_AREA,
    confidence_threshold=0.6,
    nms_threshold=0.45,
    max_detections=1,
    debug_mode=0,
)
    detector.config_preprocess()
    rtsp = wifi_transmit(pl.sensor,CAM_CHN_ID_1)
    while True:
        fps.tick()
        img = pl.get_frame()
        detections = detector.run(img)
        pl.osd_img.clear()
        pl.osd_img.draw_string_advanced(
            8,
            8,
            24,
            "balls=%d  fps=%.1f" % (0, fps_value),
            color=(255, 255, 255, 0),
        )
        if detections:
            class_id, score, x1, y1, x2, y2 = detections[0]
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            width = int(x2 - x1)
            height = int(y2 - y1)
            detector.draw_result(pl, [detections[0]], fps_value)
            pl.osd_img.draw_line(center_dot[0],center_dot[1],center_x,center_y,thickness=2,color=(255, 0, 0))
            pl.osd_img.draw_circle(center_x,center_y,4,thickness=2,fill=True,color=(255,0,0))
            uart_send(k230_uart,center_x)
        pl.osd_img.draw_rectangle(DETECT_AREA[0],DETECT_AREA[1],DETECT_AREA[2],DETECT_AREA[3],thickness=4,color=(0,0,255))
        pl.osd_img.draw_circle(center_dot[0],center_dot[1],4,thickness=2,fill=True,color=(173, 216, 230))
        fps_value = fps.fps()
        pl.show_image()
finally:
    if rtsp is not None:
        rtsp.stop()
    if detector is not None:
        detector.deinit()
    if pl is not None:
        pl.destroy()