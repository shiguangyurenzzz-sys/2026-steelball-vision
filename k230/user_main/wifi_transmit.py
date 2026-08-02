import network  # 网络模块，用于处理 WiFi 连接 / Network module for handling WiFi connections
import os      # 操作系统接口模块 / Operating system interface module
import time    # 时间模块，用于延时操作 / Time module for delay operations
import _thread # 线程模块，用于多线程操作 / Thread module for multithreading operations
import sys     # 系统模块，用于系统相关操作 / System module for system-related operations
import random  # 随机数模块 / Random number module
import ujson   # JSON 处理模块 / JSON processing module
import ulab.numpy as np  # 数值计算库 / Numerical computation library
import nncase_runtime as nn  # 神经网络运行时库 / Neural network runtime library
import aidemo  # AI 演示模块 / AI demo module
import image   # 图像处理模块 / Image processing module
import multimedia as mm  # 多媒体模块 / Multimedia module
from time import sleep  # 从 time 模块导入 sleep 函数 / Import sleep function from time module
from media.vencoder import *  # 从媒体模块导入视频编码器相关功能 / Import video encoder-related functions from media module
from media.sensor import *    # 从媒体模块导入传感器相关功能 / Import sensor-related functions from media module
from media.media import *     # 从媒体模块导入媒体管理功能 / Import media management functions from media module
from media.display import *   # 从媒体模块导入显示相关功能 / Import display-related functions from media module
from libs.PipeLine import PipeLine, ScopedTiming  # 从 libs 导入 PipeLine 和 ScopedTiming 类 / Import PipeLine and ScopedTiming classes from libs
from libs.AIBase import AIBase  # 从 libs 导入 AIBase 类 / Import AIBase class from libs
from libs.AI2D import Ai2d      # 从 libs 导入 Ai2d 类 / Import Ai2d class from libs

def load_wifi_credentials():
    """从未纳入 Git 的本地配置读取 Wi-Fi 凭据。"""
    try:
        from wifi_config import WIFI_PASSWORD, WIFI_SSID
    except ImportError:
        raise RuntimeError(
            "缺少 /sdcard/user_main/wifi_config.py；"
            "请复制 wifi_config.py.example 后填写 Wi-Fi 配置"
        )

    if not WIFI_SSID or not WIFI_PASSWORD:
        raise RuntimeError("wifi_config.py 中的 Wi-Fi 配置不能为空")
    return WIFI_SSID, WIFI_PASSWORD


# Connect to WiFi
# 连接到 WiFi 网络
def Connect_WIFI(ID, PASSWORD):
    sta = network.WLAN(0)  # 创建 WLAN 对象，0 表示站模式 / Create WLAN object, 0 indicates station mode
    if sta.isconnected():  # 检查是否已连接 / Check if already connected
        sta.disconnect()   # 如果已连接，则断开连接 / Disconnect if already connected
        time.sleep(1)      # 等待 1 秒 / Wait for 1 second

    sta.connect(ID, PASSWORD)  # 连接到指定的 WiFi 网络 / Connect to the specified WiFi network
    # 查看是否连接成功 / Check if connection is successful
    while sta.ifconfig()[0] == '0.0.0.0':  # 如果 IP 地址为 '0.0.0.0'，表示未连接 / If IP address is '0.0.0.0', it means not connected
        time.sleep(1)                      # 每秒检查一次 / Check every second

    print(sta.ifconfig()[0])  # 打印获取到的 IP 地址 / Print the obtained IP address

    return sta.isconnected()  # 返回连接状态 / Return connection status

# RTSP Server class
# RTSP 服务器类
class RtspServer:
    def __init__(self, session_name="video", port=8554, video_type=mm.multi_media_type.media_h264,
                 enable_audio=False, sensor=None, initMediaManager=None,
                 sensor_chn=CAM_CHN_ID_0, width=640, height=480,
                 bit_rate=600, frame_rate=15):
        self.session_name = session_name  # 会话名称 / Session name
        self.video_type = video_type      # 视频类型：H.264/H.265 / Video type: H.264/H.265
        self.enable_audio = enable_audio  # 是否启用音频 / Whether to enable audio
        self.port = port                  # RTSP 端口号 / RTSP port number
        self.rtspserver = mm.rtsp_server()  # 实例化 RTSP 服务器 / Instantiate RTSP server
        self.venc_chn = VENC_CHN_ID_1     # VENC 通道 / VENC channel
        self.start_stream = False         # 是否启动推流线程 / Whether to start the streaming thread
        self.runthread_over = False       # 推流线程是否已结束 / Whether the streaming thread has finished
        self.sensor = sensor              # 传感器对象 / Sensor object

        self.started = False
        self.owns_sensor = sensor is None
        if initMediaManager is None:
            initMediaManager = self.owns_sensor  # 外部 sensor 通常已由 PipeLine 初始化媒体资源
        self.sensor_chn = sensor_chn
        self.width = ALIGN_UP(width, 16)
        self.height = height
        self.bit_rate = bit_rate
        self.frame_rate = frame_rate

        self.initMediaManager = initMediaManager  # 是否初始化媒体管理器 / Whether to initialize media manager

    # Start the RTSP server
    # 启动 RTSP 服务器
    def start(self):
        if self.started:
            return

        # 初始化推流 / Initialize stream
        self._init_stream()
        self.rtspserver.rtspserver_init(self.port)  # 初始化 RTSP 服务器，指定端口 / Initialize RTSP server with specified port
        # 创建会话 / Create session
        self.rtspserver.rtspserver_createsession(self.session_name, self.video_type, self.enable_audio)
        # 启动 RTSP 服务器 / Start RTSP server
        self.rtspserver.rtspserver_start()
        self._start_stream()  # 启动推流 / Start streaming

        # 启动推流线程 / Start streaming thread
        self.start_stream = True
        self.runthread_over = False
        self.started = True
        _thread.start_new_thread(self._do_rtsp_stream, ())  # 创建新线程运行推流函数 / Create a new thread to run the streaming function

    # Stop the RTSP server
    # 停止 RTSP 服务器
    def stop(self):
        if self.start_stream == False:  # 如果推流未启动，直接返回 / If streaming hasn’t started, return directly
            return
        # 等待推流线程退出 / Wait for the streaming thread to exit
        self.start_stream = False
        while not self.runthread_over:  # 循环等待线程结束 / Loop until the thread ends
            sleep(0.1)                 # 每 0.1 秒检查一次 / Check every 0.1 seconds
        self.runthread_over = False    # 重置线程结束标志 / Reset thread completion flag

        # 停止推流 / Stop streaming
        self._stop_stream()
        self.rtspserver.rtspserver_stop()  # 停止 RTSP 服务器 / Stop RTSP server
        # self.rtspserver.rtspserver_destroysession(self.session_name)  # 销毁会话（已注释） / Destroy session (commented out)
        self.rtspserver.rtspserver_deinit()  # 反初始化 RTSP 服务器 / Deinitialize RTSP server
        self.started = False

    # Get the RTSP URL
    # 获取 RTSP 地址
    def get_rtsp_url(self):
        return self.rtspserver.rtspserver_getrtspurl(self.session_name)  # 返回 RTSP 地址 / Return RTSP URL

    # Initialize the stream
    # 初始化推流
    def _init_stream(self):
        # 设置视频分辨率（以下为可选分辨率，已注释部分为其他选项） / Set video resolution (commented sections are other options)
        # width = 1280
        # height = 720
        width = self.width
        height = self.height
        # width = 1920
        # height = 1080
        # width = 512   # 当前宽度 / Current width
        # height = 288  # 当前高度 / Current height
        # width = 384
        # height = 216

        width = ALIGN_UP(width, 16)  # 将宽度对齐到 16 的倍数 / Align width to a multiple of 16
        if self.owns_sensor:
            # 初始化传感器 / Initialize sensor
            self.sensor = Sensor()       # 创建传感器对象 / Create sensor object
            self.sensor.reset()          # 重置传感器 / Reset sensor
            self.sensor.set_framesize(
                width=width,
                height=height,
                alignment=12,
                chn=self.sensor_chn,
            )  # 设置 RTSP 通道的帧大小 / Set frame size for the RTSP channel
            self.sensor.set_pixformat(
                Sensor.YUV420SP,
                chn=self.sensor_chn,
            )  # RTSP 编码通道必须输出 YUV420SP / RTSP encoding requires YUV420SP
            
        # 实例化视频编码器 / Instantiate video encoder
        self.encoder = Encoder()     # 创建编码器对象 / Create encoder object
        self.encoder.SetOutBufs(self.venc_chn, 8, width, height)  # 当前固件显式使用 VENC 通道
        # 绑定相机和 VENC（已注释，当前未使用） / Bind camera and VENC (commented out, not currently used)
        # self.link = MediaManager.link(self.sensor.bind_info()['src'], (VIDEO_ENCODE_MOD_ID, VENC_DEV_ID, self.venc_chn))

        self.link = None  # 初始化链接为 None / Initialize link as None
        # 初始化媒体管理器 / Initialize media manager
        if self.initMediaManager:
            MediaManager.init()  # 调用媒体管理器的初始化函数 / Call the media manager’s initialization function
        # 创建编码器 / Create encoder
        chnAttr = ChnAttrStr(
            self.encoder.PAYLOAD_TYPE_H264,
            self.encoder.H264_PROFILE_MAIN,
            width,
            height,
            bit_rate=self.bit_rate,
            dst_frame_rate=self.frame_rate,
            src_frame_rate=self.frame_rate,
        )
        
        # 设置编码器属性：H.264 类型，主配置文件，宽度，高度 / Set encoder attributes: H.264 type, main profile, width, height
        self.encoder.Create(self.venc_chn, chnAttr)

    # Start the stream
    # 启动推流
    def _start_stream(self):
        # 开始编码 / Start encoding
        self.encoder.Start(self.venc_chn)
        if self.owns_sensor or self.initMediaManager:
            # 本类初始化 MediaManager 时，必须在线程抓帧前启动 sensor。
            self.sensor.run()

    # Stop the stream
    # 停止推流
    def _stop_stream(self):
        if self.owns_sensor:
            self.sensor.stop()  # 外部 sensor 的生命周期由调用方管理 / External sensor is caller-owned
        # 停止编码 / Stop encoding
        self.encoder.Stop(self.venc_chn)
        self.encoder.Destroy(self.venc_chn)
        if self.owns_sensor and self.initMediaManager:
            # 外部 sensor 和 MediaManager 由调用方（如 PipeLine.destroy）统一释放。
            MediaManager.deinit()

    # RTSP streaming thread
    # RTSP 推流线程
    def _do_rtsp_stream(self):
        try:
            streamData = StreamData()  # 创建流数据对象 / Create stream data object

            while self.start_stream:  # 当推流标志为 True 时循环 / Loop while streaming flag is True
                frame_info = self.sensor.snapshot(
                    chn=self.sensor_chn,
                    timeout=200,
                    dump_frame=True,
                )

                if frame_info is None or frame_info == -1:
                    continue
                if self.encoder.SendFrame(self.venc_chn, frame_info) != 0:
                    continue

                if self.encoder.GetStream(self.venc_chn, streamData, timeout=200) != 0:
                    continue

                try:
                    for pack_idx in range(0, streamData.pack_cnt):
                        self.rtspserver.rtspserver_sendvideodata_byphyaddr(
                            self.session_name,
                            streamData.phy_addr[pack_idx],
                            streamData.data_size[pack_idx],
                            1000,
                        )
                finally:
                    self.encoder.ReleaseStream(self.venc_chn, streamData)

                os.exitpoint()

        except BaseException as e:
            print(f"Exception {e}")  # 捕获并打印异常 / Catch and print exceptions
        finally:
            self.runthread_over = True  # 设置线程结束标志 / Set thread completion flag
            # 资源清理由调用方的 stop() 完成，避免推流线程等待自身。
            # Resource cleanup is handled by the caller's stop().


def wifi_transmit(sensor,chn,ssid=None,password=None):
    # 允许调用方临时传入配置；默认读取未纳入 Git 的本地配置文件。
    if ssid is None or password is None:
        ssid, password = load_wifi_credentials()
    print("[WIFI] 连接网络中 Connecting to network ...")  # 提示正在连接网络 / Indicate network connection in progress
    # 也可以在此显示写入wifi名称和密码
    # ！！注意WiFi必须是2.4G频段
    isConnected = Connect_WIFI(ssid, password)
    if isConnected:
        print("[WIFI] 网络连接成功 Network connection successful")  # 连接成功提示 / Connection successful message
    else:
        import sys
        print("[WIFI] 网络连接失败 Network connection failed! Please check the configuration")
        sys.exit()         # 退出程序 / Exit program
    # 创建 RTSP 服务器对象 / Create RTSP server object
    rtsp = RtspServer(
    session_name="k230video",
    port=9954,
    sensor=sensor,
    sensor_chn=chn,
    width=512,
    height=288,
    bit_rate=600,
    frame_rate=15,
    initMediaManager=True
    )
    rtsp.start()
    rtsp_address = rtsp.get_rtsp_url()
    print("[RTSP] Started successfully, address:", rtsp_address)  # 启动成功并显示地址 / Started successfully and show address
    return rtsp  

if __name__ == "__main__":
    print("[WIFI] 连接网络中 Connecting to network ...")  # 提示正在连接网络 / Indicate network connection in progress
    # 连接 WiFi 网络 / Connect to WiFi network
    wifi_ssid, wifi_password = load_wifi_credentials()
    isConnected = Connect_WIFI(wifi_ssid, wifi_password)
    if isConnected:
        print("[WIFI] 网络连接成功 Network connection successful")  # 连接成功提示 / Connection successful message
    else:
        import sys
        print("[WIFI] 网络连接失败 Network connection failed! Please check the configuration")
        # 连接失败提示 / Connection failed message
        time.sleep_ms(10)  # 延时 10 毫秒 / Delay for 10 milliseconds
        sys.exit()         # 退出程序 / Exit program

    print("[RTSP] Starting ...")  # 提示 RTSP 启动 / Indicate RTSP starting
    time.sleep(1)                # 延时 1 秒 / Delay for 1 second

    # 创建 RTSP 服务器对象 / Create RTSP server object
    rtspserver = RtspServer()
    # 启动 RTSP 服务器 / Start RTSP server
    rtspserver.start()
    # 打印 RTSP 地址 / Print RTSP URL
    rtsp_address = rtspserver.get_rtsp_url()
    print("[RTSP] Started successfully, address:", rtsp_address)  # 启动成功并显示地址 / Started successfully and show address

    # 推流 60 秒 / Stream for 60 seconds
    while True:
        time.sleep_ms(10)  # 每 10 毫秒循环一次 / Loop every 10 milliseconds
    # 停止 RTSP 服务器 / Stop RTSP server
    rtspserver.stop()
    print("done")  # 提示完成 / Indicate completion