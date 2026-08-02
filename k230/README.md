# K230 板端部署

本目录包含 K230/CanMV 端的钢球检测、RTSP 图传和 UART 坐标发送代码。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `user_main/main.py` | 创建相机与显示管线，运行检测、RTSP 和 UART |
| `user_main/yolo_benchmark.py` | AI2D 前处理、KPU 推理、TopK16 后处理、NMS 和结果绘制 |
| `user_main/wifi_transmit.py` | 连接 2.4 GHz Wi-Fi，并通过 H.264/RTSP 推流 |
| `user_main/wifi_config.py.example` | Wi-Fi 配置模板 |
| `user_main/user_uart.py` | 将钢球中心横坐标编码后发送到下位机 |

## 部署文件

将目录复制到板端：

```text
k230/user_main/  ->  /sdcard/user_main/
```

仓库已提供 `training/yolov8n_roi_320_30_topk.kmodel`，将其复制到：

```text
/data/yolov8n_roi_320_30_topk.kmodel
```

板端最终目录应包含：

```text
/sdcard/user_main/
├─ main.py
├─ yolo_benchmark.py
├─ wifi_transmit.py
├─ wifi_config.py
└─ user_uart.py
```

## Wi-Fi 配置

复制配置模板并填写 2.4 GHz Wi-Fi 信息：

```text
/sdcard/user_main/wifi_config.py.example
    -> /sdcard/user_main/wifi_config.py
```

配置文件格式：

```python
WIFI_SSID = "your_ssid"
WIFI_PASSWORD = "your_password"
```

## 模型与检测参数

| 项目 | 配置 |
| --- | --- |
| AI 图像尺寸 | `640×480` |
| 固定 ROI | `[0, 210, 640, 60]` |
| KModel 输入 | `[1, 3, 32, 320]`，`uint8`，NCHW |
| KModel 输出 | `[1, 5, 16]` |
| 输出通道 | `cx, cy, w, h, score` |
| 置信度阈值 | `0.60` |
| NMS 阈值 | `0.45` |
| 最大检测数量 | `1` |

AI2D 从 `640×480` 图像中裁剪固定 ROI，将 `640×60` 等比例缩放为 `320×30`，再上下各填充 1 行得到 `320×32`。检测结果经过置信度过滤、NMS 和坐标反变换后，重新映射到完整 AI 图像。

## RTSP 图传

程序使用同一 Sensor 的 `CAM_CHN_ID_1` 进行 H.264 编码，默认参数为：

| 项目 | 配置 |
| --- | --- |
| 分辨率 | `512×288` |
| 帧率 | `15 FPS` |
| 码率 | `600` |
| 端口 | `9954` |
| 会话名称 | `k230video` |

连接成功后，可在同一网络中的 VLC 等客户端打开：

```text
rtsp://<K230_IP>:9954/k230video
```

## UART 输出

串口波特率为 `115200`。检测到钢球后，发送 5 字节：

```text
A5 5A 百位 十位 个位
```

## 运行程序

完成文件复制、模型部署和 Wi-Fi 配置后，在 CanMV IDE 或板端运行：

```text
/sdcard/user_main/main.py
```

项目记录的约 `80 FPS` 钢球识别速度是在关闭 CanMV IDE 预览的条件下测得。开启 IDE 预览后，图像传输会占用额外资源。