import time

import json
from ybUtils.YbUart import YbUart

rx_buff = None
# 串口接收函数
def uart_receive(uart_id,buff_size=128):
    global rx_buff 

    received_data = None
   
    try:
        data_chunk =  uart_id.read()

        if data_chunk:
            if rx_buff:
                rx_buff += data_chunk
            else:
                rx_buff = data_chunk

            if len(rx_buff) > buff_size:
                rx_buff = None
                print("超出缓冲区大小，丢弃数据")
            else:
                for i in range(len(rx_buff)):
                    value = rx_buff[i]
                    if value == '\n' or value == 10:
                        received_data = json.loads(rx_buff[:i])
                        rx_buff = rx_buff[i+1:]
                        break
    except Exception as e:
        print("Error receiving data:", e)

    return received_data

def uart_send(uart_id, data):
    """发送 A5 5A + 三个十进制数位字节，data 必须位于 0..999。"""
    if not 0 <= data <= 999:
        raise ValueError("UART data must be in range 0..999")
    digit_data = bytes([data // 100,data // 10 % 10,data % 10])
    send_data = bytes([0xA5, 0x5A]) + digit_data
    print(send_data.hex(' '))
    uart_id.write(send_data)


if __name__ == "__main__":
    uart = YbUart(baudrate=115200)
    uart_send(uart,123)
    time.sleep_ms(1000)
    uart_send(uart,640)
    
    # while True:
    #     data_0 = uart_receive(uart)
    #     if data_0:
    #         print(data_0)

