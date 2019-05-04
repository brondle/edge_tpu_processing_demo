import argparse
import platform
import subprocess
from edgetpu.detection.engine import DetectionEngine
import socket
import io
import time
import numpy as np
import json
from PIL import Image


UDP_IP = '127.0.0.1'
TCP_IP = UDP_IP
#  TCP_IP = '10.0.0.1'
UDP_RECEIVE_PORT = 9100
#  UDP_SEND_PORT = 9101
TCP_PORT = 9101
#  BUFFER_SIZE = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', help='Path of the detection model.', required=True)
    args = parser.parse_args()

    # Initialize engine.
    engine = DetectionEngine(args.model)

    print('opening socket...')

    #  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    receiveSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #  senderSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    #  senderSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    receiveSocket.bind((UDP_IP, UDP_RECEIVE_PORT))
    #  senderSocket.bind((UDP_IP, UDP_SEND_PORT))

    print('listening...')

    _, width, height, _ = engine.get_input_tensor_shape()

    # imageSize = width*height*3

    print('waiting for client')

    conn, addr = s.accept()

    print('Connection address:', addr)
    # Open image.
    while 1:
        print('waiting for packet')
        data, addr = receiveSocket.recvfrom(66507)

        #  print('got packet of length', len(data))

        if (len(data) > 0):
            start_s = time.time()

            try:
                image = Image.open(io.BytesIO(data)).convert('RGB')
            except OSError:
                print('Could not read image')
                continue

            results = engine.DetectWithImage(
                image, threshold=0.25, keep_aspect_ratio=True, relative_coord=False, top_k=3)

            print('time to process image', (time.time() - start_s) * 1000)

            output = to_output(results)

            message = json.dumps({'results': output}) + '|'
            print(message)

            #  print('sending', message)
            try:
                conn.send(message.encode('utf-8'))
            except ConnectionResetError:
                print('Socket disconnected...waiting for client')
                conn, addr = s.accept()

            #  receiveSocket.sendto(message.encode('utf-8'), addr)
            #  senderSocket.sendto(message.encode('utf-8'), (UDP_IP, UDP_SEND_PORT))

    #  conn.close()


def to_output(results):
    return list(map(lambda result: {
        'box': result.bounding_box.flatten().tolist(),
    }, results))


if __name__ == '__main__':
    main()
