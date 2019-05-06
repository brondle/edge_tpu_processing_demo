import argparse
import platform
import subprocess
import io
import json
import numpy as np
import socket
import time

from edgetpu.detection.engine import DetectionEngine
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
from threading import Thread


UDP_IP = '127.0.0.1'
TCP_IP = '10.0.0.1'

DETECTION_RECEIVE_PORT = 9100
CLASSIFICATION_RECEIVE_PORT = 9101

SEND_SOCKET_PORT = 9102

DETECTION_IMAGE_BUFFER_SIZE = 66507

face_class_label_ids_to_names = {
    0: 'adi',
    1: 'brent',
    2: 'unknown'
}


def send_with_retry(sendSocket, message):
    #  print('sending', message)
    try:
        sendSocket.send(message.encode('utf-8'))
        # TODO: switch to UDP
        #  receiveSocket.sendto(message.encode('utf-8'), addr)
        #  senderSocket.sendto(message.encode('utf-8'), (UDP_IP, UDP_SEND_PORT))
    except ConnectionResetError:
        print('Socket disconnected...waiting for client')
        sendSocket, _ = sendSocket.accept()

    return sendSocket


def detect_face(engine, sendSocket):
    receiveSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiveSocket.bind((UDP_IP, DETECTION_RECEIVE_PORT))

    print('listening on %d ...' % DETECTION_RECEIVE_PORT)

    while True:
        print('waiting for packet')
        data, _ = receiveSocket.recvfrom(DETECTION_IMAGE_BUFFER_SIZE)

        if (len(data) > 0):
            start_s = time.time()

            try:
                image = Image.open(io.BytesIO(data)).convert('RGB')
            except OSError:
                print('Could not read image')
                continue

            # see https://coral.withgoogle.com/docs/reference/edgetpu.detection.engine/
            results = engine.DetectWithImage(
                image, threshold=0.25, keep_aspect_ratio=True, relative_coord=False, top_k=3)

            print('time to detect faces', (time.time() - start_s) * 1000)

            output = list(map(lambda result:
                              {'box': result.bounding_box.flatten().tolist()}, results))

            message = json.dumps({'detection': output}) + '|'
            sendSocket = send_with_retry(sendSocket, message)


def classify_face(engine, sendSocket):
    receiveSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiveSocket.bind((UDP_IP, CLASSIFICATION_RECEIVE_PORT))

    while True:
        print('waiting for packet')
        data, _ = receiveSocket.recvfrom(66507)

        if (len(data) > 0):
            start_s = time.time()

            try:
                image = Image.open(io.BytesIO(data)).convert('RGB')
            except OSError:
                print('could not read image')
                continue

            # see https://coral.withgoogle.com/docs/reference/edgetpu.classification.engine/
            results = engine.ClassifyWithImage(
                image, threshold=0.75, top_k=3)

            print('time to classify face', (time.time() - start_s) * 1000)
            output = dict(map(lambda result:
                              (face_class_label_ids_to_names[result[0]], result[1]), results))

            message = json.dumps({'classification': output}) + '|'
            sendSocket = send_with_retry(sendSocket, message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--detection_model', help='Path to the face detection model.', required=True)
    parser.add_argument(
        '--recognition_model', help='Path to the face recognition (image classification) model.', required=True)
    args = parser.parse_args()

    sendSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TODO: switch to UDP
    # sendSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Initialize engines
    detectionEngine = DetectionEngine(args.detection_model)
    recognitionEngine = ClassificationEngine(args.recognition_model)

    detectionThread = Thread(
        target=detect_face, args=(detectionEngine, sendSocket))
    recognitionThread = Thread(
        target=classify_face, args=(recognitionEngine, sendSocket))

    sendSocket.bind((TCP_IP, SEND_SOCKET_PORT))
    sendSocket.listen(1)
    sendSocket, _ = sendSocket.accept()
    # TODO: switch to UDP
    # sendSocket.bind((UDP_IP, UDP_SEND_PORT))

    detectionThread.start()
    recognitionThread.start()
    print('waiting for detection and classification client(s)')


if __name__ == '__main__':
    main()
