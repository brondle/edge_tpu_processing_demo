#!/usr/bin/env python

import argparse
import platform
import subprocess
import io
import logging
import json
import numpy as np
import socket
import time

from edgetpu.detection.engine import DetectionEngine
from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
from threading import Thread


UDP_IP = '127.0.0.1'
# TCP_IP = '10.0.0.1'
TCP_IP = UDP_IP

DETECTION_RECEIVE_PORT = 9100
CLASSIFICATION_RECEIVE_PORT = 9101

SEND_SOCKET_PORT = 9102

DETECTION_IMAGE_BUFFER_SIZE = 66507

face_class_label_ids_to_names = {
    0: 'adi',
    1: 'brent',
    2: 'unknown'
}

logger = logging.getLogger(__name__)


def send_with_retry(sendSocket, message):
    #  logger.info('sending', message)
    try:
        sendSocket.send(message.encode('utf-8'))
        # TODO: switch to UDP
        #  receiveSocket.sendto(message.encode('utf-8'), addr)
        #  senderSocket.sendto(message.encode('utf-8'), (UDP_IP, UDP_SEND_PORT))
    except ConnectionResetError:
        logger.info('Socket disconnected...waiting for client')
        sendSocket, addr = sendSocket.accept()

    return sendSocket


def detect_face(engine, sendSocket):
    receiveSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiveSocket.bind((UDP_IP, DETECTION_RECEIVE_PORT))

    logger.info('listening on %d ...' % DETECTION_RECEIVE_PORT)

    while 1:
        logger.debug('waiting for packet')
        data, _ = receiveSocket.recvfrom(DETECTION_IMAGE_BUFFER_SIZE)

        if (len(data) > 0):
            start_s = time.time()

            try:
                image = Image.open(io.BytesIO(data)).convert('RGB')
            except OSError:
                logger.info('Could not read image')
                continue

            # see https://coral.withgoogle.com/docs/reference/edgetpu.detection.engine/
            results = engine.DetectWithImage(
                image, threshold=0.25, keep_aspect_ratio=True, relative_coord=False, top_k=3)

            # logger.debug('time to detect faces: %d\n' %
            #              (time.time() - start_s) * 1000)

            output = list(map(lambda result:
                              {'box': result.bounding_box.flatten().tolist()}, results))
            logger.debug(output)

            message = json.dumps({'detection': output})
            sendSocket = send_with_retry(sendSocket, message)


def classify_face(engine, sendSocket):
    receiveSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiveSocket.bind((UDP_IP, CLASSIFICATION_RECEIVE_PORT))

    while 1:
        logger.info('waiting for packet')
        data, _ = receiveSocket.recvfrom(66507)

        if (len(data) > 0):
            start_s = time.time()

            try:
                image = Image.open(io.BytesIO(data)).convert('RGB')
            except OSError:
                logger.info('could not read image')
                continue

            logger.info('CLASSIFYING')
            image.save('crop', 'JPEG')
            # see https://coral.withgoogle.com/docs/reference/edgetpu.classification.engine/
            results = engine.ClassifyWithImage(
                image, threshold=0.6, top_k=3, resample=Image.BILINEAR)

            logger.debug('time to classify face: %d\n' %
                         (time.time() - start_s) * 1000)

            if (len(results) > 0):
                logger.info(results)

                # sort by confidence, take the highest, return the label
                highest_confidence_result = sorted(
                    results, key=lambda result: result[1], reverse=True)[0]
                highest_confidence_label_id = highest_confidence_result[0]
                highest_confidence_interval = highest_confidence_result[1]

                try:

                    # output = list(
                    #     map(lambda result: face_class_label_ids_to_names[result[0]], results))

                    message = json.dumps({
                        'classification': face_class_label_ids_to_names[highest_confidence_label_id],
                        'confidence': str(highest_confidence_interval)
                    })
                    sendSocket = send_with_retry(sendSocket, message)
                except KeyError:
                    logger.error(
                        'classified label "%d" not recognized' % highest_confidence_label_id)
            else:
                logger.debug('could not classify image')


def start_server():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--detection_model', help='Path to the face detection model.', required=True)
    parser.add_argument(
        '--recognition_model', help='Path to the face recognition (image classification) model.', required=True)
    args = parser.parse_args()

    sendSocketRaw = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # TODO: switch to UDP
    # sendSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Initialize engines
    detectionEngine = DetectionEngine(args.detection_model)
    recognitionEngine = ClassificationEngine(args.recognition_model)

    logger.info('listening for ML requests on %d ...' % DETECTION_RECEIVE_PORT)

    sendSocketRaw.bind((TCP_IP, SEND_SOCKET_PORT))
    sendSocketRaw.listen(1)
    logger.info('waiting for client to connect...')
    sendSocket, addr = sendSocketRaw.accept()
    # TODO: switch to UDP
    # sendSocket.bind((UDP_IP, UDP_SEND_PORT))

    # at this point, we know the processing client has opened the TCP socket
    logger.info('processing client connected')
    detectionThread = Thread(
        target=detect_face, args=(detectionEngine, sendSocket))
    recognitionThread = Thread(
        target=classify_face, args=(recognitionEngine, sendSocket))

    detectionThread.start()
    recognitionThread.start()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    consoleHandler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(consoleHandler)

    logger.info('running server as main')
    start_server()
