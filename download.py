#!/usr/bin/env python3
import logging
import sys
import streamlink
import numpy as np
import os.path
from time import time
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

log = logging.getLogger(__name__)

save_every_n_seconds = 5

def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError("No steams were available")

datasets = {
    'nl1': 'https://www.youtube.com/watch?v=KGEekP1102g',
    'jp1': 'https://www.youtube.com/watch?v=PmrWwYTlAVQ' # https://worldcams.tv/japan/tokyo/shibuya-crossing
}

def main(webcam):
    global datasets
    stream_url = stream_to_url(datasets[webcam], 'best')

    # Keep looping
    while True:
        try:
            cap = cv2.VideoCapture(stream_url)
            ret, frame = cap.read()
            if ret:
                print('Parsing...')
                variance = str(round(np.var(frame), 3))
                res = cv2.resize(frame, (800, 450))
                cv2.putText(res, variance, (100, 100), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('download_output', res)
                    
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Frame downloading via Streamlink")
    parser.add_argument('webcam', type=str, help='Which webcam source to stream')
    opts = parser.parse_args()

    try:
        main(opts.webcam)
    except AttributeError:
        print('You need to pass a valid webcam name')
        exit
