#!/usr/bin/env python3
import logging
import sys
import streamlink
import os.path
from time import time
import cv2

log = logging.getLogger(__name__)

save_every_n_seconds = 5

def stream_to_url(url, quality='best'):
    streams = streamlink.streams(url)
    if streams:
        return streams[quality].to_url()
    else:
        raise ValueError("No steams were available")

def main(url, quality='best', fps=30.0):
    stream_url = stream_to_url(url, quality)
    log.info("Loading stream {0}".format(stream_url))
    cap = cv2.VideoCapture(stream_url)

    frame_time = int((1.0 / fps) * 1000.0)

    previous = time()
    delta = 0

    # Keep looping
    while True:
        # Get the current time, increase delta and update the previous variable
        current = time()
        delta += current - previous
        previous = current

        try:
            ret, frame = cap.read()
            if ret:
                img = cv2.resize(frame, (800, 450)) 
                cv2.imshow('frame', img)

                if delta > save_every_n_seconds:
                    cv2.imwrite('data/' + str(int(current)) + '.png', frame)
                    delta = 0
                    
                if cv2.waitKey(frame_time) & 0xFF == ord('q'):
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
    opts = parser.parse_args()

    main('https://www.youtube.com/watch?v=KGEekP1102g', 'best', 30)
