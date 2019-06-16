from pydarknet import Detector, Image
import cv2
import os
import numpy as np

net = Detector(bytes("../yolo/yolov3.cfg", encoding="utf-8"), bytes("../yolo/yolov3.weights", encoding="utf-8"), 0, bytes("../yolo/coco.data",encoding="utf-8"))

def sliding_window(image, step_size, window_size):
	# slide a window across the image
	for y in range(0, image.shape[0], step_size):
		for x in range(0, image.shape[1], step_size):
			# yield the current window
			yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def detect(in_file, out_file, steps):
    img = cv2.imread(in_file)

    (winW, winH) = (steps, steps)

    combined = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    print(combined.shape)


    # Find lowest non-existing index
    person_index = 0
    new_fname = '/people/' + str(person_index) + '.png'
    while os.path.exists(new_fname):
        person_index += 1
        new_fname = '/people/' + str(person_index) + '.png'

    for (fromX, fromY, window) in sliding_window(img, step_size=steps, window_size=(winW, winH)):
        toX = int(min(fromX+winW, img.shape[1]))
        toY = int(min(fromY+winH, img.shape[0]))
        print('Processing (' + str(fromX) + ', ' + str(fromY) + ') to (' + str(toX) + ', ' + str(toY) + ')...')

        results = net.detect(Image(window))

        for cat, score, bounds in results:
            if str(cat.decode("utf-8")) == 'person':
                x, y, w, h = bounds
                cv2.rectangle(window, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 128, 0), thickness=1)

                cv2.imwrite('people/' + str(person_index) + '.png', window[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)])
                person_index += 1

        combined[int(fromY):toY, int(fromX):toX] = window
        cv2.imwrite('out/' + str(out_file) + '.png', combined)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break

if __name__ == "__main__":
    detect('../data/1560687345.png', '3_4', 608 * 4)
    # detect('../data/1560687345.png', '2_2', 608 * 2)
    # detect('../data/1560687345.png', '2_1', 608)
