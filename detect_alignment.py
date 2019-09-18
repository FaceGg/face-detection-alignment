from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", type=str, required=True,
        help="path to OpenCV's deep learning face detector")
    ap.add_argument("-a", "--alignment", type=str, required=True,
                    help="path to OpenCV's deep learning face alignment")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
    modelPath = os.path.sep.join([args["detector"],
        "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    protoPath2 = os.path.sep.join([args["alignment"], "2_deploy.prototxt"])
    modelPath2 = os.path.sep.join([args["alignment"], "2_solver_iter_800000.caffemodel"])
    net2 = cv2.dnn.readNetFromCaffe(protoPath2, modelPath2)


    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:

        frame = vs.read()
        frame = imutils.resize(frame, width=600)

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        start = time.time()
        detections = net.forward()
        end = time.time()
        #print('detect times : %.3f ms'%((end - start)*1000))

        for i in range(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > args["confidence"]:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                ww = (endX - startX) // 10
                hh = (endY - startY) // 5
                startX = startX - ww
                startY = startY + hh
                endX = endX + ww
                # endY = endY + hh

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                x1 = int(startX)
                y1 = int(startY)
                x2 = int(endX)
                y2 = int(endY)

                roi = frame[y1:y2, x1:x2]
                gary_frame = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                resize_mat = np.float32(gary_frame)

                m = np.zeros((40, 40))
                sd = np.zeros((40, 40))
                mean, std_dev = cv2.meanStdDev(resize_mat, m, sd)
                new_m = mean[0][0]
                new_sd = std_dev[0][0]
                new_frame = (resize_mat - new_m) / (0.000001 + new_sd)

                blob2 = cv2.dnn.blobFromImage(cv2.resize(new_frame, (40, 40)), 1.0,(40, 40), (0, 0, 0))
                net2.setInput(blob2)
                align = net2.forward()

                for i in range(0,68):
                    x = align[0][2 * i] * (x2 - x1) + x1
                    y = align[0][2 * i + 1] * (y2 - y1) + y1
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 0, 255), 2)

                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)



        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()