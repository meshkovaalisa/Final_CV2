import cv2
import numpy as np


def process_detections(frame, ui, face_cascade):
    display_frame = frame.copy()
    corn_method = ui.combo_corn_method.currentIndex()
    cont_method = ui.combo_cont_method.currentIndex()
    line_method = ui.combo_line_method.currentIndex()
    face_method = ui.combo_face_method.currentIndex()
    if corn_method == 0 and cont_method == 0 and line_method == 0 and face_method == 0:
        return display_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = ui.spin_det_thresh.value()
    param = ui.spin_det_param.value()

    if corn_method == 1:
        dst = cv2.dilate(cv2.cornerHarris(np.float32(gray), 2, 3, 0.04), None)
        display_frame[dst > (thresh / 10.0) / 100.0 * dst.max()] = [0, 0, 255]
    elif corn_method == 2:
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=param, qualityLevel=max(0.01, thresh / 100.0),
                                          minDistance=10)
        if corners is not None:
            for c in corners:
                cv2.circle(display_frame, tuple(map(int, c.ravel())), 5, (0, 255, 0), -1)
    elif corn_method == 3:
        display_frame = cv2.drawKeypoints(display_frame,
                                          cv2.FastFeatureDetector_create(threshold=thresh).detect(gray, None), None,
                                          color=(255, 0, 255))
    elif corn_method == 4:
        dst = cv2.cornerMinEigenVal(gray, blockSize=(param // 2) * 2 + 3, ksize=3)
        display_frame[dst > (101 - thresh) / 100.0 * dst.max()] = [0, 255, 255]
    elif corn_method == 5:
        display_frame = cv2.drawKeypoints(display_frame, cv2.SIFT_create(nfeatures=param).detect(gray, None), None,
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(0, 255, 255))

    if cont_method == 1:
        display_frame[cv2.Canny(gray, thresh * 2, thresh * 4) > 0] = [255, 255, 255]
    elif cont_method == 2:
        ksize = max(1, min(7, (param // 20) * 2 + 1))
        magnitude = cv2.addWeighted(cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=ksize)), 0.5,
                                    cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=ksize)), 0.5, 0)
        display_frame[cv2.subtract(magnitude, thresh * 2) > 10] = [0, 255, 0]
    elif cont_method == 3:
        contours, _ = cv2.findContours(cv2.Canny(gray, thresh, thresh * 2), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_frame, [c for c in contours if len(c) > param], -1, (255, 150, 0), 2)

    if line_method == 1:
        lines = cv2.HoughLinesP(cv2.Canny(gray, 50, 150), 1, np.pi / 180, threshold=max(10, thresh),
                                minLineLength=max(5, param), maxLineGap=15)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    elif line_method == 2:
        lines, _, _, _ = cv2.createLineSegmentDetector().detect(gray)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = map(int, line.flatten())
                if np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) > param:
                    cv2.line(display_frame, (x1, y1), (x2, y2), (0, 130, 255), 2)

    if face_method == 1:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=max(3, int(thresh / 10)),
                                              minSize=(max(30, param), max(30, param)))
        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return display_frame
