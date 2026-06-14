import cv2

def process_motion(frame, ui, face_cascade, history_queue):
    display_frame = frame.copy()
    if ui.combo_motion_type.currentIndex() == 0:
        history_queue.clear()
        return display_frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(50, 50))
    if faces is not None and len(faces) > 0:
        for (x, y, w, h) in faces:
            cx, cy = int(x + w // 2), int(y + h // 2)
            is_moving = False
            if len(history_queue) > 0:
                old_cx, old_cy = history_queue[0]
                if ((cx - old_cx) ** 2 + (cy - old_cy) ** 2) ** 0.5 > (ui.spin_motion_thresh.value() / 3.0):
                    is_moving = True
            history_queue.append((cx, cy))
            if len(history_queue) > 4:
                history_queue.pop(0)
            color, txt = ((0, 255, 0), "AI MOTION DETECTED") if is_moving else ((0, 165, 255), "AI STILL")
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2 if is_moving else 1)
            cv2.putText(display_frame, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6 if is_moving else 0.5, color, 2 if is_moving else 1)
            break
    else:
        if len(history_queue) > 0:
            history_queue.pop(0)
    return display_frame
