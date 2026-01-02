import cv2
import mediapipe as mp
import os

def draw_hand_landmarks(frame, hand_landmarks, color=(0,255,0)):
    h, w = frame.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for start_i, end_i in connections:
        start = hand_landmarks[start_i]
        end = hand_landmarks[end_i]
        start_pt = (int(start.x * w), int(start.y * h))
        end_pt = (int(end.x * w), int(end.y * h))
        cv2.line(frame, start_pt, end_pt, color, 2)

    for landmark in hand_landmarks:
        pt = (int(landmark.x * w), int(landmark.y * h))
        color = (0,0,255)
        cv2.circle(frame, pt, 4, color, -1)

    return frame

def is_hand_open(hand_landmarks, hand_label):
    """
    Basic open/closed detector using finger extension.
    Returns True if the palm looks open.
    """
    # Index, middle, ring, pinky: tip vs PIP (lower y = higher on screen)
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    extended = []
    for tip_i, pip_i in zip(tips, pips):
        extended.append(hand_landmarks[tip_i].y < hand_landmarks[pip_i].y)

    # Thumb: use x direction based on handedness
    thumb_tip = hand_landmarks[4]
    thumb_ip = hand_landmarks[3]
    if hand_label == "Left":
        thumb_extended = thumb_tip.x < thumb_ip.x
    else:
        thumb_extended = thumb_tip.x > thumb_ip.x

    extended_count = sum(extended) + (1 if thumb_extended else 0)
    return extended_count >= 4

def main(camera_index=None):

    cap = cv2.VideoCapture(0 if camera_index is None else camera_index)
    if not cap.isOpened():
        raise RuntimeError("yo twin we cannot open camera {}".format(camera_index))
    
    has_solutions = hasattr(mp, "solutions")
    if has_solutions:
        mp_hands = mp.solutions.hands
        mp_drawing = mp.solutions.drawing_utils
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_detector = None
    else:
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        candidates = [
            os.path.join(os.path.dirname(__file__), "hand_landmarker.task"),
            os.path.join(os.path.dirname(__file__), "..", "hand_landmarker.task"),
        ]
        model_path = next((p for p in candidates if os.path.exists(p)), None)
        if model_path is None:
            raise FileNotFoundError(
                "hand_landmarker.task not found. Place it in handtrack/ or meme_recon/."
            )

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_detector = vision.HandLandmarker.create_from_options(options)
        hands = None

    print("starting hand tracking, press 'q' to quit")
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("cannot read from webcam")
            break

        frame_count += 1
        if frame_count % 60 == 0:
            print("current frame at {}...".format(frame_count))

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if has_solutions:
            results = hands.process(rgb)
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = "Hand"
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        hand_label = results.multi_handedness[idx].classification[0].label
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                    )
                    status = "open" if is_hand_open(hand_landmarks, hand_label) else "closed"
                    cv2.putText(
                        frame,
                        "{}: {}".format(hand_label, status),
                        (10, 30 + idx * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = hand_detector.detect(mp_image)
            if results.hand_landmarks:
                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    hand_label = "Hand"
                    if results.handedness and idx < len(results.handedness):
                        hand_label = results.handedness[idx][0].category_name
                    draw_hand_landmarks(frame, hand_landmarks)
                    status = "open" if is_hand_open(hand_landmarks, hand_label) else "closed"
                    cv2.putText(
                        frame,
                        "{}: {}".format(hand_label, status),
                        (10, 30 + idx * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

        cv2.imshow("Hand Tracking - sh4wn", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    if hands is not None:
        hands.close()
    if hand_detector is not None:
        hand_detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        try:
            idx = int(sys.argv[1])
        except ValueError:
            print("Usage: python app.py [camera_index]")
            raise SystemExit(1)
        main(camera_index=idx)
    else:
        main()



