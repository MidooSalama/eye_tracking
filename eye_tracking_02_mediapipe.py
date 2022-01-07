import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_2eyes(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width, _ = image.shape
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            x1_left = int(landmarks.landmark[63].x * width)
            y1_left = int(landmarks.landmark[63].y * height)
            x2_left = int(landmarks.landmark[114].x * width)
            y2_left = int(landmarks.landmark[114].y * height)
            left_eye_box = (y1_left, x1_left, y2_left-y1_left, x2_left-x1_left)
            # roi_left = image[y1_left:y2_left, x1_left:x2_left, :]
            # left = cv2.resize(roi_left, (150, 150))

            x1_right = int(landmarks.landmark[336].x * width)
            y1_right = int(landmarks.landmark[336].y * height)
            x2_right = int(landmarks.landmark[346].x * width)
            y2_right = int(landmarks.landmark[346].y * height)
            right_eye_box = (y1_right, x1_right, y2_right-y1_right, x2_right-x1_right)
            # roi_right = image[y1_right:y2_right, x1_right:x2_right, :]
            # right = cv2.resize(roi_right, (150, 150), interpolation=cv2.INTER_CUBIC)

            return left_eye_box, right_eye_box
    return None, None


# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
count = 0
lost = True
left_box = None
right_box = None
tracker_left = cv2.TrackerKCF_create()
tracker_right = cv2.TrackerKCF_create()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    count += 1
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.flip(image, 1)
    if count == 60 or lost:
        # print(f"init tracker at {count}")
        count = 0
        left_box, right_box = extract_2eyes(image)
        if (left_box is None) or (right_box is None):
            continue
        lost = False
        # print(f"init tracker with {left_box}, {right_box}")
        tracker_left = cv2.TrackerKCF_create()
        tracker_right = cv2.TrackerKCF_create()
        tracker_left.init(image, left_box)
        tracker_right.init(image, right_box)

    try:
        ok_left, left_box = tracker_left.update(image)
        ok_right, right_box = tracker_right.update(image)
        # print(f"tracker results {ok_left}, {left_box}, {ok_right}, {right_box}")
        if not ok_left or not ok_right:
            lost = True
            continue

        roi_left = image[int(left_box[0]): int(left_box[0] + left_box[2]),
                   int(left_box[1]):int(left_box[1] + left_box[3]),:]
        left = cv2.resize(roi_left, (150, 150))
        cv2.imshow("left eye", left)

        roi_right = image[int(right_box[0]): int(right_box[0] + right_box[2]),
                    int(right_box[1]):int(right_box[1] + right_box[3]), :]
        right = cv2.resize(roi_right, (150, 150))
        cv2.imshow("right eye", right)
    except Exception as e:
        print("error while tracking")

    cv2.imshow('MediaPipe FaceMesh', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
