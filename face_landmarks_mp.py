import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# For static images:
# IMAGE_FILES = []
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# with mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     min_detection_confidence=0.5) as face_mesh:
#   for idx, file in enumerate(IMAGE_FILES):
#     image = cv2.imread(file)
#     # Convert the BGR image to RGB before processing.
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     # Print and draw face mesh landmarks on the image.
#     if not results.multi_face_landmarks:
#       continue
#     annotated_image = image.copy()
#     for face_landmarks in results.multi_face_landmarks:
#       print('face_landmarks:', face_landmarks)
#       mp_drawing.draw_landmarks(
#           image=annotated_image,
#           landmark_list=face_landmarks,
#           connections=mp_face_mesh.FACE_CONNECTIONS,
#           landmark_drawing_spec=drawing_spec,
#           connection_drawing_spec=drawing_spec)
#     cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
count = 0
right_eye_idx = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]#,466,388,387,386,385,384,398,362]
left_eye_idx = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]#,246,161,160,159,158,157,173,133]
with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        image = cv2.resize(image, (1280, 720))

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        height, width, _ = image.shape
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks)
            # break
            # for landmarks in results.multi_face_landmarks:
            #     x1_left = int(landmarks.landmark[63].x * width)
            #     y1_left = int(landmarks.landmark[63].y * height)
            #     x2_left = int(landmarks.landmark[114].x * width)
            #     y2_left = int(landmarks.landmark[114].y * height)
            #     roi_left = image[y1_left:y2_left, x1_left:x2_left, :]
            #     left = cv2.resize(roi_left, (150, 150))
            #
            #     x1_right = int(landmarks.landmark[336].x * width)
            #     y1_right = int(landmarks.landmark[336].y * height)
            #     x2_right = int(landmarks.landmark[346].x * width)
            #     y2_right = int(landmarks.landmark[346].y * height)
            #     roi_right = image[y1_right:y2_right, x1_right:x2_right, :]
            #     right = cv2.resize(roi_right, (150, 150))
            #
            #     # print("left",roi_left.shape)
            #     # print("right",roi_right.shape)
            #     cv2.imshow("left_eye", left)
            #     cv2.imshow("right_eye", right)
            #     # cv2.circle(image, (x1_left, y1_left), 1, (255, 0, 0), 1)
            #     # cv2.circle(image, (x2_left, y2_left), 1, (255, 0, 0), 1)
            #     # cv2.circle(image, (x1_right, y1_right), 1, (255, 0, 0), 1)
            #     # cv2.circle(image, (x2_right, y2_right), 1, (255, 0, 0), 1)
            #     # for idx, landmark in enumerate(landmarks.landmark):
            #     #     x = int(landmark.x * width)
            #     #     y = int(landmark.y * height)
            #     #     # cv2.circle(image, (x, y), 1, (255, 0, 0), 1)
            #     #     cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.25, (255, 255, 255), 1)
            for face_landmarks in results.multi_face_landmarks:
                left_pts = []
                right_pts = []
                # for idx in left_eye_idx:
                #     x = int(face_landmarks.landmark[idx].x * width)
                #     y = int(face_landmarks.landmark[idx].y * height)
                #     left_pts.append([x, y])
                #     cv2.circle(image, (x, y), 1, (0, 255, 0), 1)

                # for idx in right_eye_idx:
                #     x = int(face_landmarks.landmark[idx].x * width)
                #     y = int(face_landmarks.landmark[idx].y * height)
                #     right_pts.append([x, y])
                #     cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
                #     # cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
                # left_pts = np.array(left_pts)
                # cv2.polylines(image, [left_pts], False, (255, 0, 0), 1)
                # cv2.fillPoly(image, [left_pts], (255, 255, 255))
                #
                # right_pts = np.array(right_pts)
                # cv2.polylines(image, [right_pts], False, (255, 0, 0), 1)
                # cv2.fillPoly(image, [right_pts], (255,255,255))

                for i, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
                    cv2.putText(image, str(i),(x, y), cv2.FONT_HERSHEY_PLAIN,0.8, (0, 0, 255), 1)
            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACE_CONNECTIONS,
            #     landmark_drawing_spec=drawing_spec,
            #     connection_drawing_spec=drawing_spec)
        cv2.imshow('MediaPipe FaceMesh', image)

        key = cv2.waitKey(1)
        if key & 0xFF == 27:
            break
        elif key == ord('p'):
            cv2.imwrite('mediapipe_landmarks_points.png', image)
        elif key == ord(' '):
            count +=10
cap.release()
