import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

right_eye_idx = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]#,466,388,387,386,385,384,398,362]
left_eye_idx = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]#,246,161,160,159,158,157,173,133]


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
# cap = cv2.VideoCapture(1)

# cap = cv2.VideoCapture(0)#cv2.CAP_DSHOW)
cap = cv2.VideoCapture(2)#,cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tracker = cv2.TrackerKCF_create()
success_tracking = False

with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        print(image.shape)
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

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
        mask = np.zeros((height, width), np.uint8)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks)
            # break
            for landmarks in results.multi_face_landmarks:
                left_pts = []
                right_pts = []
                for idx in left_eye_idx:
                    x = int(landmarks.landmark[idx].x * width)
                    y = int(landmarks.landmark[idx].y * height)
                    left_pts.append([x, y])
                    # cv2.circle(image, (x, y), 1, (0, 255, 0), 1)

                for idx in right_eye_idx:
                    x = int(landmarks.landmark[idx].x * width)
                    y = int(landmarks.landmark[idx].y * height)
                    right_pts.append([x, y])
                    # cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
                    # cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255), 1)
                left_pts = np.array(left_pts)
                # cv2.polylines(image, [left_pts], False, (255, 0, 0), 1)
                cv2.fillPoly(mask, [left_pts], 255)

                right_pts = np.array(right_pts)
                # cv2.polylines(image, [right_pts], False, (255, 0, 0), 1)
                cv2.fillPoly(mask, [right_pts], 255)

                only_eyes = cv2.bitwise_and(image, image, mask=mask)
                cv2.imshow("only eyes", only_eyes)
                # crop the left eye
                x1_left = int(landmarks.landmark[63].x * width)
                y1_left = int(landmarks.landmark[63].y * height)
                x2_left = int(landmarks.landmark[114].x * width)
                y2_left = int(landmarks.landmark[114].y * height)
                roi_left = only_eyes[y1_left:y2_left, x1_left:x2_left, :]
                roi_left_original = image[y1_left:y2_left, x1_left:x2_left, :]
                left = cv2.resize(roi_left, (150, 150))
                left_original = cv2.resize(roi_left_original, (150, 150))

                # crop the right eye
                x1_right = int(landmarks.landmark[336].x * width)
                y1_right = int(landmarks.landmark[336].y * height)
                x2_right = int(landmarks.landmark[346].x * width)
                y2_right = int(landmarks.landmark[346].y * height)
                roi_right = only_eyes[y1_right:y2_right, x1_right:x2_right, :]
                roi_right_original = image[y1_right:y2_right, x1_right:x2_right, :]
                right = cv2.resize(roi_right, (150, 150))
                right_original = cv2.resize(roi_right_original, (150, 150))

                right_mask = mask[y1_right:y2_right, x1_right:x2_right]
                right_mask = cv2.resize(right_mask, (150, 150))

                # print("left",roi_left.shape)
                # print("right",roi_right.shape)


                right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
                # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                # right_gray = cv2.filter2D(right_gray, -1, kernel)

                # right_gray_blur = cv2.blur(right_gray, (3, 3), borderType=cv2.BORDER_REFLECT)
                # right_gray = cv2.addWeighted(right_gray, 2, right_gray, -1, 0)
                cv2.imshow("right gray", right_gray)
                right_gray = cv2.equalizeHist(right_gray)
                # histr = cv2.calcHist([right_gray], [0], right_mask, [64], [0, 256])
                # plt.plot(histr)
                # plt.xlim([0, 64])
                # plt.show()
                # plt.hist(right_gray.ravel(), 16, [0, 256])
                # plt.show()
                _, right_bw = cv2.threshold(right_gray, 150, 255, cv2.THRESH_BINARY)
                cv2.imshow("right black and white", right_bw)

                if not success_tracking:
                    circles = cv2.HoughCircles(right_gray, cv2.HOUGH_GRADIENT, 1, 150, param1=50, param2=15,
                                               minRadius=20, maxRadius=35)
                    try:
                        circles = circles[0]
                        # print(circles)
                        for i in circles:
                            # print(i)
                            # draw the outer circle
                            cv2.circle(right, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 2)
                            cv2.circle(right_original, (int(i[0]), int(i[1])), int(i[2]), (255, 255, 255), 2)
                            # bbox (x,y,w,h)
                            bbox = (int(i[0]-i[2]), int(i[1]-i[2]), int(2*i[2]), int(2*i[2]))
                            tracker = cv2.TrackerKCF_create()
                            tracker.init(right_original, bbox)
                            success_tracking = True
                            print(f"start new tracker, {success_tracking}")
                            # print("drawing circle")
                            # draw the center of the circle
                            cv2.circle(right, (int(i[0]), int(i[1])), 2, (255, 255, 255), 3)
                            cv2.circle(right_original, (int(i[0]), int(i[1])), 2, (255, 255, 255), 3)
                            # cv2.imshow('eye', roi_color2)
                    except:
                        pass
                else:
                    print(f"size of img : {right_original.shape}")
                    success_tracking, bbox = tracker.update(right_original)
                    print(f"tracking ... {success_tracking}")
                    radius = int((bbox[2]+bbox[3])/4) # the half average of the width and height
                    x_center = int(bbox[0] + radius)
                    y_center = int(bbox[1] + radius)
                    cv2.circle(right_original, (x_center, y_center), radius, (255, 255, 255), 2)

                cv2.imshow("left_eye", left)
                cv2.imshow("right_eye", right)
                cv2.imshow("right_eye_original", right_original)


        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
