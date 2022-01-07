import cv2
import dlib

face_detector = dlib.get_frontal_face_detector()
lm_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_frame)

    for face in faces:
        face_lm = lm_predictor(gray_frame, face)

        for i in range(36, 48):
            x, y = face_lm.part(i).x, face_lm.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)
            # cv2.putText(frame, str(i), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    cv2.imshow("face landmarks", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
