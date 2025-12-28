import cv2
from fer.fer import FER

detector = FER()

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect emotion
    results = detector.detect_emotions(frame)

    if results:
        emotions = results[0]["emotions"]
        top_emotion = max(emotions, key=emotions.get)
        cv2.putText(frame, top_emotion, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Emotion Detector", frame)

    # exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
