import cv2

video_source = 0

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Foreground Mask', fgmask)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()
