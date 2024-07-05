import cv2

min_area = 500
output_video_path = "motion_output.avi"
video_source = 0

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

fgbg = cv2.createBackgroundSubtractorMOG2()

motion_detected = False
motion_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        motion_detected = True
        motion_counter = fps

    if motion_detected:
        out.write(frame)
        motion_counter -= 1
        if motion_counter <= 0:
            motion_detected = False

    cv2.imshow('Motion Detection', frame)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
