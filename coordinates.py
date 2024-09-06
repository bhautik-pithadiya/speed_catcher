import cv2

video_file_path = '/media/hlink/hd/vehical_test_videos/new_test_video/test_file_1_clipped.mp4'

cap = cv2.VideoCapture(video_file_path)
assert cap.isOpened(), "Error reading video file"

while True:
    ret,frame = cap.read() 
    frame = cv2.resize(frame,(1280,720))
    cv2.imshow("frame",frame)
    key = cv2.waitKey(0) & 0xFF

    if key == ord('q'):
        # Quit the video
        break
    elif key == 13:  # Enter key
        # Go to the next frame
        continue

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
