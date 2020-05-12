#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imutils import face_utils
import imutils
import dlib
import cv2


# In[2]:


path_shape_predictor = "./shape_predictor_68_face_landmarks.dat"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_shape_predictor)

# create a videoCapture object with a video file or a capture device
cap = cv2.VideoCapture('./Ladykracher.mp4')

# check if we will successfully open the file
if not cap.isOpened():
    print("Error opening the file.")
    assert(False)

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# # Define the codec and create VideoWriter object.The output is stored in 'output.mp4' file.
# out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

# read until the end of the video frame by frame
while cap.isOpened():
    # cap.read (): decodes and returns the next video frame
    # variable ret: will get the return value by retrieving the camera frame, true or false (via "cap")
    # variable frame: will get the next image of the video (via "cap")
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # to display the frame
        cv2.imshow("Output", frame)
        
        # Write the frame into the file 'output.avi'
        # out.write(frame)

        # waitKey (0): put the screen in pause because it will wait infinitely that key
        # waitKey (n): will wait for keyPress for only n milliseconds and continue to refresh and read the video frame using cap.read ()
        # ord (character): returns an integer representing the Unicode code point of the given Unicode character.
        if cv2.waitKey(1) == ord('e'):
            break
    else:
        break

# to release software and hardware resources
cap.release()
# out.release()

# to close all windows in imshow ()
cv2.destroyAllWindows()

