import cv2
import numpy as np


cap= cv2.VideoCapture("videos/car.mp4")

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('bg_removed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))

if (cap.isOpened() == False):
    print("Error opening video stream or file")


while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        print(frame);

        #
        # Mask
        mask = gradient.copy()
        mask[mask > 0] = 0
        cv2.fillPoly(mask, significant, 255)
        # Invert mask

        mask = np.logical_not(mask)

        # Finally remove the background
        frame[mask] = 0
        writer.write(frame)
        cv2.imshow('result', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
writer.release()
cv2.destroyAllWindows()
