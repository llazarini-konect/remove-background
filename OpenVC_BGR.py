import cv2
import numpy as np


def edgedetect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255 # Some values seem to go above 255. However RGB channels has to be within 0-255
    return sobel

def findSignificantContours (img, edgeImg):

    contours, heirarchy = cv2.findContours(edgeImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        # print(heirarchy)
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)
    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = edgeImg.size * 20 / 100  # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]]

        area = cv2.contourArea(contour)

        if area > tooSmall:
            significant.append([contour, area])

            # Draw the contour on the original image
            cv2.drawContours(img, [contour], 0, (0, 0, 0), 1, cv2.LINE_AA, maxLevel=1)
    epsilon = 0.001 * cv2.arcLength(contour, True)
    # or epsilon = 3, so slighter contour corrections
    approx = cv2.approxPolyDP(contour, epsilon, True)
    contour = approx

    significant.sort(key=lambda x: x[1])
    # print ([x[1] for x in significant]);
    return [x[0] for x in significant]


cap= cv2.VideoCapture("videos/car.mp4")

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer= cv2.VideoWriter('bg_removed.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (width,height))


if (cap.isOpened() == False):
    print("Error opening video stream or file")





while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:



        blur = cv2.medianBlur(frame, 13)
        cv2.imshow('BLUR', blur)
        edgeImg = np.max(np.array([edgedetect(blur[:, :, 0]), edgedetect(blur[:, :, 1]), edgedetect(blur[:, :, 2])]),
                         axis=0)

        mean = np.mean(edgeImg)
        # Zero any value that is less than mean. This reduces a lot of noise.
        edgeImg[edgeImg <= mean] = 0
        cv2.imshow('edge', edgeImg)


        kernel_dilation = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(edgeImg, kernel_dilation, iterations=1)
        cv2.imshow('dilation1', dilation)

        kernel_opening = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_opening, iterations=2)
        cv2.imshow('opening1', opening)


        kernel_gradient = np.ones((2,2),np.uint8)
        gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel_gradient)

        cv2.imshow('Gradient', gradient)
        edgeImg_8u = np.asarray(opening, np.uint8)


        # Find contours
        significant = findSignificantContours(frame, edgeImg_8u)


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
