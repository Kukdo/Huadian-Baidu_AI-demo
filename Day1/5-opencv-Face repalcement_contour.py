import numpy as np
import cv2

if __name__ == '__main__':
    han = cv2.imread('han.jpg')
    head = cv2.imread('./head.jpg')

    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

    han_gray = cv2.cvtColor(han, code=cv2.COLOR_BGR2GRAY)
    head_gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)

    threshold, head_binary = cv2.threshold(head_gray, 50, 255, cv2.THRESH_OTSU)
    # cv2.imshow('hb', head_binary)
    # find all contours
    contours, hierarchy = cv2.findContours(head_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))  # calculate the area of each contour and store in 'areas'
    areas = np.asarray(areas)  # change to np.array
    index = areas.argsort()  # min -> max    index[-2] is the head

    # change the image to mask all 0 == black
    mask = np.zeros_like(head_gray, dtype=np.uint8)
    # contour to 255 == white

    # max -> min contour area
    mask = cv2.drawContours(mask, contours, index[-2], (255, 255, 255), thickness=-1)  # -1 means cv2.FILLED

    faces = face_detector.detectMultiScale(han_gray)

    for x, y, w, h in faces:
        mask2 = cv2.resize(mask, (w, h))
        head2 = cv2.resize(head, (w, h))
        for i in range(h):  # for each pixel to decide and replace
            # print("i= ", i)
            for j in range(w):
                # print("j= ", j)
                if (mask2[i, j] == 255).all():  # all 255 == white means it's contour
                    han[y + i, x + j] = head2[i, j]

    cv2.imshow('face', han)
    # cv2.imwrite('han_alter.jpg', han)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
