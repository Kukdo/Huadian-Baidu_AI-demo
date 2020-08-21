import cv2

if __name__ == '__main__':
    img = cv2.imread('./han.jpg')  # person image
    head = cv2.imread('./head.jpg')  # head image

    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)

    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

    faces = face_detector.detectMultiScale(gray)
    # print(faces)
    for x, y, w, h in faces:
        # Overall replacement
        dog = cv2.resize(head, dsize=(w, h))  # resize the head to the person's head scale
        # Limited scope
        img[y:y+h, x:x+w] = dog

    cv2.imshow('Face_replacement', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
