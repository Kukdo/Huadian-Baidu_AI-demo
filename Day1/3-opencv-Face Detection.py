import cv2

if __name__ == '__main__':
    img = cv2.imread('./nba.jpg')

    gray = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    # load feature xml (front face)
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
    faces = face_detector.detectMultiScale(gray)  # result: x, y, weight, height
    # print(faces)
    for x, y, w, h in faces:
        # rectangle mode
        # cv2.rectangle(img,
        #               pt1=(x,y),  # Upper left corner
        #               pt2=(x+w,y+h),  # Bottom right corner
        #               color=[0,0,255], # RED
        #               thickness=2)  # border

        # circle mode
        cv2.circle(img,
                   center=(x + w//2, y + h//2),  # Center of circle
                   radius=w//2,  # radius of circle
                   color=[0, 0, 255],
                   thickness=2)
    cv2.imshow('face', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
