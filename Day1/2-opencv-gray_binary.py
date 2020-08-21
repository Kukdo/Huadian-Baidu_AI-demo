import cv2

if __name__ == '__main__':
    rose = cv2.imread('./rose.jpg')
    print(rose.shape)

    gray = cv2.cvtColor(rose, code=cv2.COLOR_BGR2GRAY)  # convert to gray image
    print(gray.shape)  # two dimensions left
    # I set the threshold at 100,
    # if number <= 100 then 0,
    # else if number > 100 then 255
    threshold, binary_im = cv2.threshold(gray, 100, 255, type=cv2.THRESH_OTSU)
    # Image Binarization，0 or 255：black or white

    # Delete comment if you want to view the picture
    # cv2.imshow('rose', rose)
    # cv2.imshow('gray rose', gray)
    # cv2.imshow('Image Binarization', binary_im)

    cv2.waitKey(0)  # wait until key input
    cv2.destroyAllWindows()
