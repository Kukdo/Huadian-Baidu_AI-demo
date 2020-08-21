# import package
import cv2  # opencv-python

if __name__ == '__main__':
    xi = cv2.imread('./xi.jpg')  # read image
    rose = cv2.imread('./rose.jpg')
    print(xi.shape, rose.shape)  # print the shape of the image：height, width, channel

    rose = cv2.resize(rose, dsize=(268, 335))  # resize the image [dsize(width, height)]
    # @param gamma scalar added to each sum.
    # xi * 0.7(alpha) + rose * 0.3(beta) + 1(gamma)
    mix = cv2.addWeighted(xi, 0.7, rose, 0.3, 0)  # generate mix image

    cv2.imwrite('./xi_rose.jpg', mix)  # save image
    cv2.imshow('xi', mix)  # show image
    cv2.waitKey(0)  # 0 == inf, 5000(ms) == 5s
    cv2.destroyAllWindows()  # release memory
