import cv2
import pytesseract

IMAGE_PATH = 'images/'
NORMAL_TEXT_PATH = IMAGE_PATH + 'normal_text/'
SHELVES_PATH = IMAGE_PATH + 'normal_text/'
SPINES_PATH = IMAGE_PATH + 'spines/'


class TextSegmenter:

    def get_text(self, image):
        """
            to be called from the outside
        :return:
            list of words
        """
        binary_image = self.segment_text(image)
        cv2.imshow('binary_image', binary_image)
        cv2.waitKey()
        return self.recognize_text(binary_image)

    def segment_text(self, image):
        """

        :return:
            binarized image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray', gray)
        cv2.waitKey()

        fun = lambda im: cv2.threshold(im, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        b = image[:, :, 0]
        cv2.imshow('b', b)
        cv2.imshow('b_thresh', fun(b))

        g = image[:, :, 1]
        cv2.imshow('g', b)
        cv2.imshow('g_thresh', fun(g))

        r = image[:, :, 2]
        cv2.imshow('r', r)
        cv2.imshow('r_thresh', fun(r))
        cv2.waitKey()

        return cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def recognize_text(self, binary_image):
        """

        :return:
            text included in image
        """
        return pytesseract.image_to_string(binary_image)


if __name__ == "__main__":

    img = cv2.imread(SPINES_PATH + '6.jpg')
    img = cv2.medianBlur(img, 5)
    segmenter = TextSegmenter()
    print(segmenter.get_text(img))
