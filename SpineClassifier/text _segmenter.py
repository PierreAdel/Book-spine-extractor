import cv2

class TextSegmenter:

    def get_text(self, image):
        """
            to be called from the outside
        :return:
            list of words
        """
        pass

    def segment_text(self, image):
        """

        :return:
            binarized image
        """
        pass

    def recognize_text(self, binary_image):
        """

        :return:
            text included in image
        """
        pass


if __name__ == "__main__":
    pass
