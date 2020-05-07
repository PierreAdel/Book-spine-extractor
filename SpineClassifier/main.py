import cv2

from book_spine_extractor import SpineExtractor
from text_segmenter import process_spine_from_extractor

SHELVES_PATH = 'images/shelves/'


def resize_image(im):
    max_height = 800
    if im.shape[0] > max_height:
        shape_ = (int(im.shape[1] * max_height / im.shape[0]), max_height)
        im = cv2.resize(im, shape_)
    return im


if __name__ == "__main__":
    for i in range(16):
        img = cv2.imread(SHELVES_PATH + str(i) + '.jpg')
        img = resize_image(img)
        extractor = SpineExtractor(img)
        extractor.extract()
        for spine in extractor.spines:
            print(process_spine_from_extractor(spine))
        print("####################################")
