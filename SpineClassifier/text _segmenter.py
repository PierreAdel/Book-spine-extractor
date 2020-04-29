from __future__ import annotations
import cv2
import pytesseract
import numpy as np
from typing import Type

IMAGE_PATH = 'images/'
NORMAL_TEXT_PATH = IMAGE_PATH + 'normal_text/'
SHELVES_PATH = IMAGE_PATH + 'normal_text/'
SPINES_PATH = IMAGE_PATH + 'spines/'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class BoundingBoxWrapper:
    def __init__(self, stats, img_shape):
        self.boxes = []
        self.add_boxes(img_shape, stats)

    def add_boxes(self, img_shape, stats):
        for stat in enumerate(stats):
            box = BoundingBox(stat)
            # todo do multilevel nesting and optimize method

            if box.is_valid(img_shape):
                inside = any(box.inside(stored_box) for stored_box in self.boxes)
                if inside:
                    continue

                self.boxes.append(box)

    def get_boxes(self):
        return sorted(self.boxes, key=lambda box: box.area, reverse=True)


class BoundingBox:

    def __init__(self, stat):
        self.intersected_area = 0
        self.label, (self.left, self.up, self.width, self.height, area) = stat

        self.area = self.width * self.height
        self.bottom = self.up + self.height
        self.right = self.left + self.width
        self.top_left = (self.left, self.up)
        self.top_right = (self.right, self.up)
        self.bottom_left = (self.left, self.bottom)
        self.bottom_right = (self.right, self.bottom)
        self.rectangle_args = [self.top_left, self.bottom_right, (0, 0, 255), 1]

    def is_valid(self, img_shape):
        if self.width * self.height > img_shape[0] * img_shape[1] * 0.15:
            return False
        elif self.height > img_shape[0] * 0.82:
            return False
        elif (0.2 < self.width / self.height < 5 and
              max(self.height, self.width) > 0.05 * img_shape[0]):
            return True
        else:
            return False

    # https://stackoverflow.com/questions/27152904/calculate-overlapped-area-between-two-rectangles
    def intersection_area(self, rec):
        dx = min(self.right, rec.right) - max(self.left, rec.left)
        dy = min(self.bottom, rec.bottom) - max(self.up, rec.up)
        if (dx >= 0) and (dy >= 0):
            return dx * dy
        return 0

    # todo change inside to use inclosed area % from total area instead of pure bounds
    def inside(self, stored_box: Type[BoundingBox]):
        """
        returns true if self is inside the stored box
        """
        self.intersected_area += self.intersection_area(stored_box)
        if self.intersected_area > 0.5 * self.area:
            return True
        else:
            return False

    def inside_boundary(self, stored_box: Type[BoundingBox]):
        if self.up > stored_box.up and self.bottom < stored_box.bottom and \
                self.left > stored_box.left and self.right < stored_box.right:
            return True
        else:
            return False


class TextSegmenter:

    def __init__(self, im):
        self.image = resize_image(im)
        self.image_area = im.shape[0] * im.shape[1]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray_chen = np.zeros(self.gray.shape, dtype=np.uint8) * 255
        self.gray_otsu = np.zeros(self.gray.shape, dtype=np.uint8) * 255
        self.get_character_edges(self.image)
        # self.segmented_img = self.segment_text()
        # cv2.imshow('binary_image', self.segmented_img)
        self.text = self.recognize_text()

    def get_text(self):
        return self.text

    def get_character_edges(self, image):
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        edges = thresh_edges(b) | thresh_edges(r) | thresh_edges(g)
        edges2 = thresh_edges(b) / 3 + thresh_edges(r) / 3 + thresh_edges(g) / 3
        edges2 = edges2.astype('uint8')
        edges3 = cv2.threshold(edges2, 160, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        kernel = np.ones((2, 2), np.uint8)
        edgesmorph = cv2.morphologyEx(edges3, cv2.MORPH_CLOSE, kernel)
        edgeblur = cv2.GaussianBlur(edges2, (3, 3), 3)
        edgeblur = cv2.threshold(edgeblur, 220, 255,
                                 cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        edges3 = edges
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges3, 8, cv2.CV_32S)

        box_wrapper = BoundingBoxWrapper(stats, self.image.shape)
        for box in box_wrapper.get_boxes():
            cv2.rectangle(image, *box.rectangle_args)

            ROI = labels[box.up:box.bottom, box.left:box.right]
            gray_roi = self.gray[box.up:box.bottom, box.left:box.right]

            edge_mask_of_label = (ROI == box.label)
            true_count = np.count_nonzero(edge_mask_of_label)
            labeled_gray_roi = edge_mask_of_label * gray_roi
            labeled_mean_intensity = np.sum(labeled_gray_roi) / true_count

            self.gray_chen[box.up:box.bottom, box.left:box.right] = cv2.threshold(gray_roi, labeled_mean_intensity, 255,
                                                                                  cv2.THRESH_BINARY)[1]

            self.gray_otsu[box.up:box.bottom, box.left:box.right] = cv2.threshold(gray_roi, 0, 255,
                                                                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[
                1]

            inside_mask = edges3[box.up:box.bottom, box.left:box.right] & self.gray[box.up:box.bottom,
                                                                          box.left:box.right]
            mean_inside = np.true_divide(inside_mask.sum(), (inside_mask != 0).sum())
            mean_outside = np.mean(self.gray[max(box.up - 8, 0):max(box.up, 1), box.left:box.right])
            if mean_inside < mean_outside:
                self.gray_otsu[box.up:box.bottom, box.left:box.right] = \
                    255 - self.gray_otsu[box.up:box.bottom, box.left:box.right]

        self.gray_chen = cv2.medianBlur(self.gray_chen, 3)
        self.gray_otsu = cv2.medianBlur(self.gray_otsu, 3)
        # self.gray_otsu = cv2.GaussianBlur(self.gray_otsu, (3, 3), 3)
        # cv2.imshow('gray_otsu', self.gray_otsu)
        # cv2.imshow('edges_bounded', image)
        # cv2.waitKey()

    def recognize_text(self):
        """

        :return:
            text included in image
        """
        config = '-c tessedit_char_whitelist=\ 01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(cv2.bitwise_not(self.gray_otsu), lang='eng', config=config)
        return text
        # return "chen *************\n" + pytesseract.image_to_string(cv2.bitwise_not(self.gray_chen)) + \
        #        "\notsu *************\n" + text


def resize_image(im):
    max_width = 1500
    if im.shape[1] > max_width:
        shape_ = (max_width, int(im.shape[0] * max_width / im.shape[1]))
        im = cv2.resize(im, shape_)
    return im


def blur(im):
    return cv2.GaussianBlur((im), (3, 3), 15)


def canny(im):
    return cv2.Canny(im, 60, 190)


def thresh_edges(im):
    return cv2.threshold(canny(im), 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


# todo fix bounding boxes
#   todo mawdoo3 el inner/ nesting
# todo fix inverting the colors if necessary
# todo check the mode of the pixel above the letter -- done
# todo sort boxes by size before detecting if its inside another one -- done
# todo idea: fill connected component -- cancelled
# todo idea: increase contrast
# todo combine boxes?

from googlesearch import search
from goodreads import client
import re
from dotenv import load_dotenv
load_dotenv()
import os
KEY = os.getenv("goodreads_key")
gc = client.GoodreadsClient(KEY, "")
if __name__ == "__main__":

    gamma = 0.5
    inverse_gamma = 1 / gamma

    look_up_table = np.zeros(256, dtype="uint8")

    for i in range(17):
        cv2.destroyAllWindows()
        img = cv2.imread(SPINES_PATH + str(i) + '.jpg')
        segmenter = TextSegmenter(img)
        detected_text = segmenter.get_text()
        try:
            try:
                book = gc.search_books(detected_text)[0]
            except TypeError:
                book = None
            goodreads_url_prefix = "https://www.goodreads.com/book/show/"
            if book is None:
                gen = search(detected_text + " " + goodreads_url_prefix, tld='com', lang='en', num=1,
                             pause=0.25)
                site = next(gen)
                if goodreads_url_prefix not in site:
                    site = next(search(detected_text + " site:" + goodreads_url_prefix, tld='com', lang='en',
                                       num=1, pause=0.25))
                id = re.search("[0-9]+", site).group()
                book = gc.book(id)
            book_json = {
                'title': book.title,
                'author': book.authors[0].name,
                'average_rating': book.average_rating,
                'url': goodreads_url_prefix + book.gid
            }
            print(book_json)
        except StopIteration:
            site = f'not found \n, query="{detected_text}"'
            print(site)
