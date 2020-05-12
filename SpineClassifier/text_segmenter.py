#!/usr/bin/python3.7
from __future__ import annotations
import cv2
import pytesseract
from typing import Type
import numpy as np
from goodreads.request import GoodreadsRequestException

IMAGE_PATH = 'images/'
NORMAL_TEXT_PATH = IMAGE_PATH + 'normal_text/'
SHELVES_PATH = IMAGE_PATH + 'normal_text/'
# SPINES_PATH = IMAGE_PATH + 'spines/'
SPINES_PATH = 'detected_spines/spine'
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img = None


class BoundingBoxWrapper:
    def __init__(self, stats, img_shape):
        self.boxes = []
        self.img_shape = img_shape
        self.add_boxes(self.img_shape, stats)

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

    def get_mask(self, edges, gray_img, im):
        gray_mask = np.zeros(gray_img.shape, dtype=np.uint8)
        for box in self.get_boxes():
            gray_roi = gray_img[box.up:box.bottom, box.left:box.right]
            gray_mask[box.up:box.bottom, box.left:box.right] = \
                cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

            inside_mask = edges[box.up:box.bottom, box.left:box.right] & \
                          gray_img[box.up:box.bottom, box.left:box.right]
            mean_inside = np.true_divide(inside_mask.sum(), (inside_mask != 0).sum())
            mean_outside = np.mean(gray_img[max(box.up - 8, 0):max(box.up, 1), box.left:box.right])
            if mean_inside < mean_outside:
                gray_mask[box.up:box.bottom, box.left:box.right] = \
                    255 - gray_mask[box.up:box.bottom, box.left:box.right]
            gray_mask2 = gray_mask.copy()
            cv2.rectangle(im, *box.rectangle_args)

        # cv2.imshow("gray_img", im)
        # cv2.waitKey()
        return gray_mask


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
        elif self.height > img_shape[0] * 0.84:
            return False
        elif (0.12 < self.width / self.height < 6 and
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
        # cv2.imshow("self.image", self.image)
        self.image_area = im.shape[0] * im.shape[1]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray_otsu = np.zeros(self.gray.shape, dtype=np.uint8)
        self.get_character_edges(self.image)
        self.text = self.recognize_text()

    def get_text(self):
        return self.text

    def get_character_edges(self, image):
        edges3 = self.get_edges(image)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges3, 8, cv2.CV_32S)

        box_wrapper = BoundingBoxWrapper(stats, self.image.shape)
        self.gray_otsu = box_wrapper.get_mask(edges3, self.gray, self.image)
        # cv2.imshow("self.gray_otsu", self.gray_otsu)

    def get_edges(self, image):
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
        # cv2.imshow("edges", edges)
        # cv2.waitKey()
        return edges

    def recognize_text(self):
        config = '-c tessedit_char_whitelist=\ 01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(cv2.bitwise_not(self.gray_otsu), lang='eng', config=config)
        return text

    def get_text_with_median(self):
        config = '-c tessedit_char_whitelist=\ 01234567890ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(cv2.bitwise_not(cv2.medianBlur(self.gray_otsu, 3)), lang='eng',
                                           config=config)
        return text


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


# todo mawdoo3 el inner/ nesting
# todo combine boxes?
# todo get average outside using dilation - real

from googlesearch import search
from goodreads import client
import re
from dotenv import load_dotenv

load_dotenv()
import os


def goodreads_request(text, text2, last_trial=False):
    KEY = os.environ["goodreads_key"]
    gc = client.GoodreadsClient(KEY, "")
    book_json = {'found': False, "text": text}
    try:
        try:
            book = gc.search_books(text)[0]  # slows down performance heavily, limited to 1 request / sec
        except (TypeError, GoodreadsRequestException):
            book = None
        goodreads_url_prefix = "https://www.goodreads.com/book/show/"
        if book is None:
            gen = search(text + " " + goodreads_url_prefix, tld='com', lang='en', num=1,
                         pause=3)
            site = next(gen)
            if goodreads_url_prefix not in site:
                site = next(search(text + " site:" + goodreads_url_prefix, tld='com', lang='en',
                                   num=1, pause=3))
            id = re.search("[0-9]+", site).group()
            book = gc.book(id)
            book_json = {
                'txt': text,
                'txt2': text2,
                'title': book.title,
                'author': book.authors[0].name,
                'average_rating': book.average_rating,
                'url': goodreads_url_prefix + book.gid,
                'summary': book.description,
                'found': True
            }
        # print(book_json)
    except StopIteration:
        if last_trial:
            site = f'not found \nquery="{text}"'
            print(site)
            return {"found": False, "error": "no book found",
                    'txt': 'text2', 'title': ''}
        else:
            return goodreads_request(text2, text, True)
    return book_json


if __name__ == "__main__":
    for root, dirs, files in os.walk('detected_spines/'):
        for file in files:
            cv2.destroyAllWindows()
            img = cv2.imread(os.path.join(root, file))
            segmenter = TextSegmenter(img)
            detected_text = segmenter.get_text()
            detected_text_with_median = segmenter.get_text_with_median()
            goodreads_request(detected_text, detected_text_with_median)


def process_spine(filestr):
    # convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    segmenter = TextSegmenter(img)
    detected_text = segmenter.get_text()
    detected_text_with_median = segmenter.get_text_with_median()
    return goodreads_request(detected_text, detected_text_with_median)


def process_spine_from_extractor(spine):
    segmenter = TextSegmenter(spine)
    detected_text = segmenter.get_text()
    detected_text_with_median = segmenter.get_text_with_median()
    if detected_text == "":
        detected_text = detected_text_with_median
    if detected_text_with_median == detected_text == "":
        return {"found": False, "error": "no text detected", 'txt': '', 'title': ''}
    return goodreads_request(detected_text_with_median, detected_text)
