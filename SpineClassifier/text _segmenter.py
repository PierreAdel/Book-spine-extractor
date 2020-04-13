import cv2
import pytesseract
import numpy as np

IMAGE_PATH = 'images/'
NORMAL_TEXT_PATH = IMAGE_PATH + 'normal_text/'
SHELVES_PATH = IMAGE_PATH + 'normal_text/'
SPINES_PATH = IMAGE_PATH + 'spines/'


class TextSegmenter:

    def __init__(self, im):
        self.image = resize_image(im)
        self.image_area = im.shape[0] * im.shape[1]
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.gray_chen = np.zeros(self.gray.shape)
        self.gray_otsu = np.zeros(self.gray.shape)
        self.segmented_img = self.segment_text()
        cv2.imshow('binary_image', self.segmented_img)
        # cv2.waitKey()
        self.text = self.recognize_text()

    def preprocess_image(self, image):
        image = resize_image(image)
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.medianBlur(image, 3)
        # opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # closing = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image

    def get_text(self):
        return self.text

    def segment_text(self):
        """

        :return:
            binarized image
        """
        cv2.imshow('gray', self.gray)
        processed = self.preprocess_image(self.gray)
        cv2.imshow('processed', processed)

        edges = self.get_character_edges(self.image)

        return cv2.threshold(self.gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def get_character_edges(self, image):
        b = image[:, :, 0]
        g = image[:, :, 1]
        r = image[:, :, 2]
        edges = thresh_edges(b) | thresh_edges(r) | thresh_edges(g)
        cv2.imshow('edges', edges)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edges, 8, cv2.CV_32S)
        for index, (left, up, wid, height, area) in enumerate(stats):
            if wid * height > self.image_area * 0.15:
                continue
            elif 0.2 < wid / height < 5 and max(height, wid) > 0.05 * image.shape[0]:

                ROI = labels[up:up + height, left:left + wid]
                edge_mask_of_label = (ROI == index)
                true_count = np.count_nonzero(edge_mask_of_label)
                gray_roi = self.gray[up:up + height, left:left + wid]
                labeled_gray_roi = edge_mask_of_label * gray_roi
                labeled_mean_intensity = np.sum(labeled_gray_roi) / true_count
                # mean_intensity = np.mean(gray_roi)
                # mean_border = np.mean(np.concatenate([gray_roi[0], gray_roi[:, 0]]))
                # invert = False if mean_border < labeled_mean_intensity else True
                # variance = np.var(labeled_gray_roi)
                # standard_deviation = np.sqrt(variance * labeled_gray_roi.size / true_count)
                # if standard_deviation < 30:
                #     continue
                cv2.rectangle(image, (left, up), (left + wid, up + height), (0, 255, 0), 1)
                # cv2.imshow('img', image)
                # cv2.waitKey()
                self.gray_chen[up:up + height, left:left + wid] = cv2.threshold(gray_roi, labeled_mean_intensity, 255,
                                                                                cv2.THRESH_BINARY)[1]

                self.gray_otsu[up:up + height, left:left + wid] = cv2.threshold(gray_roi, 0, 255,
                                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                # if invert:
                #     self.gray_chen[up:up + height, left:left + wid] = \
                #         cv2.bitwise_not(self.gray_chen[up:up + height, left:left + wid])

        cv2.imshow('gray_chen', self.gray_chen)
        cv2.imshow('gray_otsu', self.gray_otsu)
        cv2.imshow('edges_bounded', image)
        cv2.waitKey()
        return edges

    def recognize_text(self):
        """

        :return:
            text included in image
        """
        return "chen *************\n" + pytesseract.image_to_string(self.gray_chen) + \
               "\notsu *************\n" + pytesseract.image_to_string(self.gray_otsu)


def resize_image(im):
    max_width = 1500
    if im.shape[1] > max_width:
        shape_ = (max_width, int(im.shape[0] * max_width / im.shape[1]))
        im = cv2.resize(im, shape_)
    return im


def blur(im):
    return cv2.GaussianBlur((im), (3, 3), 15)


def canny(im):
    return cv2.Canny((im), 60, 190)


def thresh_edges(im):
    return cv2.threshold(canny(im), 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


if __name__ == "__main__":
    for i in range(16):
        cv2.destroyAllWindows()
        img = cv2.imread(SPINES_PATH + str(i) + '.jpg')
        segmenter = TextSegmenter(img)
        print(segmenter.get_text())

# todo fix bounding boxes
#   todo mawdoo3 el inner/outer_box
# todo fix inverting the colors if necessary
