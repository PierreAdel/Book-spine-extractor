import cv2
import numpy as np
SHELVES_PATH = 'images/shelves/'


class SpineExtractor:

    def __init__(self, image):
        self.image = image

    def get_spines(self):
        """

        :return:
            list of spines
        """
        return 0

    def extract(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 70, 210, apertureSize=3)
        cv2.imshow('edges1', edges)
        kernel = np.ones((1, 2), np.uint8)

        # edges = cv2.dilate(edges, kernel, iterations=1)
        # edges = cv2.GaussianBlur(dilate, (3, 3), 3)

        # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
        cv2.imshow('edges2', edges)
        # cv2.waitKey()
        # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=150,
                                minLineLength=int(self.image.shape[0] / 4), maxLineGap=20)
        my_lines = []
        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                line = (x1, y1, x2, y2, (x1 + x2) / 2,
                        abs(np.sin(np.arctan((y2 - y1) / max((x2 - x1), 0.01)))))
                if line[-1] > 0.96 :  # a slant 15 deg & length > 0.25 height
                    my_lines.append(line)
                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.imshow('dsads', self.image)
        cv2.waitKey()
        # # np.rad2deg(np.arctan((y2 - y1) / (x2 - x1)))  # angle

        filtered_lines_by_x = []
        close_lines_set = []
        my_lines.sort(key=lambda line: line[4])  # sort by avg x
        last_x = my_lines[0][4]
        for line in my_lines:
            x1, y1, x2, y2, avg_x, sine = line
            if avg_x - last_x < 10 or not close_lines_set:
                close_lines_set.append(line)
            else:
                close_lines_set.sort(key=lambda line: (line[3] - line[1]))  # sort by length
                x1, y1, x2, y2, avg_x, sine = close_lines_set[0]
                filtered_lines_by_x.append((*close_lines_set[0], sine * (y2 - y1)))
                cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                last_x = avg_x
                close_lines_set = []
            # cv2.waitKey()
        cv2.imshow('dsads', self.image)
        cv2.waitKey()

        # my_lines.sort(key=lambda line: (line[3] - line[1]))  # sort by length
        for line in filtered_lines_by_x:
            x1, y1, x2, y2, avg_x, sine, strength = line
            # if y1 - y2 < self.image.shape[0] / 3:
            #     continue
            cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # cv2.waitKey()
            cv2.imshow('dsads', self.image)
        cv2.waitKey()

    # def extract(self):
    #     gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #     edges = cv2.Canny(gray, 50, 200, apertureSize=3)
    #     # cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
    #     cv2.imshow('edges', edges)
    #     # cv2.waitKey()
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    #     lines = cv2.HoughLinesP(edges, rho=1, theta=1 * np.pi / 180, threshold=100,
    #                             minLineLength=100, maxLineGap=20)
    #
    #     for i in range(len(lines)):
    #         for x1, y1, x2, y2 in lines[i]:
    #             cv2.line(self.image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         cv2.imshow('dsads', self.image)
    #     cv2.waitKey()


def resize_image(im):
    max_height = 1000
    if im.shape[0] > max_height:
        shape_ = (int(im.shape[1] * max_height / im.shape[0]), max_height)
        im = cv2.resize(im, shape_)
    return im


if __name__ == "__main__":
    for i in range(5):
        cv2.destroyAllWindows()
        img = cv2.imread(SHELVES_PATH + str(i) + '.jpg')
        img = resize_image(img)
        # cv2.imshow('dsads', img)
        extractor = SpineExtractor(img)
        extractor.extract()
        spines = extractor.get_spines()



