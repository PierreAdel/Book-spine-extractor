import cv2
import numpy as np
import statistics
import imutils
import math
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
        mask = np.zeros(gray.shape)
        mask2 = np.zeros(self.image.shape)

        filtered_lines = []
        fld = cv2.ximgproc.createFastLineDetector()
        lines = fld.detect(gray)
        mask = fld.drawSegments(mask, lines)
        cv2.imshow("LSD", mask)

        for i in range(len(lines)):
            for x1, y1, x2, y2 in lines[i]:
                line = [x1, y1, x2, y2, (x1 + x2) / 2,
                        abs(np.sin(np.arctan((y2 - y1) / max((x2 - x1), 0.01))))]
                if line[-1] > 0.96 and (y2 - y1) > 0.04 * self.image.shape[0]:  # a slant 15 deg & length > 0.25 height
                    filtered_lines.append([*line, line[-1] * (y2 - y1)])
                    cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.line(mask2, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.imshow('filter1', self.image)

        filtered_lines.sort(key=lambda line: line[4])
        for index in range(len(filtered_lines)):
            i = 0
            filtered_lines[index][-1] += filtered_lines[index][-1]
            while index + i < len(filtered_lines) and filtered_lines[index + i][4] - filtered_lines[index][4] <= 15:
                filtered_lines[index][-1] += filtered_lines[index + i][-1]
                filtered_lines[index + i][-1] += filtered_lines[index][-1]
                i += 1
            mask2 = cv2.putText(mask2, str(int(filtered_lines[index][-1])),
                                (filtered_lines[index][0], filtered_lines[index][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("strengths", mask2)

        filtered_lines_by_x = []
        last_x = filtered_lines[0][4]
        max_line = filtered_lines[0]
        for line in filtered_lines:
            _, y1, _, y2, avg_x, _, _ = line
            if line[-1] < gray.shape[0] * 1.5 or y2 - y1 < 0.07 * gray.shape[0]:
                continue
            elif avg_x - last_x < 10:
                max_line = line if line[-1] > max_line[-1] else max_line
            else:
                x1, y1, x2, y2, avg_x, sine, strength = max_line
                filtered_lines_by_x.append(max_line)
                cv2.line(mask2, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                max_line = line
            last_x = avg_x

        # if max_line:
        filtered_lines_by_x.append(max_line)
        x1, y1, x2, y2, avg_x, sine, strength = max_line
        cv2.line(mask2, (x1, y1), (x2, y2), (255, 0, 0), 1)
        cv2.line(self.image, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # cv2.waitKey()
        cv2.imshow("strengths_filtered", mask2)
        cv2.imshow('filter1', self.image)
        cv2.waitKey()

def resize_image(im):
    max_height = 800
    if im.shape[0] > max_height:
        shape_ = (int(im.shape[1] * max_height / im.shape[0]), max_height)
        im = cv2.resize(im, shape_)
    return im

def rotate_image(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fld = cv2.ximgproc.createFastLineDetector()
    lines = fld.detect(gray)

    # in this piece of code we am trying to rotate the image so that the books are always as vertical
    # method: by filtering the taller lines which represent the length of a book while discarding the shorter lines which are letters or noise
    # then getting the median of the taller lines and their angle and rotate the image by this angle
    #lengtharr = []
    anglearr = []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            length = int(math.sqrt(abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2))
            # print(length)
            #lengtharr.append(length)
            if length > 250:
                slope = (y2 - y1) / (x2 - x1)
                angle = math.degrees(math.atan(slope))
                anglearr.append(angle)

    ang_median = statistics.median(anglearr)
    #print("The photo is tilted by: " + str(ang_median) + " degrees")
    rotation_angle = ang_median + 90
    image = imutils.rotate(image, rotation_angle)

    #cv2.imshow("rotation", image)
    #cv2.waitKey()
    print("end rotation")
    return image


if __name__ == "__main__":
    for i in range(15):
        cv2.destroyAllWindows()
        img = cv2.imread(SHELVES_PATH + str(i) + '.jpg')
        img = rotate_image(img)
        img = resize_image(img)
        extractor = SpineExtractor(img)
        extractor.extract()
        spines = extractor.get_spines()
