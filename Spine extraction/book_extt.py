import cv2
import numpy as np
import math


img = cv2.imread('a (2).jpg', cv2.IMREAD_COLOR)
imgclear = img
black = cv2.imread('BLACK.png', cv2.IMREAD_COLOR)

b1 = cv2.imread('BLACK.png', cv2.IMREAD_COLOR)

imgsmooth = cv2.GaussianBlur(img, (5, 5), 0)

hsv = cv2.cvtColor(imgsmooth, cv2.COLOR_BGR2HSV)


def filter_by_color(black):

    if black == True:
        image = b1
    else:
        image = img

    lower_red = np.array([160, 30, 70])
    upper_red = np.array([190, 255, 255])
    maskred = cv2.inRange(hsv, lower_red, upper_red)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    maskgreen = cv2.inRange(hsv, lower_green, upper_green)

    lower_blue = np.array([101, 50, 38])
    upper_blue = np.array([130, 255, 255])
    maskblue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    maskyellow = cv2.inRange(hsv, lower_yellow, upper_yellow)


    lower_brown = np.array([10,100,200])
    upper_brown = np.array([20, 255, 200])
    maskbrown = cv2.inRange(hsv, lower_brown, upper_brown)



    resultred = cv2.bitwise_and(img, img, mask=maskred)
    resultgreen = cv2.bitwise_and(img, img, mask=maskgreen)
    resultblue = cv2.bitwise_and(img, img, mask=maskblue)
    resultyellow = cv2.bitwise_and(img, img, mask=maskyellow)
    resultbrown = cv2.bitwise_and(img, img, mask=maskbrown)


    edgesred = cv2.Canny(resultred, 10, 250)
    kernelred = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedred = cv2.morphologyEx(edgesred, cv2.MORPH_CLOSE, kernelred)
    contoursred, hierarchy = cv2.findContours(closedred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edgesgreen = cv2.Canny(resultgreen, 10, 250)
    kernelgreen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedgreen = cv2.morphologyEx(edgesgreen, cv2.MORPH_CLOSE, kernelgreen)
    contoursgreen, hierarchy = cv2.findContours(closedgreen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edgesblue = cv2.Canny(resultblue, 10, 250)
    kernelblue = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedblue = cv2.morphologyEx(edgesblue, cv2.MORPH_CLOSE, kernelblue)
    contoursblue, hierarchy = cv2.findContours(closedblue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edgesyellow = cv2.Canny(resultyellow, 10, 250)
    kernelyellow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedyellow = cv2.morphologyEx(edgesyellow, cv2.MORPH_CLOSE, kernelyellow)
    contoursyellow, hierarchy = cv2.findContours(closedyellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    edgesbrown = cv2.Canny(resultbrown, 10, 250)
    kernelbrown = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closedbrown = cv2.morphologyEx(edgesbrown, cv2.MORPH_CLOSE, kernelbrown)
    contoursbrown, hierarchy = cv2.findContours(closedbrown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contourshere = [contoursred ,contoursgreen, contoursblue, contoursyellow, contoursbrown]

    dims = img.shape
    print("width", dims)

    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    for contourCOLOR in contourshere:
        for c in contourCOLOR:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)

            ratio = w / h
            print("ratio= "+str(ratio))

            if area >= 3000 and ratio > 0.01 and ratio < 0.4:

                cv2.drawContours(image, c, -1, (160, 30, 70), 3)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y-200), (x + w, y + h), (160, 130, 70), 4)

                center = (x, y)

                cropped = imgclear[y:y + h, x:x + w]


              #  cv2.imshow("cropped", cropped)
               # cv2.imwrite( str(x)+".png", cropped)
               # cv2.waitKey()

    cv2.imshow("color",image)
    cv2.waitKey()








# Convert the image to gray-scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 10, 250)
cv2.imshow("canny", gray)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

total = 0


for c in contours:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.1 * peri, True)

    # if the approximated contour has four points, then assume that the
    # contour is a book -- a book is a rectangle and thus has four vertices
    if len(approx) == 4:
        #cv2.drawContours(black, [approx], -1, (0, 255, 0), 4)
        total += 1
#print(total)



#cv2.imshow('Canny Edges After Contouring', closed)
#cv2.waitKey(0)

print("Number of Contours found = " + str(len(contours)))

# Draw all contours
# -1 signifies drawing all contours
cv2.drawContours(black, contours, -1, (0, 255, 0), 3)


#print(contours)
#cv2.imshow('Contours', black)



def filter_by_Hough():
    lines = cv2.HoughLinesP(edges,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = 100,maxLineGap = 10)
    linecoords, count = fill_array(lines)


#    linecoords = get_num_books(linecoords,count)



    c = 0
    for line in lines:
        x1 = linecoords[c][0]
        y1 = linecoords[c][1]
        x2 = linecoords[c][2]
        y2 = linecoords[c][3]
        c = c + 1
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    sorted_array = linecoords[np.argsort(linecoords[:, 0])]
    print("sorted")
    print(sorted_array)
    cv2.imshow("Result Image", img)

    cv2.waitKey()


# fills linecoord array with line coords and gradient
def fill_array(lines):
    count = 0
    for line in lines:  # count the number of lines detected
        count = count + 1
    linecoord = np.zeros((count, 5), dtype=np.int32)
    row = 0
    for line in lines:

        x1, y1, x2, y2 = line[0]

        if abs(x2 - x1) > 0.01:
            line_grad = (y2 - y1) / (x2 - x1)
        else:
            line_grad = 2147483646  # max 32-bit int


        linecoord[row][0] = x1
        linecoord[row][1] = y1
        linecoord[row][2] = x2
        linecoord[row][3] = y2
        linecoord[row][4] = line_grad
        row = row + 1
    return linecoord, count

#filter_by_Hough()
#filter_by_color(black=False)







"""""
def get_num_books(lines,length):
    for line in lines:
        print(line+100)
        if False == True:
            line[0] = 0
            line[1] = 0
            line[2] = 0
            line[3] = 0
            line[4] = 0
    return lines
"""""








"""""


def get_lines_sorted_by_n(lines,n):  # return a sorted list of circle center coordiantes sorted by n
    if lines is not None:
        return sorted([center[n] for center in lines])

def remove_very_close_parralel_line(lines,count): # if lines (distance betwwn them and gradient is simmilar) less than threshold remove one of them
    for i in range(count):
        j=i+1
        while j < count:
            if abs(lines[i][0]-lines [j][0]) < 10 and abs(lines[i][4])-abs(lines [j][4]) < 30 :
                lines[j][0] =0
                lines[j][2] =0
            j = j + 1
    return lines

def remove_very_apart_lines(lines): ## useless lines that appear in the corners
    return lines


def Detect_book_lines(lines): # if close lines gradients are not close then they are not a book
    return lines

def remove_short_lines(lines):
    length_threshold = 0;
    for line in lines:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]
        line_length = math.sqrt((x2-x1)**2 + (y2-y1)**2)

      #  if line_length < length_threshold:  # remove line





"""