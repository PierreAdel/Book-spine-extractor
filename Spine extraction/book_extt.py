# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2
import statistics

# load the input image and grab the image dimensions
image = cv2.imread("a.jpeg",)
blackimage = cv2.imread("BLACK.png")

disize = (1440,1080)


image = cv2.resize(image,disize)

image = cv2.GaussianBlur(image, (3, 3), 0)




orig = image
width = 32 *45
height = 32 *24
min_confi = 0.1

H, W, C = image.shape



# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (width, height)
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < min_confi:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])
		#rects = cv2.groupRectangles(rects, 0, 0.85)

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)


rectsx1 = []
rectsy1 = []
rectsx2 = []
rectsy2 = []
c=0
# loop over the bounding boxes
for (startX, startY, endX, endY) in boxes:
	# scale the bounding box coordinates based on the respective
	# ratios

	startX = int(startX * rW)
	startY = int(startY * rH) -100
	endX = int(endX * rW)
	endY = int(endY * rH)

	# draw the bounding box on the image

	cv2.rectangle(blackimage, (startX, startY), (endX, endY), (0, 255, 0), 2)



edgesbrown = cv2.Canny(blackimage, 10, 250)
contours, hierarchy = cv2.findContours(edgesbrown,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



xcoordarray = np.zeros([len(contours),2])
count=0
for c in contours:
	area = cv2.contourArea(c)
	x, y, w, h = cv2.boundingRect(c)
	#ratio = w/h
	#print("ratio= " + str(ratio))
	cv2.drawContours(blackimage, c, -1, (160, 30, 70), 3)
	x, y, w, h = cv2.boundingRect(c)
	cv2.rectangle(blackimage, (x, 0), (x + w, 1010), (160, 130, 70), 2)

	xcoordarray[count,0] =x
	xcoordarray[count,1] =x+w

	count=count+1

	center = (x, y)

	#cropped = imgclear[y:y + h, x:x + w]

	#  cv2.imshow("cropped", cropped)
	# cv2.imwrite( str(x)+".png", cropped)
	# cv2.waitKey()

sorted_array = xcoordarray[np.argsort(xcoordarray[:, 0])]
book_gap =[]
book_width =[]

print(sorted_array)
lastx1 = 0
lastx2 =0
drawn =0
for x in range(len(contours)):
	x1 = int(sorted_array[x][0])-5
	x2 = int(sorted_array[x][1])+5
	#print(x1)

	if abs(lastx1 - x1) > 30 :
		cv2.rectangle(image, (x1, 0), (x2, 1010), (78, 0, 20), 2)

		cropped = image[0:1010, x1:x2]
#		cv2.imshow("cropped"+str(drawn), cropped)
#		cv2.waitKey()
		if drawn > 0:
			book_gap.append(x1 - lastx2)
			book_width.append(x2-x1)
		drawn = drawn + 1

	lastx1 = x1
	lastx2 = x2





print(drawn)
print(book_gap)
print(book_width)
book_gap.sort()
book_gapsmean = statistics.mean(book_gap)
book_gapstd = statistics.stdev(book_gap)
book_gapmead = statistics.median(book_gap)
q1 = statistics.median(book_gap[0:3])
q3 = statistics.median(book_gap[6:9])
iqr =q3 - q1
iqr = 1.5* iqr

q3iqr = q3 + iqr
q1iqr = q1 - iqr

print(book_gapmead)
print(book_gapsmean)
print(book_gapstd)

print(q3iqr)
print(q1iqr)

for number in book_gap:
	if q3iqr < number:
		print("outlier " + str(number))


#for num in book_gap:
	#if num > book_gapstd + book_gapsmean



cv2.imshow("color", image)
cv2.waitKey()
