
#Base on source code from the book Practical Python and OpenCV
# import the necessary packages
from __future__ import print_function
from pyimagesearch.rgbhistogram import RGBHistogram
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import argparse
import glob
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-t", "--targets", required = True,
	help = "Path to the target image ")
ap.add_argument("-i", "--images", required = True,
	help = "path to the image dataset")
ap.add_argument("-m", "--masks", required = True,
	help = "path to the image masks")

args = vars(ap.parse_args())


# grab the image and mask paths
targetPaths = sorted(glob.glob(args["targets"] + "/*.jpg"))
imagePaths = sorted(glob.glob(args["images"] + "/*.jpg"))
maskPaths = sorted(glob.glob(args["masks"] + "/*.png"))


# initialize the list of data and class label targets
data = []
target = []

# initialize the image descriptor
desc = RGBHistogram([8, 8, 8])

# loop over the image and mask paths
for (imagePath, maskPath) in zip(imagePaths, maskPaths):
	# load the image and mask
	image = cv2.imread(imagePath)
	mask = cv2.imread(maskPath)
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

	# describe the image
	features = desc.describe(image, mask)

	# update the list of data and targets
	data.append(features)
	target.append(imagePath.split("_")[-2])

# grab the unique target names and encode the labels
targetNames = np.unique(target)
le = LabelEncoder()
target = le.fit_transform(target)

# construct the training and testing splits
(trainData, testData, trainTarget, testTarget) = train_test_split(data, target,
	test_size = 0.3, random_state = 42)

# train the classifier
model = RandomForestClassifier(n_estimators = 25, random_state = 84)
model.fit(trainData, trainTarget)

# evaluate the classifier
print(classification_report(testTarget, model.predict(testData),
	target_names = targetNames))


for i in np.arange(0, len(targetPaths)):

	targetPath = targetPaths[i]
	targetImage = cv2.imread(targetPath)

	mask = np.zeros(targetImage.shape[:2], dtype = "uint8")
	(cX, cY) = (targetImage.shape[1] // 2, targetImage.shape[0] // 2)
	r = int(round(cX/3))
	cv2.circle(mask, (cX, cY), r, 255, -1)
	features = desc.describe(targetImage, mask)



	# predict what type of flower the image is
	flower = le.inverse_transform(model.predict(features))[0]
	print(targetPath)
	print("This flower is a Motherfucking {}".format(flower.upper()))
	small = cv2.resize(targetImage, (0,0), fx=0.2,fy=0.2)
	cv2.imshow("image", small)
	cv2.waitKey(0)
