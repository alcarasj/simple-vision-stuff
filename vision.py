import numpy
import cv2
import sys
from argparse import ArgumentParser

COLOR = cv2.IMREAD_COLOR
GRAYSCALE = cv2.IMREAD_GRAYSCALE
UNCHANGED = cv2.IMREAD_UNCHANGED

parser = ArgumentParser(description="Histogram back-projection.")
parser.add_argument("-i", "--input", dest="input_path",
					help="Input path for the image.")
parser.add_argument("-o", "--output", dest="output_path",
					help="Output path for the resulting image.")
args = parser.parse_args()
input_path = args.input_path
output_path = args.output_path

if not input_path or not output_path:
	print("Specify input and output paths for images as arguments.")
else:
	input_image = cv2.imread(input_path, GRAYSCALE)
	cv2.imshow("image", input_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
