import sys, os
import keras
import cv2
import traceback

from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes

# [ST210919] for vehicle detection
import numpy as np
from time import time

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.colab.patches import cv2_imshow
import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
from darknet.python.darknet import detect

from tempfile import NamedTemporaryFile
################################################################

# [ST210919] for OCR
#S from os.path 				import splitext, basename
from src.label				import dknet_label_conversion
from src.utils 				import nms
##############################################################

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	try:

		# [ST210919] for vehicle detection
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2] # [ST210917] This will be passed into OCR def-function
		csv_file = sys.argv[3]

#    lp_model_path = 'data/lp-detector/wpod-net_update1.h5' # [ST210918] This is for licenseplatedetectionfunc

		vehicle_threshold = .2

		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		print("[INFO] Load vehicle detection model...")
		vehicle_net  = dn.load_net(bytes(vehicle_netcfg, encoding='utf-8'), bytes(vehicle_weights, encoding='utf-8'), 0)
		vehicle_meta = dn.load_meta(bytes(vehicle_dataset, encoding='utf-8'))
		############################################
		
        # [ST210918] for ocr
        ocr_weights = 'data/ocr/ocr-net.weights'
        ocr_netcfg  = 'data/ocr/ocr-net.cfg'
        ocr_dataset = 'data/ocr/ocr-net.data'

        print("[INFO] Loading ocr model...")
        ocr_net  = dn.load_net(bytes(ocr_netcfg, encoding='utf-8'), bytes(ocr_weights, encoding='utf-8'), 0)
        ocr_meta = dn.load_meta(bytes(ocr_dataset, encoding='utf-8'))
        ############################################
                    
        imgs_paths = image_files_from_folder(input_dir)
        imgs_paths.sort()

		
#S		input_dir  = sys.argv[1]
#S		output_dir = input_dir

		lp_threshold = .5

		print("[INFO] Load license plate detection model...")
		wpod_net_path = 'data/lp-detector/wpod-net_update1.h5' # [ST210919] change from [2] t0 [4] for matching cmd line arguments 
		wpod_net = load_model(wpod_net_path)

        # [ST210919] Add vehicle detection code












		imgs_paths = glob('%s/*car.png' % input_dir)

		print ('Searching for license plates using WPOD-NET')

		for i,img_path in enumerate(imgs_paths):

			print ('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)

			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print ("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

			Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)

