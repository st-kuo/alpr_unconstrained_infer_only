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
################################################################

# [ST210921]
from src.drawing_utils			import draw_label, draw_losangle, write2img
from src.label 					import lread, Label, readShapes # [ST210921] lread and Label can be omitted
from pdb import set_trace as pause
################################################################


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	try:

		# [ST210919] argv[1] is to hold the images with cars
		#            argv[2] is the directory to place the output images in all stages
		#            argv[3] is the csv file name which to record all the images and 
		#                    their license plate numbers
		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]
		csv_file = sys.argv[3]

		# [ST210919] set up the detection thresholds for 3 stages
		vehicle_threshold = .2
		lp_threshold = .5
		ocr_threshold = .4

		# [ST210919] set up the vehicle model architecture and weights
		vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		vehicle_dataset = 'data/vehicle-detector/voc.data'

		print("[INFO] Loading vehicle detection model...")
		vehicle_net  = dn.load_net(bytes(vehicle_netcfg, encoding='utf-8'), bytes(vehicle_weights, encoding='utf-8'), 0)
		vehicle_meta = dn.load_meta(bytes(vehicle_dataset, encoding='utf-8'))
		############################################

		# [ST210919] set up the license plate detection model artchitecture and weights
		print("[INFO] Loading license plate detection model...")
		wpod_net_path = 'data/lp-detector/wpod-net_update1.h5' 
		wpod_net = load_model(wpod_net_path)
		############################################
	
		# [ST210919] set up the ocr model architecture and weights
		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		print("[INFO] Loading ocr model...")
		ocr_net  = dn.load_net(bytes(ocr_netcfg, encoding='utf-8'), bytes(ocr_weights, encoding='utf-8'), 0)
		ocr_meta = dn.load_meta(bytes(ocr_dataset, encoding='utf-8'))
		############################################


		# [ST210919] Start vehicle detection             
		img_paths = image_files_from_folder(input_dir)
		img_paths.sort()

		if not isdir(output_dir):
		    makedirs(output_dir)

		print ('[INFO] Searching for vehicles using YOLO...')

		for i, img_path in enumerate(img_paths):

		    print ('\tScanning %s' % img_path)

		    # [ST210918] hold all the tempfile handlers and then evoke each .close() to release
		    #            the file in the memory when the current image process is finished.
		    tempfile_handlers = []

		    # [ST210917] This original filename will be used to indicate each image in the csv file
		    img_filename = basename(splitext(img_path)[0])

		    # [ST210917] This is the successful pattern to use tempfile to store and
		    #            retrive image.
#S		    image_st = cv2.imread(img_path)
#S		    image_bytes = cv2.imencode('.png', image_st)[1].tobytes()
#S
#S		    f = NamedTemporaryFile(mode='w+b', suffix='.png')
#S		    f.write(image_bytes)

#S		    image_st_ = cv2.imread(f.name)
#            print("difference between disk version and tempfile version: ", image_st - image_st_)
		    ##################################

		    starttime = time()

		    R, _ = detect(vehicle_net, vehicle_meta, bytes(img_path, encoding='utf-8'), thresh=vehicle_threshold)
#			R, _ = detect(vehicle_net, vehicle_meta, bytes(f.name, encoding='utf-8'), thresh=vehicle_threshold)

#S		    f.close()

		    car_detection_time = time() - starttime

		    R = [r for r in R if r[0] in [b'car',b'bus',b'motorbike']] # [ST210914] add motorbike for motorcycle detection

		    print('\t\t%d cars found in %0.4f seconds' % (len(R), car_detection_time))

		    # [ST210917] All the following license plate detection and OCR will be put into
		    #            the if-block of "if len(R):" 
		    if len(R):

		        Iorig = cv2.imread(img_path) # [ST210919] original image for the final annotation
		        WH = np.array(Iorig.shape[1::-1], dtype=float)

		        Lcars = [] # [ST210919] store the car class labels and car bboxes (Line 152) predicted by YOLO
		        Dcars = [] # [ST210917] collect all the tempfile paths of crops of detected cars

		        for i, r in enumerate(R):

		            cx, cy, w, h = (np.array(r[2])/np.concatenate((WH, WH))).tolist()
		            tl = np.array([cx - w/2., cy - h/2.]) # [ST210919] tl = top left
		            br = np.array([cx + w/2., cy + h/2.]) # [ST210919] br = bottom right
		            label = Label(0, tl, br)
		            Dcar = crop_region(Iorig, label)

		            # [ST210917] save the croped image to tempfile
		            Dcar_bytes = cv2.imencode('.png', Dcar)[1].tobytes()
		            f = NamedTemporaryFile(mode='w+b', suffix='.png')
		            f.write(Dcar_bytes)
		            Dcars.append(f.name)
		            tempfile_handlers.append(f)
		            ##########################

		            Lcars.append(label)

#                    cv2.imwrite('%s/%s_%dcar.png' % (output_dir,bname,i),Icar)

#                lwrite('%s/%s_cars.txt' % (output_dir,bname),Lcars)
		#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


		        # [ST210919] Start license plate detection and ocr in each croped car image
		        #S imgs_paths = glob('%s/*car.png' % input_dir)
		        img_paths = Dcars # [ST210919] all the croped car images are in tempfile

		        print('[INFO] Searching for license plates using WPOD-NET...')

		        # [ST210917] Dlps is to hold all the copred images of detected license plates
		        Dlps = []
		        lp_bboxes = [] # [ST210921] hold the license plate bboxes in a image
		        lp_texts = [] # [ST210921] hold the license plate numbers in a image

		        for i, img_path in enumerate(img_paths):

		            print('\t Processing %s' % img_path)

		            bname = splitext(basename(img_path))[0] # [ST210919] remove the .png
		            Ivehicle = cv2.imread(img_path)

		            ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
		            side  = int(ratio*288.)
		            bound_dim = min(side + (side%(2**4)), 608)
		            print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

		            Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240, 80), lp_threshold)

		            if len(LlpImgs):
		                Ilp = LlpImgs[0]
		                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
		                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

		                s = Shape(Llp[0].pts)
		                lp_bboxes.append(s) # [ST210921] see Line 179

#S		                cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
#S		                writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

		                # [ST210917]
		                Ilp_bytes = cv2.imencode('.png', Ilp*255)[1].tobytes()
		                f = NamedTemporaryFile(mode='w+b', suffix='.png')
		                f.write(Ilp_bytes)

		                R, (width, height) = detect(ocr_net, ocr_meta, bytes(f.name, encoding='utf-8') ,thresh=ocr_threshold, nms=None)

		                if len(R):

		                    L = dknet_label_conversion(R, width, height)
		                    L = nms(L, .45)

		                    L.sort(key=lambda x: x.tl()[0])
		                    lp_str = ''.join([chr(l.cl()) for l in L])

		                    lp_recog_time = time() - starttime

		                    lp_texts.append(lp_str) # [ST210921] for annotation later
#S		                    with open('%s/%s_str.txt' % (output_dir, img_filename),'w') as f: # [ST210917] change "bname" to "img_filename"
#S		                        f.write(lp_str + '\n')

		                    print ('\t\tLP: %s\nfound in %0.4f seconds' % (lp_str, lp_recog_time))

		                else:

		                    print ('No characters found')
		                    lp_texts.append(None) # [ST210921] License plate is detected, but no number is detected
		            else:

		                lp_bboxes.append(None) # [ST210921] see Line 201
		                lp_texts.append(None) # [ST210921] Car is detected but no license plate detected
						
#S		                Dlps.append(f.name)
#S		                tempfile_handlers.append(f)
		                #####################

		        # [ST210921] Start annotating the output image/frame
		        YELLOW = (0, 255, 255)
		        RED    = (0, 0, 255)

		        if Lcars: # [ST210921] Checking if any cars detected

		            for i, lcar in enumerate(Lcars):
		                # [ST210921] draw box for cars detected
		                draw_label(Iorig, lcar, color=YELLOW, thickness=3) 

		                if lp_bboxes[i]: # [ST210921] Checking if any license plates detected
		                    pts = lp_bboxes[i].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
		                    ptspx = pts*np.array(Iorig.shape[1::-1], dtype=float).reshape(2,1)
		                    draw_losangle(Iorig, ptspx, RED, 3)

		                    llp = Label(0,tl=pts.min(1),br=pts.max(1))
#S		                    write2img(Iorig, llp, lp_str)
		                    write2img(Iorig, llp, lp_texts[i])

		        cv2.imwrite('%s/%s_output.png' % (output_dir, img_filename), Iorig)			

		        ##################################################################
		        '''
		        # [ST210919] Starting OCR for each license plate detected
		        ocr_threshold = .4

		        img_paths = Dlps # [ST210917] i.e., Dlps from license plate detection

		        print ('Performing OCR...')

		        for i, img_path in enumerate(img_paths):

		            print ('\tScanning %s' % img_path)
		            cv2_imshow(img_path)

#S		            bname = basename(splitext(img_path)[0])

		            start = time()

		            R, (width, height) = detect(ocr_net, ocr_meta, bytes(img_path, encoding='utf-8') ,thresh=ocr_threshold, nms=None)

		            total_time = time() - start

		            if len(R):

		                L = dknet_label_conversion(R, width, height)
		                L = nms(L, .45)

		                L.sort(key=lambda x: x.tl()[0])
		                lp_str = ''.join([chr(l.cl()) for l in L])

		                with open('%s/%s_str.txt' % (output_dir, img_filename),'w') as f: # [ST210917] change "bname" to "img_filename"
		                    f.write(lp_str + '\n')

		                print ('\t\tLP: %s\nfound in %0.4f seconds' % (lp_str, total_time))

		            else:

		                print ('No characters found')

		        '''
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)

