import sys
import cv2
import numpy as np
import traceback
import os
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

from licenseplatedetectionfunc import lp_detect
from licenseplateocrfunc import lp_ocr
from tempfile import NamedTemporaryFile

# [ST210918] for license plate detection module
from src.utils 				import im2single
from src.keras_utils 		import load_model, detect_lp
from src.label 				import Shape, writeShapes
from glob 					import glob
from os.path 				import splitext, basename
import keras
############################################

# [ST210918] for ocr
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms
############################################

if __name__ == '__main__':

    try:
	
        input_dir  = sys.argv[1]
        output_dir = sys.argv[2] # [ST210917] This will be passed into OCR def-function
        csv_file = sys.argv[3]


#        lp_model_path = 'data/lp-detector/wpod-net_update1.h5' # [ST210918] This is for licenseplatedetectionfunc

        vehicle_threshold = .2

        vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
        vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
        vehicle_dataset = 'data/vehicle-detector/voc.data'
        
        vehicle_net  = dn.load_net(bytes(vehicle_netcfg, encoding='utf-8'), bytes(vehicle_weights, encoding='utf-8'), 0)
        vehicle_meta = dn.load_meta(bytes(vehicle_dataset, encoding='utf-8'))

        # [ST210918] for license plate detection
        print("[INFO] Loading license plate detection model...")
        wpod_net_path = sys.argv[4] 
        wpod_net = load_model(wpod_net_path)        
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

        if not isdir(output_dir):
            makedirs(output_dir)

        print ('Searching for vehicles using YOLO...')

        for i, img_path in enumerate(imgs_paths):

            print ('\tScanning %s' % img_path)
            
            # [ST210918] hold all the tempfile handlers and then evoke each .close() to release
		    #            the file in the memory when the current image process is finished.
            tempfile_handlers = []

            img_filename = basename(splitext(img_path)[0]) # [ST210917] This will be passed into OCR def-function

            # [ST210917] This is the successful pattern to use tempfile to store and
            #            retrive image.
            image_st = cv2.imread(img_path)
            image_bytes = cv2.imencode('.png', image_st)[1].tobytes()

            f = NamedTemporaryFile(mode='w+b', suffix='.png')
            f.write(image_bytes)

            image_st_ = cv2.imread(f.name)
#            print("difference between disk version and tempfile version: ", image_st - image_st_)
            ##################################

            starttime = time()

            R, _ = detect(vehicle_net, vehicle_meta, bytes(img_path, encoding='utf-8'), thresh=vehicle_threshold)
#			R, _ = detect(vehicle_net, vehicle_meta, bytes(f.name, encoding='utf-8'), thresh=vehicle_threshold)
			
            f.close()
			
            total_time = time() - starttime

            R = [r for r in R if r[0] in [b'car',b'bus',b'motorbike']] # [ST210914] add motorbike for motorcycle detection

            print ('\t\t%d cars found in %0.4f seconds' % (len(R), total_time))

            # [ST210917] All the following license plate detection and OCR will be put into
            #            the if-block of "if len(R):" 
            if len(R):

                Iorig = cv2.imread(img_path)
                WH = np.array(Iorig.shape[1::-1],dtype=float)
                Lcars = []
				
                Dcars = [] # [ST210917] collect all the file paths of crops of detected cars

                for i,r in enumerate(R):

                    cx, cy, w, h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                    tl = np.array([cx - w/2., cy - h/2.])
                    br = np.array([cx + w/2., cy + h/2.])
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
				
                # [ST210918] Add the portion of license plate detection and OCR
                Detected_lps = lp_detect(Dcars, wpod_net)
                lp_ocr(Detected_lps, output_dir, img_filename, ocr_net, ocr_meta)
                
                for f in tempfile_handlers: # [ST210918] release tempfile handlers
                    f.close()

    except:
        traceback.print_exc()
        sys.exit(1)

sys.exit(0)
	
