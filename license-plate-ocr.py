import sys
import cv2
import numpy as np
import traceback
from time import time

import darknet.python.darknet as dn

from os.path 				import splitext, basename
from glob					import glob
from darknet.python.darknet import detect
from src.label				import dknet_label_conversion
from src.utils 				import nms


if __name__ == '__main__':

	try:
	
		input_dir  = sys.argv[1]
		output_dir = input_dir

		ocr_threshold = .4

		ocr_weights = 'data/ocr/ocr-net.weights'
		ocr_netcfg  = 'data/ocr/ocr-net.cfg'
		ocr_dataset = 'data/ocr/ocr-net.data'

		ocr_net  = dn.load_net(bytes(ocr_netcfg, encoding='utf-8'), bytes(ocr_weights, encoding='utf-8'), 0)
		ocr_meta = dn.load_meta(bytes(ocr_dataset, encoding='utf-8'))

		imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

		print ('Performing OCR...')

		for i,img_path in enumerate(imgs_paths):

			print ('\tScanning %s' % img_path)

			bname = basename(splitext(img_path)[0])
			
			start = time()

			R,(width,height) = detect(ocr_net, ocr_meta, bytes(img_path, encoding='utf-8') ,thresh=ocr_threshold, nms=None)
			
			total_time = time() - start

			if len(R):

				L = dknet_label_conversion(R,width,height)
				L = nms(L,.45)

				L.sort(key=lambda x: x.tl()[0])
				lp_str = ''.join([chr(l.cl()) for l in L])

				with open('%s/%s_str.txt' % (output_dir,bname),'w') as f:
					f.write(lp_str + '\n')

				print ('\t\tLP: %s\nfound in %0.4f seconds' % (lp_str, total_time))

			else:

				print ('No characters found')

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
