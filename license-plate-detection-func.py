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

from tempfile import NamedTemporaryFile


def adjust_pts(pts, lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

# [ST210917] move out of function defintion and then load the model only once
wpod_net = load_model("data/lp-detector/wpod-net_update1.h5")

def lp_detect(image_paths_list):	
#S    input_dir  = sys.argv[1] ## [ST210916] replaced by lp_detect's argument
#S    output_dir = input_dir   ## [ST210917] no need to save to disk now

    lp_threshold = .5

#S    wpod_net_path = sys.argv[2] ## [ST210916] replaced by lp_detect's argument
#S    wpod_net = load_model(wpod_net_path)

#S    imgs_paths = glob('%s/*car.png' % input_dir)
    img_paths = image_paths_list # [ST210917] i.e., Dcars from vehicle-detection

    print ('Searching for license plates using WPOD-NET')
	
    # [ST210917] collect all the detected license plates
    Dlps = []

    for i, img_path in enumerate(img_paths):

        print ('\t Processing %s' % img_path)

#S        bname = splitext(basename(img_path))[0]
        Ivehicle = cv2.imread(img_path)

        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
        print ("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

        Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2**4, (240,80), lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)

#S            cv2.imwrite('%s/%s_lp.png' % (output_dir,bname), Ilp*255.)
#S            writeShapes('%s/%s_lp.txt' % (output_dir,bname), [s])
			
			# [ST210917]
            Ilp_bytes = cv2.imencode('.png', Ilp)[1].tobytes()
            f = NamedTemporaryFile(mode='w+b', suffix='.png')
            f.write(Ilp_bytes*255)
            Dlps.append(f.name)
			tempfile_handlers.append(f)
			#####################

    return Dlps
