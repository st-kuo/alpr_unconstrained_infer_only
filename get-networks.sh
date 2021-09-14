#!/bin/bash

set -e

# To only download necessary files and avoid redundant actions
fileid="1MkfoWE4WFbY3MENX4arJA5kuYACrLSC6"
filename="yolov3.weights"



#mkdir data/lp-detector -p
#mkdir data/ocr -p
#mkdir data/vehicle-detector -p

#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/lp-detector/wpod-net_update1.h5   -P data/lp-detector/
#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/lp-detector/wpod-net_update1.json -P data/lp-detector/

#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.cfg     -P data/ocr/
#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.names   -P data/ocr/
#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.weights -P data/ocr/
#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/ocr/ocr-net.data    -P data/ocr/

#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/yolo-voc.cfg     -P data/vehicle-detector/
#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/voc.data         -P data/vehicle-detector/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/yolo-voc.weights -P data/vehicle-detector/
#wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/vehicle-detector/voc.names        -P data/vehicle-detector/


# copy from https://stackoverflow.com/a/49444877
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o data/vehicle-detector/${filename}
