import sys
sys.path.append("/home/fereshteh/caffe-fast-rcnn/python")
import caffe
import os
prototxt = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.ROOT_DIR, '/home/fereshteh/faster', 'VGG16_faster_rcnn_final_kaist_fusion_2.caffemodel')
net = caffe.Net(prototxt, caffemodel, caffe.TEST)