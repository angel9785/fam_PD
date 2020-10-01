#!/usr/bin/env python

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import os
os.environ['GLOG_minloglevel'] = '2'
from multiprocessing import Process, Queue, Value
import matplotlib 
#matplotlib.use('Agg')
import argparse
import pickle
import _init_paths
from fast_rcnn.config import cfg
# from fast_rcnn.test import im_detect
from fast_rcnn.test import im_detect_2in
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sys
sys.path.append("/home/fereshteh/caffe-faster-rcnn1/python")
import caffe
import os
import cv2
import argparse
from copy import copy, deepcopy
from datasets.voc_eval import voc_eval
import xml.dom.minidom as minidom
import time
from imutils.video import FPS

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--train_scale', dest='train_scale', help='which train scale?', default=400,type=int)
    parser.add_argument('--test_scale', dest='test_scale', help='which test scale?', default=800,type=int)
    parser.add_argument('--proposals', dest='num_proposals', help='how many proposals?', default=100,type=int)

    args = parser.parse_args()

    return args


def load_kitti_annotation(data_dir, index):
    """
    Load image and bounding boxes info from txt file in the KITTI format.
    """
    def label_lookup(str): # return the label according to annotations
        return{
            '%': 0,
            'person-fa': 0,
            'person?': 0,
            'people': 0,
            'cylist': 0,
            'Ignored': 0,
            'person': 1
        }[str]

    print (index)
    print (data_dir)
    filename = os.path.join(data_dir, 'annotations','I'+ index + '.txt')
    lines = open(filename).readlines()
    num_inst = len(lines)

    # Exclude ignored samples
    lines_new = list()
    num_ignored = 0
    for line in lines:
        data = line.split()
        if label_lookup(data[0]) == 0:
            num_ignored += 1
        else:
            lines_new.append(line)

    if num_ignored > 0:
        print ('Removed {} igonored objects' \
            .format(num_ignored))

    num_inst -= num_ignored
    boxes = np.zeros((num_inst, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_inst,), dtype=np.int32)

    for ix, line in enumerate(lines_new):
        data = line.split()
        complabel = label_lookup(data[0])

        x1 = float(data[1]) -1
        y1 = float(data[2]) -1
        x2 = float(data[3]) + x1
        y2 = float(data[4]) + y1
        occl = float(data[5])
        vx1 = float(data[6]) -1
        vy1 = float(data[7]) -1
        vx2 = float(data[8]) + vx1
        vy2 = float(data[9]) + vy1
        ignore = float(data[10])

        vrate = (float(data[9])*float(data[8]))/((float(data[3])+cfg.EPS)*(float(data[4])+cfg.EPS))

        # discard samples according to caltech criteria
        if x1 < 5 or y1 < 5 or x2 > 634 or y2 > 474 or (y2-y1) <50 or ignore > 0:
            continue

        if occl > 0 and vrate < 0.65:
            continue

        boxes[ix,:] = [x1,y1,x2,y2]
        gt_classes[ix] = complabel
        
    assert (boxes[:, 2] >= boxes[:, 0]).all()
    assert (boxes[:, 3] >= boxes[:, 1]).all()
    assert (boxes[:, 0] >= 0).all
    assert (boxes[:, 1] >= 0).all

    return {'boxes' : boxes,
            'gt_classes': gt_classes, 
            'flipped' : False}


OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

cfg.TEST.HAS_RPN = True  # Use RPN for proposals
args = parse_args()
print (args)
model_dir = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16')
prototxt = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16', 'faster_rcnn_test.pt')
caffemodel = os.path.join(model_dir, 'VGG16_faster_rcnn_final_kaist_fusion.caffemodel')

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))


    

CLASSES = ('__background__','person')
 


print ('\n\nLoaded network {:s}'.format(caffemodel))
print (prototxt)
cfg.GPU_ID = args.gpu_id    
cfg.TEST.SCALES = (args.train_scale,)
cfg.TEST.RPN_POST_NMS_TOP_N = args.num_proposals
cfg.TEST.MAX_SIZE = args.test_scale

# initialize OpenCV's special multi-object tracker

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
help="OpenCV object tracker type")
args = vars(ap.parse_args())

def detect(qf,qex,qdet,qtrack,qbox,qnum,f,isInited):
    
    caffe.set_mode_gpu()
    caffe.set_device(0)
    # f=1 
    # a=f
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    data_dir = os.path.join(cfg.ROOT_DIR, 'data/demo_pedestrian/set07_V000')
    CLASSES = ('__background__','person')
    isInited.value=True
    while True:
        im=qdet.get()
        if (im=='done'):
            break
        im = format(im, '05d')
        qex.put(im)


        # print(im)

        im1_file = os.path.join(data_dir,'color', 'I'+ im + '.jpg')
        im2_file = os.path.join(data_dir,'thermal', 'I'+ im + '.jpg')
        # print (im1_file)
        # print (im2_file)
        im1 = cv2.imread(im1_file)
        im2 = cv2.imread(im2_file)
        timer = Timer()
        timer.tic()

        scores, boxes = im_detect_2in(net, im1, im2)
        #f.value = False
        #fps.update()    
        timer.toc()
        # print("1111111111111111111111111111111111111111111")
        # print (f'Detection took {timer.total_time}s')
        NMS_THRESH = 0.3    
        all_dets = None
        # cls_boxes = boxes[:, 4:4 * (cls_ind + 1)]
        box=[]
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]

            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            cls_inds = np.ones((len(keep),)) * cls_ind
            inds = np.where(dets[:,-1]>0.5)[0]
            for i in inds:
               bbox = dets[i, :4]
               score = dets[i, -1]
               if(bbox[0]<0):
                   bbox[0]=0
               if(bbox[1]<0):
                   bbox[1]=0
               bb=(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
               box.append(bb)

               # ax.add_patch(
               #     plt.Rectangle((bbox[0], bbox[1]),
               #                 bbox[2] - bbox[0],
               #                 bbox[3] - bbox[1], fill=False,
               #                 edgecolor=cls_color[2], linewidth=3.5)
               #     )
               # ax.text(bbox[0], bbox[1] - 2,
               #         '{:s} {:.3f}'.format(cls, score),
               #         bbox=dict(facecolor='blue', alpha=0.5),
               #         fontsize=14, color='white')

               # qnum.put(inds)
        # if (len(box)>0):
        qbox.put(box)
        f.value = False

        #print("Detet: ",str(qnum.qsize()))
        # time.sleep(5)
        #print(1111)
        # qf.get()
        # print("f235556 is: ", str(qf.qsize()))
    print('out o det')
    qtrack.put('done')
    #plt.axis('off')
    #plt.tight_layout()
    #plt.draw()    

def track(qstop,qex,f,qtrack,qbox,qnum):
    data_dir = os.path.join(cfg.ROOT_DIR, 'data/demo_pedestrian/set07_V000')
    trackers = cv2.MultiTracker_create()
    # firstDetect=False
    # while(qbox.empty()):
    #     pass
    im1 = qtrack.get()
    imname=format(im1, '05d')

    imStop=0
    fps=FPS().start()
    while True:
        if (im1 < imStop ):
            im1 = qtrack.get()
            if (im1=='done'):
                break
            imname = format(im1, '05d')
            # print("in trac ",im)
            # imname=im

            im= os.path.join(data_dir,'color', 'I'+ imname + '.jpg')
            im = cv2.imread(im)
            fps.update()
            #fps.update()
            #print("while track", str(f.value),str(qnum.qsize()))
            # if(qbox.qsize()>0):
            #     firstDetect=True
            #     bboxes=qbox.get()
            #     imtst=qex.get()
            #
            #     print("fereshte ",imname,imtst)
            #     # bboxes=(bboxes).astype(np.int32)
            #     n=len(bboxes)
            #     # print("eeee",str(n))
            #     trackers = cv2.MultiTracker_create()
            #     for i in range (0,n):
            #         # bbox=qbox.get()
            #         # qnum.get()
            #         # print("2222222222222222222222222222222222")
            #         # bb=(bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1])
            #         tracker = OPENCV_OBJECT_TRACKERS["csrt"]()
            #         bb= np.hstack(bboxes[i]).astype(np.int32)
            #         bb1=(bb[0],bb[1],bb[2],bb[3])
            #         trackers.add(tracker, im,bb1)
            #         # for box in bb:
            #         (x, y, w, h) = [int(v) for v in bb]
            #         cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     cv2.imshow('image', im)
            #
            #     key = cv2.waitKey(1) & 0xFF
            # else:
            (success, boxes) = trackers.update(im)
            # #feri
            # # for box in boxes:
            # #     (x, y, w, h) = [int(v) for v in box]
            # #     cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # # cv2.imshow('image', im)
            # # # feri
            # # key = cv2.waitKey(1) & 0xFF
            ##print("mmm ", imname, str(imStop))

        elif (qbox.qsize() > 0):
            imStop = qstop.get()
            print("imstop is: ", str(imStop),str(qbox.qsize() ))
            # firstDetect = True
            bboxes = qbox.get()

            imtst = qex.get()

            ##print("fereshte ", imname, imtst)
            im = os.path.join(data_dir, 'color', 'I' + imname + '.jpg')
            im = cv2.imread(im)
            fps.update()
            # bboxes=(bboxes).astype(np.int32)
            n = len(bboxes)
            # print("eeee",str(n))
            trackers = cv2.MultiTracker_create()
            for i in range(0, n):
                # bbox=qbox.get()
                # qnum.get()
                # print("2222222222222222222222222222222222")
                # bb=(bboxes[i][0], bboxes[i][1], bboxes[i][2] - bboxes[i][0], bboxes[i][3] - bboxes[i][1])
                tracker = OPENCV_OBJECT_TRACKERS["csrt"]()#"csrt"]()
                bb = np.hstack(bboxes[i]).astype(np.int32)
                bb1 = (bb[0], bb[1], bb[2], bb[3])
                trackers.add(tracker, im, bb1)
                # for box in bb:
            #     (x, y, w, h) = [int(v) for v in bb]
            #     cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imshow('image', im)

        #    key = cv2.waitKey(1) & 0xFF
        # if(firstDetect==False and )
        # if(qstop.empty()==False):
        #     imStop = qstop.get()
        #     print("imstop is: ",str(imStop))


    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    #print("out o track")


qtrack = Queue()
qbox = Queue()
qnum = Queue()
qdet = Queue()
qex= Queue()
qstop=Queue()
qf= Queue()
f = Value('b', False)
isInited = Value('b', False)
d = Process(target=detect, args=(qf,qex,qdet,qtrack,qbox,qnum,f,isInited,))
d.start() 
t = Process(target=track, args=(qstop,qex,f,qtrack,qbox,qnum,))
t.start()


f.value=False
while(isInited.value==False):
    pass

for i in range(0,2530):
    # print(",,,,,,,",str(i))
    if(i%20==0):
        #
        qdet.put(i)
        qstop.put(i+20)
    else:
        qtrack.put(i)
        #

    #fps=FPS().start()
    # j=format(i, '05d')
    #j=i
    # print("f is: ",str(qf.qsize()))
    # time.sleep(0.011)
    # print(str(f.value))

    # if(f.value==True):
        # print("11111111111111111")
    # print("xxxx: ", j)
    # if (f.value == False):
    #     print("22222222222222222")
    #     qdet.put(j)
    #     qstop.put(j)
    #     print("detect: ",j)
    #     qf.put(1)
    #     f.value=True
    # qtrack.put(j)
    # print("track: ", j)

#fps=FPS().stop()

# qtrack.put('done')
#qstop.put(2600)
qdet.put('done')
print ('dne')
d.join()     
t.join()