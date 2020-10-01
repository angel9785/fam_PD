#!/usr/bin/env python

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
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
import argparse
from copy import copy, deepcopy
from datasets.voc_eval import voc_eval
import xml.dom.minidom as minidom
import time
import cv2

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
    filename = os.path.join(data_dir, 'annotations', index + '.txt')
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

def demo_detection(net, data_dir, image_name, CLASSES, gt_roidb):
    """Detect object classes in an image using pre-computed object proposals."""
#    print cfg.TEST.SCALES
    # Load the demo image
    im1_file = os.path.join(data_dir,'color', image_name + '.jpg')
    im2_file = os.path.join(data_dir,'thermal', image_name + '.jpg')
    print (im1_file)
    print (im2_file)
    im1 = cv2.imread(im1_file)
    im2 = cv2.imread(im2_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect_2in(net, im1, im2)
    timer.toc()
    print (f'Detection took {timer.total_time}s' )

    # Visualize detections for each class
    #CONF_THRESH = 0.1
    NMS_THRESH = 0.3
#    NMS_THRESH = [0.2,0.2, 0.2]
    all_dets = None
    #for cls_ind, cls in enumerate(classes):
    #    cls_ind += 1 # because we skipped background
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        cls_inds = np.ones((len(keep),)) * cls_ind
        #cls_inds = np.ones((len(cls_scores),)) * cls_ind
        #cls_inds = np.ones((len(cls_scores),)) * (-1)
        #cls_inds[keep] = cls_ind
        dets = np.hstack((cls_inds[:,np.newaxis], dets))
        if all_dets is None:
            all_dets = dets
        else:
            all_dets = np.vstack((all_dets, dets))

    visual_detection_results(im1, boxes, scores, CLASSES, gt_roidb, threds=0.5)
    return all_dets

    # nms again
    #y = deepcopy(all_dets[:,1:6]).astype(np.float32)
    #keep = nms(y, 0.4)
    #all_dets = all_dets[keep,:]
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

def visual_detection_results(im, boxes, scores, CLASSES, gt_roidb, threds = 0.9):
    NMS_THRESH = 0.3
    im = im[:,:,(2,1,0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    cls_color = ['yellow','cyan','red','green','blue','white']


    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        if str(cls) != 'person':
            continue
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        cls_inds = np.ones((len(keep),)) * cls_ind

        inds = np.where(dets[:,-1]>threds)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            if(bbox[0]<0):
                bbox[0]=0
            if(bbox[1]<0):
                bbox[1]=0    
            bb=(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            tracker = OPENCV_OBJECT_TRACKERS["mosse"]()
            trackers.add(tracker, im, bb)
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=cls_color[2], linewidth=3.5)
                )
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.3f}'.format(cls, score),
                    bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')

    gt_boxes = gt_roidb['boxes']
    for gt_ind in range(gt_boxes.shape[0]):
        bbox = gt_boxes[gt_ind,:]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor=cls_color[3], linewidth=3.5)
            )


    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def save_detection_results(det_file, classes, dets):
    with open(det_file, 'w') as fid:
        if dets is None:
            return

        for det in dets:
           class_id = int(det[0])
           if class_id < 0:
               label = 'unknown'
           else:
               label = classes[class_id]
               if label.find('car') >= 0:
                   label = 'car'
           #fmt = '%s -1 -1 -10 %9.4f %9.4f %9.4f %9.4f -1 -1 -1 -1 -1 -1 -1 %9.4f\n'
           fmt = '%s %9.4f %9.4f %9.4f %9.4f %9.8f\n'
           fid.write(fmt % (label, det[1], det[2], det[3], det[4], det[5],))

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

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    print (args)
    model_dir = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16')
    prototxt = os.path.join(cfg.ROOT_DIR, 'models/kaist_fusion/VGG16', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(model_dir, 'VGG16_faster_rcnn_final_kaist_fusion.caffemodel')

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    

    CLASSES = ('__background__','person')
 
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print ('\n\nLoaded network {:s}'.format(caffemodel))
    print (prototxt)
    # print det_classes

    #train_scale = model_instance.split('_')[-1]
#    train_scale = 600
    cfg.TEST.SCALES = (args.train_scale,)
    cfg.TEST.RPN_POST_NMS_TOP_N = args.num_proposals
    cfg.TEST.MAX_SIZE = args.test_scale
    
    im_names = ['I02555.jpg', 'I02556.jpg', 
                'I02557.jpg', 'I02558.jpg',
                'I02559.jpg', 'I02560.jpg', 
                'I02561.jpg', 'I02562.jpg', 
                'I02563.jpg', 'I02564.jpg', 
                'I02565.jpg', 'I02566.jpg',
                'I02567.jpg', 'I02568.jpg',
                'I02569.jpg', 'I02570.jpg']
    data_dir = os.path.join(cfg.ROOT_DIR, 'data/demo_pedestrian')            
    im="I02555"
    #tracker = OPENCV_OBJECT_TRACKERS["mosse"]()
    # print(im)

    #print("22222")
    gt = load_kitti_annotation(data_dir, im)
    det = demo_detection(net, data_dir, im, CLASSES, gt)
    plt.show()
    #trackers.add(tracker, im, gt['boxes'])
    t=0
    for im_name in im_names:
        
        im="/home/fereshteh/multispectral-pedestrian-py-faster-rcnn-master/data/demo_pedestrian/color/"+im_name
        im = cv2.imread(im)
        t1=time.time()
        (success, boxes) = trackers.update(im)
        t2=time.time() 
        t=t+(t2-t1)
        print(t2-t1)
        for box in boxes:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('image',im)
        key = cv2.waitKey(1) & 0xFF 
    t=t/len(im_names)  
    print("------",t)  
        
		
    



