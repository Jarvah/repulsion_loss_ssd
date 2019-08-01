from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import JE_ROOT, JE_CLASSES
from PIL import Image
from data import JEDetection, BaseTransform, JE_CLASSES
import torch.utils.data as data
from ssd import build_ssd
import os
import cv2


parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/home/waiyang/crowd_counting/repulsion_loss_ssd/weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='/home/waiyang/crowd_counting/repulsion_loss_ssd/eval', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.55, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--je_root', default=JE_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

VOC_CLASSES = (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')

labelmap=VOC_CLASSES

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

def test_net(save_folder, net, cuda, testset, transform, thresh,testset_name):

    pd_filename = os.path.join(save_folder,testset_name+'_pred.txt')
    test_output_root='/home/waiyang/crowd_counting/repulsion_loss_ssd/test_output2'

    num_images = len(testset)

    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img,img_id = testset.pull_image(i)
        frame = img

        output_img_path = os.path.join(test_output_root, testset_name, img_id[1] + "_new.jpg")

        x = torch.from_numpy(transform(img)).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data

        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:
                if pred_num == 0:
                    with open(pd_filename, mode='a') as f:
                        f.write(img_id[1]+' ')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                if label_name=='person':
                    pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    cv2.rectangle(frame,
                                (int(pt[0]), int(pt[1])),
                                (int(pt[2]), int(pt[3])),
                                 COLORS[i % 3], 2)

                    pred_num += 1
                    print(label_name)
                    with open(pd_filename, mode='a') as f:
                        f.write(str(i-1) + ' ' + str(score) + ' ' +' '.join(str(c) for c in coords)+' ')
                j += 1

        with open(pd_filename, mode='a') as f:
            f.write('\n')
        cv2.imwrite(output_img_path,frame)

def test():
    # load net

    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = JEDetection(args.je_root, ['IMM'])
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold,testset_name='IMM')


if __name__ == '__main__':
    test()
