from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import glob

JE_CLASSES = (  # always index 0
    'person')

#JE dataset path
JE_ROOT = osp.join(HOME, "crowd_counting/Dataset/test_image_20190527")

class JEDetection:
    """JE Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to JE dataset folder.
        image_set (string): IMM, jCube, WestGate imageset
        dataset_name : which dataset to load

    """

    def __init__(self, root,

                 image_sets=['IMM','jCube','WestGate'],
                 dataset_name='JE'):
        self.root = root
        self.image_set = image_sets
        #self.target_transform = target_transform
        self.name = dataset_name
        #self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', '%s.jpg')
        self.ids = list()
        for location in image_sets:
            rootpath = osp.join(self.root, location)
            folders = glob.glob(osp.join(rootpath, '*'))
            for folder in folders:
                for path in glob.glob(folder + '/*.jpg'):
                    self.ids.append((folder,path.replace(folder+'/','').replace('.jpg','')))


    #def __getitem__(self, index):
    #    im, gt, h, w = self.pull_item(index)

     #   return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]
        #target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape
        return torch.from_numpy(img).permute(2, 0, 1), height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR),img_id


    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
