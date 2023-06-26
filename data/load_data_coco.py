"""
fuse training and testing
"""

import os
import copy
import numpy as np
from typing import Any
import torch.utils.data as data
from pycocotools.coco import COCO


def bbox_clip_xyxy(xyxy, width, height):
    """Clip bounding box with format (xmin, ymin, xmax, ymax) to specified boundary.

    All bounding boxes will be clipped to the new region `(0, 0, width, height)`.

    Parameters
    ----------
    xyxy : list, tuple or numpy.ndarray
        The bbox in format (xmin, ymin, xmax, ymax).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.
    width : int or float
        Boundary width.
    height : int or float
        Boundary height.

    Returns
    -------
    type
        Description of returned object.

    """
    if isinstance(xyxy, (tuple, list)):
        if not len(xyxy) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xyxy)))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[3]))
        return (x1, y1, x2, y2)
    elif isinstance(xyxy, np.ndarray):
        if not xyxy.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xyxy.shape))
        x1 = np.minimum(width - 1, np.maximum(0, xyxy[:, 0]))
        y1 = np.minimum(height - 1, np.maximum(0, xyxy[:, 1]))
        x2 = np.minimum(width - 1, np.maximum(0, xyxy[:, 2]))
        y2 = np.minimum(height - 1, np.maximum(0, xyxy[:, 3]))
        return np.hstack((x1, y1, x2, y2))
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xyxy)))


def bbox_xywh_to_xyxy(xywh):
    """Convert bounding boxes from format (x, y, w, h) to (xmin, ymin, xmax, ymax)

    Parameters
    ----------
    xywh : list, tuple or numpy.ndarray
        The bbox in format (x, y, w, h).
        If numpy.ndarray is provided, we expect multiple bounding boxes with
        shape `(N, 4)`.

    Returns
    -------
    tuple or numpy.ndarray
        The converted bboxes in format (xmin, ymin, xmax, ymax).
        If input is numpy.ndarray, return is numpy.ndarray correspondingly.

    """
    if isinstance(xywh, (tuple, list)):
        if not len(xywh) == 4:
            raise IndexError(
                "Bounding boxes must have 4 elements, given {}".format(len(xywh)))
        w, h = np.maximum(xywh[2] - 1, 0), np.maximum(xywh[3] - 1, 0)
        return (xywh[0], xywh[1], xywh[0] + w, xywh[1] + h)
    elif isinstance(xywh, np.ndarray):
        if not xywh.size % 4 == 0:
            raise IndexError(
                "Bounding boxes must have n * 4 elements, given {}".format(xywh.shape))
        xyxy = np.hstack((xywh[:, :2], xywh[:, :2] + np.maximum(0, xywh[:, 2:4] - 1)))
        return xyxy
    else:
        raise TypeError(
            'Expect input xywh a list, tuple or numpy.ndarray, given {}'.format(type(xywh)))


class COCO_Dataset(data.Dataset):
    def __init__(self, opt, root_path, train=True, skip_empty=True) -> None:
        super().__init__()
        self.train = train 
        self._root = root_path
        self._skip_empty = skip_empty
        self.items = []
        self.labels = []
        self.CLASSES = ['person']
        self.EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.num_joints = 17
        self.joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8],
                        [9, 10], [11, 12], [13, 14], [15, 16]]
        self.joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                        'left_shoulder', 'right_shoulder',                           # 6
                        'left_elbow', 'right_elbow',                                 # 8
                        'left_wrist', 'right_wrist',                                 # 10
                        'left_hip', 'right_hip',                                     # 12
                        'left_knee', 'right_knee',                                   # 14
                        'left_ankle', 'right_ankle')                                 # 16
        
        if not self.train:
            opt.annotations = "/users/shuoyang67/data/coco/annotations/person_keypoints_val2017.json"
        self.coco = COCO(opt.annotations)

        classes = [c['name'] for c in self.coco.loadCats(self.coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(self.coco.getCatIds())}
        
        # iterate through the annotations
        image_ids = sorted(self.coco.getImgIds())
        for entry in self.coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self._root, dirname, filename)
            label = self._check_load_keypoints(self.coco, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                self.items.append(abs_path)
                self.labels.append(obj)

    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= 1:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue
            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 1), dtype=np.float32)
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
                joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
                joints_3d[i, 2, 0] = obj['keypoints'][i * 3 + 2]
                # joints_3d[i, 2, 0] = 0
                # visible = min(1, obj['keypoints'][i * 3 + 2])
                # joints_3d[i, :2, 1] = visible
                # joints_3d[i, 2, 1] = 0

            # if np.sum(joints_3d[:, 0, 1]) < 1:
            #     # no visible keypoint
            #     continue

            # if self._check_centers and self.train:
            #     bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
            #     kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
            #     ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
            #     if (num_vis / 80.0 + 47 / 80.0) > ks:
            #         continue

            # valid_objs.append({
            #     'bbox': (xmin, ymin, xmax, ymax),
            #     'width': width,
            #     'height': height,
            #     'joints_3d': joints_3d
            # })

            valid_objs.append({
                'joints_3d': joints_3d,
                'bbox': np.array([xmin, ymin, xmax, ymax])
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, index) -> Any:
         # load ground truth, including bbox, keypoints, image size
         # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
        label = copy.deepcopy(self.labels[index])
        key_points_3d = label.pop('joints_3d')
        key_points_2d = key_points_3d[:, :2, :]
        bb_box = label.pop('bbox')
        scale = np.float32(1.0)
        action = np.int32(0)
        return key_points_3d[:, np.newaxis, ...], key_points_2d[:, np.newaxis, ...], scale, action, bb_box
    

if __name__ == '__main__':
    class opts:
        def __init__(self) -> None:
            self.annotations = '/users/shuoyang67/data/coco/annotations/person_keypoints_train2017.json'

    root = '/users/shuoyang67/data/coco'
    opt_coco = opts()
    dataloader = COCO_Dataset(opt_coco, root)

    sample = dataloader[1]

    print('finished')
