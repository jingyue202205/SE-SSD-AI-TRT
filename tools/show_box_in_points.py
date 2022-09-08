import pdb
import json
import glob
import numpy as np
import sys
sys.path.insert(0, 'mayavi_tool')
from kitti_util import roty,rotz
import mayavi.mlab as mlab
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d


def compute_box_3d(obj):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    # print(obj['rt'])
    R = rotz(-obj['rt'])  # 15  +0.2617  -0.5234

    # 3d bounding box dimensions
    l = obj['l']
    w = obj['w']
    h = obj['h']

    # 3d bounding box corners
    # y_corners = [l / 2, l / 2, l / 2, l / 2, -l / 2, -l / 2, -l / 2, -l / 2]; # 1111---- a
    # x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2];                 # 11--11-- b
    # z_corners = [h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2]; # 1--11--1 c

    y_corners = [l / 2, l / 2, l / 2, l / 2, -l / 2, -l / 2, -l / 2, -l / 2]; # 1111---- a
    x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];                 # 1--11--1 c
    z_corners = [h / 2, h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2]; # 11--11-- b

    # y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]; # 11--11-- b
    # x_corners = [w/2, w/2, w/2, w/2, -w/2, -w/2, -w/2, -w/2];                 # 1111---- a
    # z_corners = [h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2]; # 1--11--1 c

    # y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]; # 11--11-- b
    # x_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2];                 # 1--11--1 c
    # z_corners = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]; # 1111---- a


    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    # print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj['x']
    corners_3d[1, :] = corners_3d[1, :] + obj['y']
    corners_3d[2, :] = corners_3d[2, :] + obj['z']
    # print 'cornsers_3d: ', corners_3d
    # print 'corners_2d: ', corners_2d
    return np.transpose(corners_3d)


def read_bin(path):
    return np.fromfile(path,dtype=np.float32).reshape((-1,4))


def read_txt(path):
    with open(path,'r') as f:
        data = f.readlines()
    data = [da.strip() for da in data]
    return data[1:]


def txt2json_dict(data):
    res_dict = {}
    labels = []
    for da in data:
        temp_dict = {}
        x,y,z,w,l,h,rt = da.split(',')[0:7]
        temp_dict['x'] = float(x)
        temp_dict['y'] = float(y)
        temp_dict['z'] = float(z)
        temp_dict['l'] = float(l)
        temp_dict['w'] = float(w)
        temp_dict['h'] = float(h)
        temp_dict['rt'] = float(rt)
        temp_dict['id'] = 0.0
        temp_dict['score'] = 1.0
        labels.append(temp_dict)
    res_dict['labels'] = labels
    return res_dict

def show():

    txt_path_prefix = '../data/outputs/'
    bin_root_ = '../data/kitti_training_velodyne_reduced/*.bin'
    fig = mlab.figure(figure=None, bgcolor=(0, 0, 0),
                      fgcolor=None, engine=None, size=(1000, 600))
    bin_paths = glob.glob(bin_root_)
    bin_paths = sorted(bin_paths,key=lambda x:int(x.split('/')[-1].split('.')[0]))
    for bin_path in bin_paths[0:]:
        txt_path = txt_path_prefix + bin_path.split('/')[-1].replace('.bin','.txt')
        print(bin_path)
        print(txt_path)
        points = read_bin(bin_path)
        data = read_txt(txt_path)
      
        res_data = txt2json_dict(data)
        

        draw_lidar(points, fig=fig)

        for label in res_data['labels']:
            # print(label)
            box3d_pts_3d = compute_box_3d(label)
            draw_gt_boxes3d([box3d_pts_3d], fig=fig)

        mlab.show(1)
        pdb.set_trace()
        mlab.clf(fig)


if __name__ == '__main__':
    show()