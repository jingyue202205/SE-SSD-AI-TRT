import pdb
import struct
import sys
import torch


import argparse
import logging
import os
import os.path as osp
import shutil
import tempfile

import torch
import torch.distributed as dist
from det3d import torchie
from det3d.core import coco_eval, results2json
from det3d.datasets import  build_dataset
from det3d.datasets.kitti import kitti_common as kitti
from det3d.datasets.kitti.eval import get_official_eval_result
from det3d.datasets.utils.kitti_object_eval_python.evaluate import (evaluate as kitti_evaluate,)
from det3d.models import build_detector
from det3d.torchie.apis import init_dist
from det3d.torchie.apis.train import example_convert_to_torch
from det3d.torchie.parallel import MegDataParallel, MegDistributedDataParallel
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.trainer import example_to_device
from det3d.utils.dist.dist_common import (all_gather, get_rank, get_world_size, is_main_process, synchronize,)
from tqdm import tqdm
from det3d.torchie.parallel import collate, collate_kitti
from torch.utils.data import DataLoader



def parse_args():
    parser = argparse.ArgumentParser(description="MegDet test detector")
    parser.add_argument("--config", default='../examples/second/configs/kitti_car_vfev3_spmiddlefhd_rpn1_mghead_syncbn.py', help="test config file path")
    parser.add_argument("--checkpoint", default='cia-ssd-model.pth',  help="checkpoint file")
    parser.add_argument("--out", default='out.pkl', help="output result file")
    parser.add_argument("--json_out",  default='json_out.json', help="output result file name without extension", type=str)
    parser.add_argument("--eval", type=str, nargs="+", choices=["proposal", "proposal_fast", "bbox", "segm", "keypoints"], help="eval types",)
    parser.add_argument("--show", action="store_true", help="show results")
    parser.add_argument("--txt_result", default=True, help="save txt")
    parser.add_argument("--tmpdir", help="tmp dir for writing some results")
    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm", "mpi"], default="none",help="job launcher",)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    print(args)
    assert args.out or args.show or args.json_out, ('Please specify at least one operation (save or show the results) with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith((".pkl", ".pickle")):
        raise ValueError("The output file must be a pkl file.")

    if args.json_out is not None and args.json_out.endswith(".json"):
        args.json_out = args.json_out[:-5]

    cfg = torchie.Config.fromfile(args.config)
    if cfg.get("cudnn_benchmark", False):  # False
        torch.backends.cudnn.benchmark = True

    # cfg.model.pretrained = None
    # cfg.data.test.test_mode = True
    cfg.data.val.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint_path = os.path.join(cfg.work_dir, args.checkpoint)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")
    model.to('cuda:0')
    model.eval()

    with open('cia-ssd.wts', 'w') as f:
        f.write('{}\n'.format(len(model.state_dict().keys())))
        for k, v in model.state_dict().items():
            print('**'*10)
            print(k)
            # print(v)
            print(v.shape)
            # pdb.set_trace()
            vr = v.reshape(-1).cpu().numpy()
            f.write('{} {} '.format(k, len(vr)))
            for vv in vr:
                f.write(' ')
                f.write(struct.pack('>f', float(vv)).hex())
            f.write('\n')

if __name__ == "__main__":
    main()




