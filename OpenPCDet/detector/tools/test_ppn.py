from pcdet.models.backbones_3d.pfe.PointProposal import PointProposalNet_v2
from pcdet.datasets import build_dataloader
import argparse
from pcdet.config import cfg, cfg_from_yaml_file, log_config_to_file
from pathlib import Path
from pcdet.utils import common_utils
import torch
import open3d as o3d
import numpy as np
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
def compute_3d_box_corners(x, y, z, h, w, l, yaw,is_in_velodyne=False):
    # 旋转矩阵（绕y轴）
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    
    # 3D边界框的8个顶点在物体坐标系下的坐标
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    
    # 将顶点坐标转换为numpy数组
    corners_3d = np.array([x_corners, y_corners, z_corners])
    
    # 使用旋转矩阵R旋转顶点坐标，然后加上中心点坐标以转换到相机坐标系
    corners_3d_rotated = np.dot(R, corners_3d)
    corners_3d_camera = corners_3d_rotated + np.array([[x], [y], [z]])
    #corners_3d_camera=np.append(corners_3d_camera,np.array([1,1,1,1,1,1,1,1]).reshape(1,8),axis=0)
    corners_3d_camera = np.vstack((corners_3d_camera, np.zeros((1, corners_3d_camera.shape[-1]))))
    if(is_in_velodyne==True):
        corners_3d_camera[-1][-1]=1
        return corners_3d_camera
    return corners_3d_camera

def draw_3dframeworks(vis, points):
    render_option = vis.get_render_option()
    position = points
    points_box = np.transpose(position[0:3,:])
    lines_box = np.array([[0, 1], [1, 2], [0, 3], [2, 3], [4, 5], [4, 7], [5, 6], [6, 7],
                          [0, 4], [1, 5], [2, 6], [3, 7], [0, 5], [1, 4]])
    colors = np.array([[1., 0., 0.] for j in range(len(lines_box))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points_box)
    line_set.lines = o3d.utility.Vector2iVector(lines_box)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    render_option.line_width = 5.0
    vis.update_geometry(line_set)
    render_option.background_color = np.asarray([1, 1, 1])
    # vis.get_render_option().load_from_json('renderoption_1.json')
    render_option.point_size = 4
    # param = o3d.io.read_pinhole_camera_parameters('BV.json')

    print(render_option.line_width)
    ctr = vis.get_view_control()

    vis.add_geometry(line_set)
    # ctr.convert_from_pinhole_camera_parameters(param)
    vis.update_geometry(line_set)
    vis.update_renderer()

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size for training')
    parser.add_argument('--epochs',type=int,default=150,required=False, help='number of epochs to train for')
    parser.add_argument('--workers',type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--num_object_points',type=int,default=6000,help='number of points you selected from original points, 6000 is the number for kitti dataset')
    parser.add_argument('--num_keypoints',type=int,default=2048, help='number of keypoints you want to get')
    
    #useless parameters
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    # TODO: it is not elegant
    parser.add_argument('--cfg_file',type=str, default=None, help='specify the config for training')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    return args, cfg
def main():
    args, cfg = parse_config()
    if getattr(args, 'launcher', None) == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers, logger=None, training=False
    )
    model_dict=torch.load("ppn.pth",map_location=torch.device('cuda'))
    model=PointProposalNet_v2(num_object_points=6000,num_keypoints=2048)
    model.load_state_dict(model_dict['model_state'])
    model.cuda()
    model.eval()

    with torch.no_grad():
        for data_dict in test_loader:
            sampled_points=torch.from_numpy(data_dict['painted_points']).unsqueeze(0)
            sampled_points=sampled_points.to(torch.float32).to('cuda')
            keypoints=model(data_dict,is_training=False)
            keypoints_xyz=keypoints
            keypoints_xyz=keypoints_xyz.cpu().numpy()
            frame_id=data_dict['frame_id'][0]
            # cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
            #         sampled_points[:, :, 0:3].contiguous(), 2048
            #     ).long()
            # if sampled_points.shape[1] < 2048:
            #     times = int(2048/ sampled_points.shape[1]) + 1
            #     non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
            #     cur_pt_idxs[0] = non_empty.repeat(times)[:2048]
            # keypoints = sampled_points[0][cur_pt_idxs[0]]
            # keypoints_xyz=keypoints[:,1:4].cpu().numpy()
            np.save(f'{frame_id}.npy', keypoints_xyz)
            print("yes")
if __name__=='__main__':
    main()