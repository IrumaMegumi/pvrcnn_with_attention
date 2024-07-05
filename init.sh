#!/bin/bash

# copy files
cp -r src/models/backbones_2d/* OpenPCDet/detector/pcdet/models/backbones_2d/

cp -r src/models/backbones_3d/*.py OpenPCDet/detector/pcdet/models/backbones_3d/
cp -r src/models/backbones_3d/pfe/* OpenPCDet/detector/pcdet/models/backbones_3d/pfe/
cp -r src/models/backbones_3d/cfe OpenPCDet/detector/pcdet/models/backbones_3d/

cp -r src/models/detectors/* OpenPCDet/detector/pcdet/models/detectors/

cp -r src/ops/pointnet2/pointnet2_stack/* OpenPCDet/detector/pcdet/ops/pointnet2/pointnet2_stack/
cp -r src/ops/pointnet2/pointnet2_batch/* OpenPCDet/detector/pcdet/ops/pointnet2/pointnet2_batch/

cp -r src/tools/* OpenPCDet/detector/tools/

cp -r configs/* OpenPCDet/detector/tools/cfgs/kitti_models/

cp -r requirements.txt OpenPCDet/detector/


