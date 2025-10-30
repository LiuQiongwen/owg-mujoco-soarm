#!/usr/bin/env bash
set -e
export PYBULLET_EGL=1
python grasp_6dof/generate_grasps_open3d.py \
  --obj cube.urdf --cube-scale 0.08 \
  --n-cand 1000 --yaw-samples 12 --voxel 0.005 \
  --topk 50 --topk-bullet 150 \
  --renderer opengl --vis 0 --img 256 256

