
###########
import os

import numpy as np
from PIL import Image

from libyana.visutils import imagify

# Get images in "images" folder according to alphabetical order
image_folder = "images/hn10_65_74_crop3"  # NOTE:change
image_names = sorted(os.listdir(image_folder))
start_idx = 0
# if start_idx >= len(image_names) - 10:
#   start_idx = len(image_names) - 11
image_paths = [os.path.join(image_folder, image_name) for image_name in image_names[start_idx:start_idx + 10]]
# image_paths = [os.path.join(image_folder, image_name) for image_name in image_names[start_idx:start_idx + 5]]
print(image_paths)

# Convert images to numpy 
images = [Image.open(image_path) for image_path in image_paths if (image_path.lower().endswith(".jpg") or image_path.lower().endswith(".png"))]
# images就是一个size为10的list
images_np = [np.array(image) for image in images]


# Visualize the 10 frames
viz_num_images = 10
print(f"Loaded {len(images)} images, displaying the first {viz_num_images}")

#########
import trimesh 

# Get local object model
obj_path = "local_data/datasets/hoi4d/hn10/009.obj"
# obj_path = "local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj"


# Initialize object scale
obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm) 
                  # 这个之后也要调

obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
# 不进行scale操作,差不多高是-0.06到0.06
# obj_verts_can = obj_verts
obj_faces = np.array(obj_mesh.faces)


########
import os
import sys
sys.path.insert(0, "detectors/hand_object_detector/lib")
sys.path.insert(0, "external/frankmocap")


import numpy as np
from PIL import Image

from handmocap.hand_bbox_detector import HandBboxDetector   # 卡在这一步有问题

from homan.tracking import trackseq
from homan.mocap import get_hand_bbox_detector
from homan.utils.bbox import  make_bbox_square, bbox_xy_to_wh

# Load object mesh
hand_detector = get_hand_bbox_detector()
# seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1, "objects": 1})  # 没有检测到物体
seq_boxes = trackseq.track_sequence(images, 256, hand_detector=hand_detector, setup={"right_hand": 1})  # 这里有可视化的东西
hand_bboxes = {key: make_bbox_square(bbox_xy_to_wh(val), bbox_expansion=0.1) for key, val in seq_boxes.items() if 'hand' in key}
# obj_bboxes = [seq_boxes['objects']]
print("hand_bboxes", hand_bboxes)  # 好像只是前10帧的

# 自己specify object bounding box并展示
from homan.tracking import preprocess, trackboxes
from homan.utils.bbox import bbox_wh_to_xy
from libyana.visutils import detect2d, vizlines



###########################################  
# 这个是crop3的情况
obj_detected_boxes = {'object': [np.array([ 390 , 380  ,  195,  195 ]), 
                                 np.array([ 390 , 380  ,  195,  195 ]),
                                 np.array([ 390 , 380  ,  195,  195 ]), 
                                 np.array([ 390 , 380  ,  195,  195 ]),
                                 np.array([ 395 , 380  ,  195,  195 ]), 
                                 np.array([ 395 , 380  ,  195,  195 ]),
                                 np.array([ 400 , 380  ,  195,  195 ]),
                                 np.array([ 405 , 380  ,  195,  195 ]),
                                 np.array([ 405 , 375  ,  200,  200 ]),
                                 np.array([ 410 , 370  ,  200,  200 ])]}
#############################################

#############################################
# 这个是crop1的情况
# obj_detected_boxes = {'object': [np.array([ 220 , 300  ,  180,  180 ]), 
#                                  np.array([ 220 , 300  ,  180,  180 ]),
#                                  np.array([ 220 , 300  ,  180,  180 ]), 
#                                  np.array([ 220 , 300  ,  180,  180 ]),
#                                  np.array([ 225 , 300  ,  180,  180 ]), 
#                                  np.array([ 225 , 296  ,  180,  180 ]),
#                                  np.array([ 230 , 292  ,  180,  180 ]),
#                                  np.array([ 230 , 288  ,  180,  180 ]),
#                                  np.array([ 235 , 284  ,  200,  200 ]),
#                                  np.array([ 240 , 280  ,  200,  220 ])]}
########################################3333

obj_detected_boxes_wh = obj_detected_boxes['object']
obj_boxes_fwd = trackboxes.track_boxes(obj_detected_boxes_wh, out_xyxy=True)
obj_boxes_bwd = trackboxes.track_boxes(obj_detected_boxes_wh[::-1],
                                           out_xyxy=True)[::-1]
# Average predictions in both directions to get more
# robust tracks
obj_tracked_boxes = {}
obj_tracked_boxes['object'] = (obj_boxes_fwd + obj_boxes_bwd) / 2

# obj_tracked_boxes在后续也有用
obj_bboxes = [obj_tracked_boxes['object']]

# 加载模型
from homan.prepare.frameinfos import get_frame_infos
from homan.pointrend import MaskExtractor
from handmocap.hand_mocap_api import HandMocap
sample_folder = "tmp/"

# Initialize segmentation and hand pose estimation models
mask_extractor = MaskExtractor(pointrend_model_weights="detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl")
# 感觉是不是这个mask_extractor没有用固定的权重导致的
frankmocap_hand_checkpoint = "extra_data/hand_module/pretrained_weights/pose_shape_best.pth"
hand_predictor = HandMocap(frankmocap_hand_checkpoint, "extra_data/smpl")

# Define camera parameters
# NOTE: change
height, width, _ = images_np[0].shape
image_size = max(height, width)
# focal = 1080   # 为什么准确的focal效果还不如480,是不是bbox太小了??
# focal = 480
focal = 800
camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)
# camintr = np.array([[1079.19, 0, 952.03], [0, 1082.61, 531.14], [0, 0, 1]]).astype(np.float32)
camintrs = [camintr for _ in range(len(images_np))]

# Initialize object motion
person_parameters, obj_mask_infos, super2d_imgs = get_frame_infos(images_np,
                                                                  mask_extractor=mask_extractor,
                                                                  hand_predictor=hand_predictor,
                                                                  hand_bboxes=hand_bboxes,
                                                                  obj_bboxes=np.stack(obj_bboxes),
                                                                  sample_folder=sample_folder,
                                                                  camintr=camintrs,
                                                                  image_size=image_size,
                                                                  debug=False)

# Image.fromarray(super2d_imgs)
print("half done")
# 1877 MiB显存

from homan.pose_optimization import find_optimal_poses
from homan.lib2d import maskutils

object_parameters = find_optimal_poses(
    images=images_np,
    image_size=images_np[0].shape,
    vertices=obj_verts_can,
    faces=obj_faces,
    annotations=obj_mask_infos,
    num_initializations=200,  
    # num_initializations=600,    # 初始pose不准可能是这两个参数不够大
    # num_iterations=10, # Increase to get more accurate initializations
    num_iterations=50,
    Ks=np.stack(camintrs),
    viz_path=os.path.join(sample_folder, "optimal_pose.png"),
    debug=False,
)


# Add object object occlusions to hand masks
for person_param, obj_param, camintr in zip(person_parameters,
                                        object_parameters,
                                        camintrs):
    maskutils.add_target_hand_occlusions(
        person_param,
        obj_param,
        camintr,
        debug=False,
        sample_folder=sample_folder)

# 1413MiB显存

from homan.viz.colabutils import display_video
from homan.jointopt import optimize_hand_object

# coarse_num_iterations = 201 # Increase to give more steps to converge
coarse_num_iterations = 201
coarse_viz_step = 10 # Decrease to visualize more optimization steps
coarse_loss_weights = {
        "lw_inter": 1,
        "lw_depth": 0,
        "lw_sil_obj": 1.0,
        "lw_sil_hand": 0.0,
        "lw_collision": 0.0,
        "lw_contact": 0.0,
        "lw_scale_hand": 0.001,
        "lw_scale_obj": 0.001,
        "lw_v2d_hand": 50,
        "lw_smooth_hand": 2000,
        "lw_smooth_obj": 2000,
        "lw_pca": 0.004,
    }

# Camera intrinsics in normalized coordinate
camintr_nc = np.stack(camintrs).copy().astype(np.float32)
camintr_nc[:, :2] = camintr_nc[:, :2] / image_size

step2_folder = os.path.join(sample_folder, "jointoptim_step2")
step2_viz_folder = os.path.join(step2_folder, "viz")

# Coarse hand-object fitting
model, loss_evolution, imgs = optimize_hand_object(
    person_parameters=person_parameters,
    object_parameters=object_parameters,
    hand_proj_mode="persp",
    objvertices=obj_verts_can,
    objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
    optimize_mano=True,
    optimize_object_scale=True,
    loss_weights=coarse_loss_weights,
    image_size=image_size,
    num_iterations=coarse_num_iterations + 1,  # Increase to get more accurate initializations
    images=np.stack(images_np),
    camintr=camintr_nc,
    state_dict=None,
    viz_step=coarse_viz_step,
    viz_folder=step2_viz_folder,
)

last_viz_idx = (coarse_num_iterations // coarse_viz_step) * coarse_viz_step

# finegrained_num_iterations = 201   # Increase to give more time for convergence
finegrained_num_iterations = 5
finegrained_loss_weights = {
        "lw_inter": 1,
        "lw_depth": 0,
        "lw_sil_obj": 1.0,
        "lw_sil_hand": 0.0,
        "lw_collision": 0.001,
        "lw_contact": 1.0,
        "lw_scale_hand": 0.001,
        "lw_scale_obj": 0.001,
        "lw_v2d_hand": 50,
        "lw_smooth_hand": 2000,
        "lw_smooth_obj": 2000,
        "lw_pca": 0.004,
    }
finegrained_viz_step = 10 # Decrease to visualize more optimization steps

# Refine hand-object fitting
step3_folder = os.path.join(sample_folder, "jointoptim_step3")
step3_viz_folder = os.path.join(step3_folder, "viz")
model_fine, loss_evolution, imgs = optimize_hand_object(
    person_parameters=person_parameters,
    object_parameters=object_parameters,
    hand_proj_mode="persp",
    objvertices=obj_verts_can,
    objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
    optimize_mano=True,
    optimize_object_scale=True,
    loss_weights=finegrained_loss_weights,
    image_size=image_size,
    num_iterations=finegrained_num_iterations + 1,
    images=np.stack(images_np),
    camintr=camintr_nc,
    state_dict=model.state_dict(),
    viz_step=finegrained_viz_step, 
    viz_folder=step3_viz_folder,
)
last_viz_idx = (finegrained_num_iterations // finegrained_viz_step) * finegrained_viz_step

# 存储最后的mesh,之后可以用来计算
model_fine.save_obj("output/handnerf10_65_74/union_result.obj")
model_fine.save_hand_obj("output/handnerf10_65_74/hand_result.obj")
model_fine.save_object_obj("output/handnerf10_65_74/object_result.obj")