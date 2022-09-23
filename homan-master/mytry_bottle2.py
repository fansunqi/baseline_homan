import os
import numpy as np
from PIL import Image
from libyana.visutils import imagify

# Get images in "images" folder according to alphabetical order
image_folder = "images/bottle2_crop1"
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
# imagify.viz_imgrow(images_np[:viz_num_images])



import trimesh 

# Get local object model
obj_path = "local_data/datasets/hoi4d/bottle/bottle2.obj"
# obj_path = "local_data/datasets/shapenetmodels/206ef4c97f50caa4a570c6c691c987a8.obj"

# Initialize object scale
# obj_scale = 0.08  # Obj dimension in meters (0.1 => 10cm, 0.01 => 1cm) # 
obj_scale = 0.08

obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
# 不进行scale操作
# obj_verts_can = obj_verts
obj_faces = np.array(obj_mesh.faces)

# Display object vertices as scatter plot to visualize object shape
# print(f"Two projections of the centered object vertices, scaled to {obj_scale * 100} cm")
imagify.viz_pointsrow([obj_verts, obj_verts[:, 1:]])



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

# 1621MiB显存


# 自己specify object bounding box并展示
from homan.tracking import preprocess, trackboxes
from homan.utils.bbox import bbox_wh_to_xy
from libyana.visutils import detect2d, vizlines

# 先展示一下图片
from matplotlib import pyplot as plt
show_nb = 10
fig, axes = plt.subplots(2, show_nb, figsize=(4 * show_nb, 8))
frame_idxs = np.linspace(0, len(images) - 1, show_nb).astype(np.int)
for show_idx, frame_idx in enumerate(frame_idxs):
    image = images[frame_idx]
    if isinstance(image, str):
        image = preprocess.get_image(image, image_size)
    axis = axes[0, show_idx]
    axis.imshow(image)
    axis.axis("off")
    axis = axes[1, show_idx]
    axis.imshow(image)
    axis.axis("off")
    
#################################################
# 这个是crop1的情况
                                                       #width,height
obj_detected_boxes = {'object': [np.array([ 230 , 500  ,  150,  250 ]), 
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ]), 
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ]),
                                 np.array([ 230 , 500  ,  150,  250 ])]}


obj_detected_boxes_wh = obj_detected_boxes['object']
obj_boxes_fwd = trackboxes.track_boxes(obj_detected_boxes_wh, out_xyxy=True)
obj_boxes_bwd = trackboxes.track_boxes(obj_detected_boxes_wh[::-1],
                                           out_xyxy=True)[::-1]
# Average predictions in both directions to get more
# robust tracks
obj_tracked_boxes = {}
obj_tracked_boxes['object'] = (obj_boxes_fwd + obj_boxes_bwd) / 2

# Visualize tracked and detected hand+object bboxes
for show_idx, frame_idx in enumerate(frame_idxs):   # 一帧一帧地来
    # Display detected boxes
    orig_box = obj_detected_boxes['object'][frame_idx]
    axis = axes[0, show_idx]
    if orig_box is not None:
        detect2d.visualize_bbox(axis,
                                bbox_wh_to_xy(orig_box),
                                label='object')
    if show_idx == show_nb // 2:
        axis.set_title("Detected boxes")

    # Display tracked boxes
    axis = axes[1, show_idx]
    if show_idx == show_nb // 2:
        axis.set_title("Tracked boxes")
    detect2d.visualize_bbox(axis,
                            obj_tracked_boxes['object'][frame_idx],
                            label='object')

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
focal = 480   # 为什么准确的focal效果还不如480,是不是bbox太小了??
# focal = 480
# focal = 800
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

Image.fromarray(super2d_imgs)