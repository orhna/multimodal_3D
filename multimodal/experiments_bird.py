import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from os.path import join, exists
import open3d as o3d
import matplotlib.pyplot as plt
from itertools import combinations
import copy
from tabulate import tabulate
import clip

import sys
sys.path.append('../')
from utils import *

model, _ = clip.load("ViT-L/14@336px")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# load all the required data
source_path = "/mnt/project/AT3DCV_Data/Preprocessed_OpenScene/data/augmented/birds_new/scannet_3d/example/scene0024_00_vh_clean_2.pth"
fused_path = "/mnt/project/AT3DCV_Data/Preprocessed_OpenScene/data/augmented/birds_new/fused/scene0024_00_0.pt"
distilled_path = "/mnt/project/AT3DCV_Data/Preprocessed_OpenScene/data/augmented/birds_new/features_3D/scene0024_00_vh_clean_2_openscene_feat_distill.npy"

source_points, source_colors, source_labels = load_scene(source_path, False)

fused_f, filtered_pc, filtered_pc_c, filtered_pc_labels, indices = load_fused_features(fused_path,
                                                                              source_points, 
                                                                              source_colors,
                                                                              source_labels)
distilled_f = load_distilled_features(distilled_path, indices)

#------------------------------------
#query to highlight
query = ["a bird"]    
#highlight and return similarity matrices
similarity = highlight_query(query, "fused", "max", distilled_f, fused_f, filtered_pc, filtered_pc_c, device)
#------------------------------------
#parse descriptors from openai api with gpt, set _nr to the number of descriptors you'd want to retrieve
_nr = 10
_prompt = f'Generate {str(_nr)} visual descriptors for each of the following categories, they are bird species: [Nicobar Pigeon, Eastern Rosella]. The descriptors will be used for input queries for a CLIP model. The descriptors should be concise and distinct from the descriptors of the other classes. Do not focus on behavior, but purely on attributes which are recognizable by the CLIP model. The output should be in the following form as a string: *bird name*: *descriptor1*, *descriptor2*, etc."'
descriptors = descriptors_from_prompt(_prompt, verbose = True)
#------------------------------------
#manually set the combinations of descriptors you want to evaluate
#
comb_dict_list =[ {'Nicobar Pigeon': [  'a bird which is/has Green-blue plumage.',
                                        'a bird which is/has Metallic-sheen feathers.',
                                        'a bird which is/has Slender body.',
                                        'a bird which is/has Long tail feathers.',
                                        'a bird which is/has Light brown head.'],
                  'Eastern Rosella': [  'a bird which is/has Red head.',
                                        'a bird which is/has Red shoulder patches.',
                                        'a bird which is/has Blue wings.',
                                        'a bird which is/has White breast.',
                                        'a bird which is/has Yellow belly.']
  }]
#------------------------------------
# set scannet labels
SCANNET_LABELS_20 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa',
                     'table', 'door', 'window', 'bookshelf', 'picture','counter', 'desk', 'curtain', 'refrigerator', 'shower curtain',
                     'toilet', 'sink', 'bathtub', 'otherfurniture']
UNKNOWN_ID = 255
NO_FEATURE_ID = 256

#------------------------------------
# iterate all the combinations and store their results in a list
class_IoU_result_list, class_accs_result_list, mean_iou_result_list, mean_acc_result_list, pred_ids_list = try_diff_combs(SCANNET_LABELS_20, comb_dict_list, model, "fused", "max", distilled_f, fused_f, filtered_pc_labels)


c1, c2= [], [] # Nicobar Pigeon, Easter Rosella
# store tp/ (tp + fp + fn) values in list per augmented class
for idx in range(len(class_IoU_result_list)):
    c1.append(class_IoU_result_list[idx]["Nicobar Pigeon"][0])
    c2.append(class_IoU_result_list[idx]["Eastern Rosella"][0])

# maximum IoU scores
print(max(c1), max(c2))

# these 5 descriptors gives the highest class IoU for Easter Rosella
print(comb_dict_list[c2.index(max(c2))]['Eastern Rosella'])

# these 5 descriptors gives the highest class IoU for Mouse-colored Tyrannulet
print(comb_dict_list[c1.index(max(c1))]['Nicobar Pigeon'])
#------------------------------------
# segmentation&visualization
pred_labels = pred_ids_list[c1.index(max(c1))].numpy()
#create color arrays and paint 
other_color = np.array([0.773, 0.922, 0.651])
color_gt = np.tile(other_color, (len(filtered_pc_labels), 1))
color_pred = np.tile(other_color, (len(filtered_pc_labels), 1))
color_gt[np.where(filtered_pc_labels == 20)] = [1, 0.294, 0.165] # nicobar pigeon :red
color_gt[np.where(filtered_pc_labels == 21)] = [0.024, 0.788, 1] # eastern rosella  :blue 
color_pred[np.where(pred_labels == 20)] = [1, 0.294, 0.165] # nicobar pigeon :red
color_pred[np.where(pred_labels == 21)] = [0.024, 0.788, 1] # eastern rosella  :blue 

# Create an Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_pc)
pcd.colors = o3d.utility.Vector3dVector(color_gt)

pcd_pred = o3d.geometry.PointCloud()
pcd_pred.points = o3d.utility.Vector3dVector(np.asarray(filtered_pc) + [0,10,0])
pcd_pred.colors = o3d.utility.Vector3dVector(color_pred)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd,pcd_pred])