import os
import torch
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from os.path import join, exists
import open3d as o3d
import matplotlib.pyplot as plt
from itertools import combinations
import copy
from tabulate import tabulate
import clip

def load_scene(path, visualize = True):
    
    sample = torch.load(path)
    sample_points  = sample[0]
    sample_colors = sample[1]
    sample_labels = sample[2]
    
    if visualize:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(sample_points))
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(sample_colors))
        o3d.visualization.draw_geometries([pcd])
        
    return sample_points, sample_colors, sample_labels

def load_fused_features(path, sample_points, sample_colors, sample_labels):
    
    features = torch.load(path)
    indices = torch.nonzero(features["mask_full"]).squeeze()
    filtered_point_cloud = sample_points[indices, :]
    filtered_point_cloud_colors = sample_colors[indices, :]
    filtered_point_cloud_labels = sample_labels[indices]
    fused_features = (features["feat"]/(features["feat"].norm(dim=-1, keepdim=True)+1e-5))
    
    return fused_features, filtered_point_cloud, filtered_point_cloud_colors, filtered_point_cloud_labels, indices

def load_distilled_features(path, indices):
    
    distilled = np.load(path)
    #cast and normalize embeddings for distilled 
    distilled = distilled[indices, :]
    distilled_t = torch.from_numpy(distilled).half()
    distilled_f = (distilled_t/(distilled_t.norm(dim=-1, keepdim=True)+1e-5))
    
    return distilled_f

def highlight_query(query, feature_type, agg_type, distill, fused, fpc, fpcc, device):
    
    import clip
    model, preprocess = clip.load("ViT-L/14@336px")
    
    with torch.no_grad():
        per_descriptor_embeds = []
        for descriptor in tqdm(query):
            texts = clip.tokenize(descriptor)  #tokenize
            texts = texts.to(device)
            text_embeddings = model.encode_text(texts)  #embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            per_descriptor_embeds.append(text_embeddings)

        per_descriptor_embeds = torch.stack(per_descriptor_embeds, dim=1).squeeze()

    if feature_type == "fused":
        similarity_matrix = fused.to(device) @ per_descriptor_embeds.T
    elif feature_type == "distilled":
        similarity_matrix = distill.to(device) @ per_descriptor_embeds.T
    elif feature_type == "ensembled":
        pred_fusion = fused.to(device) @ per_descriptor_embeds
        pred_distill = distill.to(device) @ per_descriptor_embeds.T
        feat_ensemble = distill.clone().half()
        mask_ = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
        feat_ensemble[mask_] = fused_f[mask_]
        similarity_matrix = feat_ensemble.to(device) @ per_descriptor_embeds.T
        

    if similarity_matrix.ndim == 2:
        if agg_type == "mean":
            agg_sim_mat = torch.mean(similarity_matrix, dim=1)
        elif agg_type == "max":
            agg_sim_mat, _ = torch.max(similarity_matrix, dim=1)
    else: 
        agg_sim_mat = similarity_matrix
        
    agg_sim_mat = agg_sim_mat.reshape(-1, 1)
    
    # creating pc
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(fpc))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(fpcc))

    # heatmap
    cmap = plt.get_cmap('cividis')

    # normalize the tensor to the range [0, 1]
    normalized_tensor = (agg_sim_mat - torch.min(agg_sim_mat)) / (torch.max(agg_sim_mat) - torch.min(agg_sim_mat))

    colors = cmap(normalized_tensor.detach().cpu().numpy().squeeze())
    pcd_heatmap = o3d.geometry.PointCloud()

    pcd_heatmap.points = o3d.utility.Vector3dVector(pcd.points)
    pcd_heatmap.colors = o3d.utility.Vector3dVector(colors[:, :3])

    #transform heatmap to the side
    pcd_heatmap.points = o3d.utility.Vector3dVector(np.asarray(pcd.points) + [0,10,0])

    o3d.visualization.draw_geometries([pcd, pcd_heatmap])
    
    return agg_sim_mat

def confusion_matrix(pred_ids, gt_ids, num_classes):
    '''calculate the confusion matrix.'''

    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs = gt_ids != UNKNOWN_ID
    if NO_FEATURE_ID in pred_ids: # some points have no feature assigned for prediction
        print("no features")
        pred_ids[pred_ids==NO_FEATURE_ID] = num_classes
        confusion = np.bincount(
            pred_ids[idxs] * (num_classes+1) + gt_ids[idxs],
            minlength=(num_classes+1)**2).reshape((
            num_classes+1, num_classes+1)).astype(np.ulonglong)
        return confusion[:num_classes, :num_classes]

    return np.bincount(
        pred_ids[idxs] * num_classes + gt_ids[idxs],
        minlength=num_classes**2).reshape((
        num_classes, num_classes)).astype(np.ulonglong)

def get_iou(label_id, confusion, gts):
    '''calculate IoU.'''

    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false positives
    fp = np.longlong(confusion[label_id, :].sum()) - tp
    # false negatives
    fn = np.longlong(confusion[:, label_id].sum()) - tp

    total = np.sum(gts == label_id)
    denom = (tp + fp + fn)
    if denom == 0:
        return float('nan')
    return float(tp) / denom, tp, denom, total

def evaluate(labelset, descriptors, model, feature_type, agg_type, distill, fused, gt_ids):
        
    descriptor_lengths = []
    
    with torch.no_grad():
        label_embeds = []
        for category in labelset:
            if not isinstance(category, str): # if not string, process in another loop
                descriptor_lengths.append(len(category)) # get length of descriptors
                for desc in category:
                    texts = clip.tokenize(desc)  #tokenize
                    texts = texts.cuda()
                    text_embeddings = model.encode_text(texts)  #embed with text encoder
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                    label_embeds.append(text_embeddings)
            else: # if string, just process
                _prompt = f'a {category} in a scene' 
                texts = clip.tokenize(_prompt)  #tokenize
                texts = texts.cuda()
                text_embeddings = model.encode_text(texts)  #embed with text encoder
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                label_embeds.append(text_embeddings)
                
    label_embeds = torch.cat(label_embeds, dim=0) # has the shape of [768, *original labels*+*descriptors*]

    if feature_type == "fused":
        similarity_matrix = fused.to(device) @ label_embeds.T
    elif feature_type == "distilled":
        similarity_matrix = distill.to(device) @ label_embeds.T
    elif feature_type == "ensembled":
        pred_fusion = fused.to(device) @ label_embeds.T
        pred_distill = distill.to(device) @ label_embeds.T
        feat_ensemble = distill.clone().half()
        mask_ = pred_distill.max(dim=-1)[0] < pred_fusion.max(dim=-1)[0]
        feat_ensemble[mask_] = fused_f[mask_]
        similarity_matrix = feat_ensemble.to(device) @ label_embeds.T
    
    # separating the similarity matrix for the original labels and descriptors, 20 only for ScanNet
    sim_labels = similarity_matrix[:, :20]
    sim_descriptors = similarity_matrix[:, 20:]
            
    # aggregate the corresponding descriptor vectors by the length of each
    # keep them in a list to stack it later
    _idx = 0
    agg_desc= []
    for elem in descriptor_lengths:
        if agg_type == "mean":
            to_mean = sim_descriptors[:, _idx : _idx + elem]
            agg_desc_sim_mat = torch.mean(to_mean, dim=1)
        elif agg_type == "max":
            to_max = sim_descriptors[:, _idx : _idx + elem]
            agg_desc_sim_mat, _ = torch.max(to_max, dim=1)
        _idx += elem
        agg_desc.append(agg_desc_sim_mat)
        
    # stack aggregated descriptor similarity matrices
    agg_sim_descriptors = torch.stack(agg_desc, dim = 1)
    # combine with the similarity matrix of labels
    agg_sim_mat = torch.cat([sim_labels,agg_sim_descriptors], dim = 1)

    # get the predictions
    pred_ids = torch.max(agg_sim_mat, 1)[1].detach().cpu()    
    
    N_CLASSES = len(labelset)
    
    confusion = confusion_matrix(pred_ids, gt_ids, N_CLASSES)
    class_ious = {}
    class_accs = {}
    mean_iou = 0
    mean_acc = 0
    
    count = 0
    for i in range(N_CLASSES):
        label_name = labelset[i]
        
        if not isinstance(label_name, str): 
            for key, value in descriptors.items():
                if value == label_name:
                    label_name = key
        if (gt_ids==i).sum() == 0: # at least 1 point needs to be in the evaluation for this class
            continue
            
        class_ious[label_name] = get_iou(i, confusion, gt_ids)
        class_accs[label_name] = class_ious[label_name][1] / (gt_ids==i).sum()
        count+=1

        mean_iou += class_ious[label_name][0]
        mean_acc += class_accs[label_name]

    mean_iou /= N_CLASSES
    mean_acc /= N_CLASSES
    
    return class_ious, class_accs, mean_iou, mean_acc, pred_ids

def print_results(labelset, class_ious, descriptors):
    
    print('classes                 IoU/ total')
    print('----------------------------')
    for i in range(len(labelset)):
        label_name = labelset[i]
        if not isinstance(label_name, str): 
            for key, value in descriptors.items():
                if value == label_name:
                    label_name = key
        try:
            print('{0:<14s}             :          {1:>5.5f}           ({2:>6d}/{3:<6d}   /{4:<6d})'.format(
                    label_name,
                    class_ious[label_name][0],
                    class_ious[label_name][1],
                    class_ious[label_name][2],
                    class_ious[label_name][3]))
        except:
            print(label_name + ' error!')
            continue
            
def print_results_table(labelset, class_ious, descriptors):
    
    results = [["classes","IoU", "tp/(tp + fp + fn)", "total points" ]]
    
    for i in range(len(labelset)):
        label_name = labelset[i]
        if not isinstance(label_name, str): 
            for key, value in descriptors.items():
                if value == label_name:
                    label_name = key
        try:
            results.append([label_name,
                            format(class_ious[label_name][0], '.5f'),
                            f'{class_ious[label_name][1]}/{class_ious[label_name][2]}',
                            class_ious[label_name][3]])
        except:
            results.append([label_name, "---", "---","---"])        
            continue
            
        table = tabulate(results, headers="firstrow", tablefmt="rounded_outline")
        
    print(table)
        
def descriptors_from_prompt(text, verbose = True):
    
    import openai

    openai.api_key = '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'

    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=text,

      temperature=0.5,
      max_tokens=200
    )
    
    lines = [s for s in [line.strip() for line in response["choices"][0].text.splitlines()] if s]
    
    if verbose:
        print(response["choices"][0].text)
        print(lines)
    
    
    descriptors = {}
    
    for line in lines:
        parts = line.split(":")
        key = parts[0].strip()
        features = [f'a bird which is/has {f.strip()}.' for f in parts[1].split(",")]
        descriptors[key] = features
        
    return descriptors


def combinations_descriptor(descriptors, subset_len):
    
    combinations_dict= {}
    for key, value in descriptors.items():
        value_combinations = list(combinations(value, subset_len))
        combinations_dict[key] = value_combinations
    
    comb_dict_list = []
    for i in range(len(combinations_dict[next(iter(combinations_dict))])):
        temp = {}
        for key in combinations_dict.keys():
            temp[key] = [str(item) for item in combinations_dict[key][i]]
        comb_dict_list.append(temp)
        
    return comb_dict_list

def try_diff_combs(labelset, comb_dict_list, model, feature_type, agg_type, distill, fused, gt_ids):

    class_IoU_result_list = []
    class_accs_result_list = []
    mean_iou_result_list = []
    mean_acc_result_list = []
    pred_ids_list = []
    
    for elem in tqdm(comb_dict_list):
        temp_labelset = copy.deepcopy(labelset)
        
        for key, value in elem.items():
            temp_labelset.append(value)
        
        class_ious, class_accs, mean_iou, mean_acc, pred_ids = evaluate(temp_labelset, elem, model, feature_type, agg_type , distill, fused, gt_ids)
        class_IoU_result_list.append(class_ious)
        class_accs_result_list.append(class_accs)
        mean_iou_result_list.append(mean_iou)
        mean_acc_result_list.append(mean_acc)
        pred_ids_list.append(pred_ids)
        
    return class_IoU_result_list, class_accs_result_list, mean_iou_result_list, mean_acc_result_list, pred_ids_list