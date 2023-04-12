import torch

from utils import preprocess_roi, get_positive_depth, get_absolute_depth, get_mask, mask_mean

from depth_proc_tools.plot_depth_utils import *
import cv2
from PIL import Image
import os


def a1_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt))
    return torch.mean((thresh < 1.25).type(torch.float))


def a2_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return torch.mean((thresh < 1.25 ** 2).type(torch.float))


def a3_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return torch.mean((thresh < 1.25 ** 3).type(torch.float))


def rmse_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    se = (depth_prediction - depth_gt) ** 2
    return torch.mean(torch.sqrt(torch.mean(se, dim=[1, 2, 3])))


def rmse_log_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    sle = (torch.log(depth_prediction) - torch.log(depth_gt)) ** 2
    return torch.mean(torch.sqrt(torch.mean(sle, dim=[1, 2, 3])))


def abs_rel_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    return torch.mean(torch.abs(depth_prediction - depth_gt) / depth_gt)


def sq_rel_metric(data_dict: dict, roi=None, max_distance=None):
    depth_prediction = data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    return torch.mean(((depth_prediction - depth_gt) ** 2) / depth_gt)


def find_mincost_depth(cost_volume, depth_hypos):
    argmax = torch.argmax(cost_volume, dim=1, keepdim=True)
    mincost_depth = torch.gather(input=depth_hypos, dim=1, index=argmax)
    return mincost_depth

def a1_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)

    return a1_base(depth_prediction, depth_gt, mask)



def a2_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return a2_base(depth_prediction, depth_gt, mask)


def a3_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return a3_base(depth_prediction, depth_gt, mask)


def rmse_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return rmse_base(depth_prediction, depth_gt, mask)


def rmse_log_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return rmse_log_base(depth_prediction, depth_gt, mask)


def abs_rel_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return abs_rel_base(depth_prediction, depth_gt, mask)


def sq_rel_sparse_metric(data_dict: dict, roi=None, max_distance=None, pred_all_valid=True, use_cvmask=False, eval_mono=False):
    depth_prediction = data_dict["result_mono"] if eval_mono else data_dict["result"]
    depth_gt = data_dict["target"]
    depth_prediction, depth_gt = preprocess_roi(depth_prediction, depth_gt, roi)
    mask = get_mask(depth_prediction, depth_gt, max_distance=max_distance, pred_all_valid=pred_all_valid)
    if use_cvmask: mask |= ~ (data_dict["mvobj_mask"] > .5)
    depth_prediction, depth_gt = get_positive_depth(depth_prediction, depth_gt)
    depth_prediction, depth_gt = get_absolute_depth(depth_prediction, depth_gt, max_distance)
    return sq_rel_base(depth_prediction, depth_gt, mask)



def save_results(path, name, img, gt_depth, pred_depth, validmask, cv_mask, costvolume):
    savepath = os.path.join(path, name)
    device=img.device
    bs,_,h,w = img.shape
    img = img[0,...].permute(1,2,0).detach().cpu().numpy() + 0.5
    gt_depth = gt_depth[0,...].permute(1,2,0).detach().cpu().numpy()
    gt_depth[gt_depth==80] = 0
    pred_depth = pred_depth[0,...].permute(1,2,0).detach().cpu().numpy()
    validmask = validmask[0,0,...].detach().cpu().numpy()
    cv_mask = cv_mask[0,0,...].detach().cpu().numpy()

    img = img #* np.expand_dims(cv_mask, axis=-1).astype(float)

    error_map, _ = get_error_map_value(pred_depth,gt_depth, grag_crop=False, median_scaling=False)

    errorpil = numpy_intensitymap_to_pcolor(error_map,vmin=0,vmax=0.5,colormap='jet')
    pred_pil = numpy_intensitymap_to_pcolor(pred_depth)
    gt_pil = numpy_intensitymap_to_pcolor(gt_depth)
    img_pil = numpy_rgb_to_pil(img)

    # generate pil validmask
    validmask_pil = Image.fromarray((validmask * 255.0).astype(np.uint8))
    cv_mask_pil = Image.fromarray((cv_mask * 255.0).astype(np.uint8))

    # cost
    print(bs,h,w)
    # print(f"cost volume shape:{costvolume.shape}")
    depths = (1 / torch.linspace(0.0025, 0.33, 32,device=device)).cuda().view(1, -1, 1, 1).expand(bs, -1, h, w)
    cost_volume_depth = find_mincost_depth(costvolume, depths)
    # print(f"cost shape:{cost_volume_depth.shape}")
    cost_volume_depth = cost_volume_depth[0,...].permute(1,2,0).detach().cpu().numpy()
    
    cv_depth_pil = numpy_intensitymap_to_pcolor(cost_volume_depth)


    h,w,_ = gt_depth.shape
    dst = Image.new('RGB', (w, h*3))
    dst.paste(img_pil, (0, 0))
    dst.paste(pred_pil, (0, h))
    dst.paste(gt_pil, (0, 2*h))
    # dst.paste(errorpil, (0, 3*h))
    # dst.paste(validmask_pil,(0,4*h))
    # dst.paste(cv_mask_pil,(0,5*h))
    # dst.paste(cv_depth_pil,(0,2*h))

    dst.save(savepath)
    print(f"saved to {savepath}")



def a1_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return a1_sparse_metric(data_dict, roi, max_distance, False)


def a2_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return a2_sparse_metric(data_dict, roi, max_distance, False)


def a3_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return a3_sparse_metric(data_dict, roi, max_distance, False)


def rmse_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_sparse_metric(data_dict, roi, max_distance, False)


def rmse_log_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_log_sparse_metric(data_dict, roi, max_distance, False)


def abs_rel_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return abs_rel_sparse_metric(data_dict, roi, max_distance, False)


def sq_rel_sparse_onlyvalid_metric(data_dict: dict, roi=None, max_distance=None):
    return sq_rel_sparse_metric(data_dict, roi, max_distance, False)


def a1_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return a1_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def a2_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return a2_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def a3_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return a3_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def rmse_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def rmse_log_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return rmse_log_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def abs_rel_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return abs_rel_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def sq_rel_sparse_onlydynamic_metric(data_dict: dict, roi=None, max_distance=None):
    return sq_rel_sparse_metric(data_dict, roi, max_distance, use_cvmask=True)


def a1_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt))
    return mask_mean((thresh < 1.25).type(torch.float), mask)


def a2_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return mask_mean((thresh < 1.25 ** 2).type(torch.float), mask)


def a3_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    thresh = torch.max((depth_gt / depth_prediction), (depth_prediction / depth_gt)).type(torch.float)
    return mask_mean((thresh < 1.25 ** 3).type(torch.float), mask)


def rmse_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    se = (depth_prediction - depth_gt) ** 2
    return torch.mean(torch.sqrt(mask_mean(se, mask, dim=[1, 2, 3])))


def rmse_log_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    depth_gt[mask] = 1
    depth_prediction[mask] = 1
    sle = (torch.log(depth_prediction) - torch.log(depth_gt)) ** 2
    return torch.mean(torch.sqrt(mask_mean(sle, mask, dim=[1, 2, 3])))


def abs_rel_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    return mask_mean(torch.abs(depth_prediction - depth_gt) / depth_gt, mask)


def sq_rel_base(depth_prediction: torch.Tensor, depth_gt: torch.Tensor, mask):
    return mask_mean(((depth_prediction - depth_gt) ** 2) / depth_gt, mask)