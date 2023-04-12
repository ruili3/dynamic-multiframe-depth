import os
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
# import jax

def save_pseudo_color_depth(depth:np.ndarray, colormap='rainbow', save_path="./"):
    return True

def disp_to_depth_metric(disp, min_depth=0.1, max_depth=100.0):
  """Convert network's sigmoid output into depth prediction, ref: Monodepth2
  """
  min_disp = 1 / max_depth
  max_disp = 1 / min_depth
  scaled_disp = min_disp + (max_disp - min_disp) * disp
  depth = 1 / scaled_disp
  return scaled_disp, depth



def save_color_imgs(image:np.ndarray, save_id=None, post_fix="img", save_path="./"):
    '''
    image: with shape (h,w,c=3)
    save_id = specify the name of the saved image
    '''
    if not isinstance(image, np.ndarray):
        raise Exception("Input image is not a np.ndarray")
    if not len(image.shape) == 3:
        raise Exception("Wong input shape. It should be a 3-dim image vector of shape (h,w,c)")

    if save_id is None:
        dirnames = os.listdir(save_path)
        save_id = len(dirnames)
    
    save_name = os.path.join(save_path,"{}_{}.jpg".format(save_id,post_fix))

    # for pytorch 
    if image.shape[-1]==3 or image.shape[-1]==1:
        plt.imsave(save_name, image)
    else:
        raise Exception("invalid color channel of the last dim")

    print(f"successfully saved {save_name}!")


def save_pseudo_color(input:np.ndarray, save_id=None, post_fix="error", pseudo_color="rainbow", save_path="./", vmax=None):
    '''
    input: with shape (h,w,c=3)
    save_id = specify the name of the saved error map
    '''
    if not isinstance(input, np.ndarray):
        raise Exception("Input input is not a np.ndarray")
    if not len(input.shape) == 3:
        raise Exception("Wong input shape. It should be a 3-dim input vector of shape (h,w,c)")
    if save_id is None:
        dirnames = os.listdir(save_path)
        save_id = len(dirnames)
    
    save_name = os.path.join(save_path,"{}_{}.jpg".format(save_id,post_fix))

    # for pytorch 
    if input.shape[-1]==1:
        disp_resized_np = input.squeeze(-1)
        # if "error_" in post_fix:
        #     print("save/photomtric error {} map:{},max:{},min:{},mean:{}".format(post_fix, disp_resized_np[:20,0],disp_resized_np.max(),disp_resized_np.min(),
        #                                                                             disp_resized_np.mean()))
        vmax = np.percentile(disp_resized_np, 95) if vmax is None else vmax
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=pseudo_color)
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        im.save(save_name)
    else:
        raise Exception("invalid color channel of the last dim")

    print(f"successfully saved {save_name}!")


def numpy_intensitymap_to_pcolor(input, vmin=None, vmax=None, colormap='rainbow'):
    '''
    input: h,w,1
    '''
    if input.shape[-1]==1:
        colormapped_im = numpy_1d_to_coloruint8(input, vmin, vmax, colormap)
        im = pil.fromarray(colormapped_im.astype(np.uint8))
        return im
    else:
        raise Exception("invalid color channel of the last dim")


def numpy_1d_to_coloruint8(input, vmin=None, vmax=None, colormap='rainbow'):
    '''
    input: h,w,1
    '''
    if input.shape[-1]==1:
        input = input.squeeze(-1)
        invalid_mask = (input == 0).astype(float)
        vmax = np.percentile(input, 95) if vmax is None else vmax
        vmin = 1e-3 if vmin is None else vmin  # vmin = input.min() if vmin is None else vmin
        normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
        colormapped_im = (mapper.to_rgba(input)[:, :, :3] * 255).astype(np.uint8)
        invalid_mask = np.expand_dims(invalid_mask,-1)
        colormapped_im = colormapped_im * (1-invalid_mask) + (invalid_mask * 255)
        return colormapped_im
    else:
        raise Exception("invalid color channel of the last dim")


def numpy_distancemap_validmask_to_pil(input, vmax=None, colormap='rainbow'):
    if input.shape[-1]==1:
        # For PIL image, you [should] squeeze the addtional dim and set the resolution strictly to "h,w"
        input = input.squeeze(-1)
        validmask = (input != 0).astype(float)
        mask_img = (validmask*255.0).astype(np.uint8)
        im = pil.fromarray(mask_img)
        return im
    else:
        raise Exception("invalid color channel of the last dim")

def numpy_rgb_to_pil(input):
    if input.shape[-1]==3:
        if input.max()<=1:
            colormapped_im = (input[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)
        else:
            im = pil.fromarray(input)
        return im
    else:
        raise Exception("invalid color channel of the last dim")


def get_error_map_value(pred_depth, gt_depth, grag_crop=True, median_scaling=True):
    '''
    input shape: h,w,c
    '''
    validmask = (gt_depth!=0).astype(bool)
    h,w,_ = gt_depth.shape

    if grag_crop:
        valid_area = (int(0.40810811 * h), int(0.99189189 * h), int(0.03594771 * w),  int(0.96405229 * w))
        area_mask = np.zeros(gt_depth.shape)
        area_mask[valid_area[0]:valid_area[1],valid_area[2]:valid_area[3],:] = 1.0
        validmask = (validmask * area_mask).astype(bool)

    if median_scaling:
        pred_median = np.median(pred_depth[validmask])
        gt_median = np.median(gt_depth[validmask])
        ratio = gt_median/pred_median
        pred_depth_rescale = pred_depth * ratio
    else:
        pred_depth_rescale = pred_depth

    absrel_map = np.zeros(gt_depth.shape)
    absrel_map[validmask] = np.abs(gt_depth[validmask]-pred_depth_rescale[validmask]) / gt_depth[validmask]

    absrel_val = absrel_map.sum() / validmask.sum()

    return absrel_map, absrel_val


# def save_concat_res(out_path, pred_disp, img, gt_depth):
#     '''
#     input shape: h,w,c
#     output concatenated img-gt-depth
#     '''
#     h,w,_ = gt_depth.shape

#     if img.shape[1]!=gt_depth.shape[1]:
#         img = jax.image.resize(img, (gt_depth.shape[0],gt_depth.shape[1], 3),"bilinear")
#         pred_disp = jax.image.resize(pred_disp, (gt_depth.shape[0],gt_depth.shape[1], 1),"bilinear")

#     pred_disp = np.array(pred_disp)
#     img = np.array(img)
#     gt_depth = np.array(gt_depth).squeeze(-1)

#     kernel = np.ones((5, 5), np.uint8)
#     gt_depth = cv2.dilate(gt_depth, kernel, iterations=1)
    
#     gt_depth = np.expand_dims(gt_depth,-1)
 
#     # get pil outputs
#     _, pred_depth = disp_to_depth_metric(pred_disp,min_depth=0.001,max_depth=80.0)

#     error_map, error_val = get_error_map_value(pred_depth, gt_depth)
    
#     error_pil = numpy_intensitymap_to_pcolor(error_map, vmin=0, vmax=0.5,colormap='jet')
#     pred_pil = numpy_intensitymap_to_pcolor(pred_depth)
#     gt_pil = numpy_intensitymap_to_pcolor(gt_depth)
#     img_pil = numpy_rgb_to_pil(img)


#     save_id = len(os.listdir(out_path))
#     save_name = os.path.join(out_path,"{}.png".format(save_id))

#     dst = Image.new('RGB', (w, h*4))
#     dst.paste(img_pil, (0, 0))
#     dst.paste(pred_pil, (0, h))
#     dst.paste(gt_pil, (0, 2*h))
#     dst.paste(error_pil, (0, 3*h))

    

#     dst.save(save_name)


def directly_save_intensitymap(input, out_path, save_id=None, post_fix="error"):
    '''
    input shape: h,w
    '''
    im = Image.fromarray(input)
    im.save(os.path.join(out_path, "{:06}_{}.png".format(save_id,post_fix)))
