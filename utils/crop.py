# author: Anil Armagan
# contact: a.armagan@imperial.ac.uk
# date: 20/07/2019
import numpy as np


def crop(img, skeleton, u0=315.944855, v0=245.287079, fx=475.065948, fy=475.065857, bbsize=300.0, center_joint=3, offset=30):
    # crop hand from input depth img with gt joint annotations
    # hand is cropped by fitting a 3D bounding box of size bbsize around the MCP joint of the middle finger.
    # img: WxH
    # skeleton: 21x3
    # returns: cropped_image:
    #             center_joint_uvd:
    #             joints2d_cropped
    success = False
    minu, maxu = min(skeleton[:, 0])-offset, max(skeleton[:, 0])+offset
    minv, maxv = min(skeleton[:, 1])-offset, max(skeleton[:, 1])+offset
    mind, maxd = min(skeleton[:, 2])-offset, max(skeleton[:, 2])+offset

    height = img.shape[0]
    width = img.shape[1]

    # create point cloud and mask with minu/v/d and maxu/v/d
    rows = np.repeat((np.arange(height)+1).reshape((1,height)).T, width, axis=1)
    columns = np.repeat((np.arange(width)+1).reshape((1,width)), height, axis=0)

    xyz = pixel2world(columns, rows, (img + 1).copy(), u0, v0, fx, fy)
    gtx, gty, gtz = xyz[:,0], xyz[:,1], xyz[:,2]
    #assert ()
    mask1 = np.zeros((height, width))
    mask2 = np.zeros((height, width))
    mask3 = np.zeros((height, width))

    mask1[np.logical_and(gtx >= minu, gtx <= maxu)] = 1
    mask2[np.logical_and(gty >= minv, gty <= maxv)] = 1
    mask3[np.logical_and(gtz >= mind, gtz <= maxd)] = 1
    mask = np.logical_and(np.logical_and(mask1 ,mask2 ),mask3)

    bg = np.where(mask == 0)

    vals = np.ones(gtx.shape, np.bool)
    vals[bg[0], bg[1]] = False
    gtx_crop, gty_crop, gtz_crop = gtx[vals], gty[vals], gtz[vals]

    uvd = world2pixel(skeleton[:, 0], skeleton[:, 1], skeleton[:, 2], u0, v0, fx, fy)
    ptu, ptv, ptd = uvd[:,0], uvd[:,1], uvd[:,2]

    uvd = world2pixel(gtx_crop, gty_crop, gtz_crop, u0, v0, fx, fy)
    gtu, gtv, gtd = uvd[:,0], uvd[:,1], uvd[:,2]

    mean_u = ptu[center_joint]
    mean_v = ptv[center_joint]
    mean_z = skeleton[center_joint, 2]
    norm_size = bbsize / mean_z * fx

    # left, top, right, bottom, x1,y1,x2,y2
    bb2d = [max(0, np.round(ptu[center_joint]) - norm_size // 2), max(0, np.round(ptv[center_joint]) - norm_size // 2),
                min(np.round(ptu[center_joint]) + norm_size // 2, img.shape[1]), min(np.round(ptv[center_joint]) + norm_size // 2, img.shape[0])]
    bb2d = np.array(bb2d, np.int)

    ptu = (ptu - mean_u) / norm_size + 0.5
    ptv = (ptv - mean_v) / norm_size + 0.5
    ptd = (ptd - mean_z) / bbsize + 0.5

    gtu = (gtu - mean_u) / norm_size + 0.5
    gtv = (gtv - mean_v) / norm_size + 0.5
    gtd = (gtd - mean_z) / bbsize + 0.5

    gtu[np.where(gtu > 1)] = 1
    gtu[np.where(gtu < 0)] = 0
    gtv[np.where(gtv > 1)] = 1
    gtv[np.where(gtv < 0)] = 0
    gtd[np.where(gtd > 1)] = 1
    gtd[np.where(gtd < 0)] = 0

    newhand = np.ones((int(np.ceil(norm_size)), int(np.ceil(norm_size))), np.float) * -999

    for l in range(gtd.shape[0]):
        ind = [np.floor(max(0, (gtv[l] ) * norm_size)), np.floor(max(0, (gtu[l]) * norm_size))]
        newhand[int(ind[0]), int(ind[1])] = gtd[l]

    newhand[np.where(newhand == -999)] = 0

    joints2d_crop = np.zeros((21,3), np.float)
    joints2d_crop[:, 0] = ptu * norm_size
    joints2d_crop[:, 1] = ptv * norm_size
    joints2d_crop[:, 2] = ptd * bbsize

    if np.any(newhand > 0):
        success = True

    newhand = (newhand - 0.5) * bbsize + joints2d_crop[center_joint, 2]

    return newhand, joints2d_crop, bb2d, success


def pixel2world(u,v,d,u0,v0,fx,fy):
        x = ((u - u0) * d) / fx
        y = ((v - v0) * d) / fy
        return np.concatenate([x[:, np.newaxis], y[:, np.newaxis], d[:, np.newaxis]], axis=1)


def world2pixel(x,y,z,u0,v0,fx,fy):
    u = ((x * fx) / z) + u0
    v = ((y * fy) / z) + v0
    return np.concatenate([u[:, np.newaxis], v[:, np.newaxis], z[:, np.newaxis]], axis=1)


# paste rendering in the cropped space to original image space
def copypaste_crop2org(org_img, crop_img, center2d_org, center2d_crop, bbsize):
    # background set to 0 and unnormalize depth
    mask = crop_img > (bbsize - 1)  # zfar
    if len(mask) > 0:
        crop_img[mask] = 1500  # set background to zfar

    # find start/end indices to copy and paste
    success = False
    step_x, step_y = crop_img.shape[1]/2, crop_img.shape[0]/2
    y_start_crop, x_start_crop, y_end_crop, x_end_crop = 0, 0, crop_img.shape[0], crop_img.shape[1]
    top_offset = int(np.ceil(center2d_org[1] - step_y))
    left_offset = int(np.ceil(center2d_org[0] - step_x))
    bottom_offset = int(np.ceil(center2d_org[1] + step_y))
    right_offset = int(np.ceil(center2d_org[0] + step_x))
    y_start_org, x_start_org, y_end_org, x_end_org = top_offset, left_offset, bottom_offset, right_offset

    if top_offset < 0:
        y_start_crop = -1*top_offset
        y_start_org = 0
    if left_offset < 0:
        x_start_crop =  -1*left_offset
        x_start_org = 0
    if right_offset > org_img.shape[1] - 1:
        x_end_crop = crop_img.shape[1] - (right_offset - org_img.shape[1])
        x_end_org = org_img.shape[1]
    if bottom_offset > org_img.shape[0] - 1:
        y_end_crop = crop_img.shape[0] - (bottom_offset - org_img.shape[0])
        y_end_org = org_img.shape[0]
    # paste the crop into original image space
    if y_start_crop < y_end_crop and x_start_crop < x_end_crop:
        org_img[y_start_org: y_end_org, x_start_org: x_end_org] = crop_img[y_start_crop: y_end_crop, x_start_crop: x_end_crop]
        org_img[org_img != np.max(org_img)] += center2d_org[2] - center2d_crop[2]
        success = True
    return org_img, success

