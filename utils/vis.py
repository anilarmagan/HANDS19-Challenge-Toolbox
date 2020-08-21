# author: Anil Armagan
# contact: a.armagan@imperial.ac.uk
# date: 20/07/2019
import cv2
import numpy as np

# blend depth image with synth image
# img1 & img2: WxH depth images
# returns: canvas WxHx3
def blend_frames(img1, img2, min_depth=0, max_depth=1500, min_color=50, show=False, show_name=0, joints2d_gt=None, joints2d_ren=None):
    # img 1 - real
    img = img1[:,:,np.newaxis]

    mask = np.logical_and(img > min_depth, img < max_depth)
    vals = img[mask]
    if len(vals) != 0:
        min_v = np.min(vals)
        max_v = np.max(vals)
    else:
        min_v = 0
        max_v = 300
    step = (255. - min_color) / (max_v - min_v)
    img[~mask] = 0
    img[mask] = (img[mask] - min_v) * step + min_color
    rgb_img1 = np.repeat((img/np.max(img)*255).astype(np.uint8), 3, axis=2)
    rgb_img1[:,:, 1:] = 0

    # img 2 - synthetic
    img = img2[:, :, np.newaxis]
    mask = np.logical_and(img > min_depth, img < max_depth)
    vals = img[mask]

    min_v = np.min(vals)
    max_v = np.max(vals)
    step = (255. - min_color) / (max_v - min_v)
    img[~mask] = 0
    img[mask] = (img[mask] - min_v) * step + min_color
    rgb_img2 = np.repeat(img.astype(np.uint8), 3, axis=2)
    rgb_img2[:, :, :2] = 0

    canvas = cv2.addWeighted(rgb_img1, 1, rgb_img2, 1, 0).astype(np.uint8)

    if joints2d_gt is not None:
        for i in range(joints2d_gt.shape[0]):
            cv2.circle(canvas, (int(joints2d_gt[i,0]), int(joints2d_gt[i,1])), 3, (255, 0, 0), -1)
    if joints2d_ren is not None:
        for i in range(joints2d_ren.shape[0]):
            cv2.circle(canvas, (int(joints2d_ren[i,0]), int(joints2d_ren[i,1])), 3, (0, 0, 255), -1)
    if show:
        cv2.imshow('blended img: {}'.format(show_name), canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return canvas

# draw on depth image, returns rgb
# joints2d: 21x3
# img: WxH
# returns: vis_img WxHx3
def draw_joints(img, joints2d=None, color=(0,0,255)):
    vis_img = img.copy()
    vis_img[vis_img > 0] = (
                vis_img[vis_img > 0] - np.min(vis_img[vis_img > 0]))
    vis_img = vis_img / np.max(vis_img) * 255
    vis_img = np.repeat(vis_img[:, :, np.newaxis], axis=2, repeats=3).astype(np.uint8)

    # plot joints
    if joints2d is not None:
        for i in range(joints2d.shape[0]):
            cv2.circle(vis_img, (int(joints2d[i, 0]), int(joints2d[i, 1])), 3, color, -1)
    return vis_img

# img: RGB input, WxHx3
# bb2d: [x1, y1, x2, y2] -> top, left, bottom, right
def draw_bbox2d(img, bb2d, color=(0,0,255)):
    return cv2.rectangle(img, (bb2d[0], bb2d[1]), (bb2d[2], bb2d[3]),color,3)

# this function assumes annotations are in Bighand ordering
# author: Shreyas Hampali
def showHandJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=2):
    # index, middle, ring, pinky, thumb
    jointConns = [[0, 2, 9, 10, 11], [0, 3, 12, 13, 14], [0, 4, 15, 16, 17], [0, 5, 18, 19, 20], [0, 1, 6, 7, 8]]

    imgIn[imgIn == 1500] = 0 # make background 0. if input depth

    if len(imgIn.shape) < 3:
        imgIn = imgIn[:, :, np.newaxis]
        imgIn = np.repeat(imgIn, 3, axis=2)

    jointColsGt = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]
    jointColsEst = []
    for col in jointColsGt:
        newCol = (col[0] + col[1] + col[2]) / 3
        jointColsEst.append((newCol, newCol, newCol))
    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    if np.max(imgIn) > 255:
        imgIn = imgIn.astype(np.float32) / np.max(imgIn) * 255.
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j + 1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC, 0]), int(gt[jntC, 1])), (int(gt[jntN, 0]), int(gt[jntN, 1])),
                         jointColsGt[i], lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC, 0]), int(est[jntC, 1])), (int(est[jntN, 0]), int(est[jntN, 1])),
                         jointColsEst[i], lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img

def showPalm(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=2):
    imgIn[imgIn == 1500] = 0 # make background 0.
    if len(imgIn.shape) < 3:
        imgIn = imgIn[:, :, np.newaxis]
        imgIn = np.repeat(imgIn, 3, axis=2)
    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)
    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
        cv2.circle(img, (int(gt[0, 0]), int(gt[0, 1])), radius=2, color=(0,0,255), thickness=lineThickness)
    if estIn is not None:
        est = estIn.copy() * upscale
        cv2.circle(img, (int(est[0, 0]), int(est[0, 1])), radius=2, color=(255,0,0), thickness=lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)
    return img