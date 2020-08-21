# author: Anil Armagan
# contact: a.armagan@imperial.ac.uk
# date: 20/07/2019
# description: this file is provided for HANDS19 Challenge to render a synthetic image using the MANO model and the provided prarameters.
# usage: python3 visualize_task1_task2.py --task-id=1 --frame-id=0 --use-mano=1 --mano-model-path=./MANO_RIGHT.pkl

import numpy as np
import cv2
from os.path import join, exists
import argparse

from utils.MANO_SMPL import MANO_SMPL
from utils.renderer import HandRenderer
from utils.crop import crop, pixel2world, world2pixel, copypaste_crop2org
from utils.reader import read_image, read_anno
from utils.vis import blend_frames, showHandJoints, draw_bbox2d

# BigHand camera parameters
u0 = 315.944855
v0 = 245.287079
fx = 475.065948
fy = 475.065857
bbsize = 300.0  # mm
center_joint_id = 3  # Middle finger MCP joint in BigHand indexing


# make a few checks for the arguments and paths
def check_args(args):
    if args.task_id == 1:
        assert (args.frame_id < 175951),'Invalid frame id for Task 1!' # 175951 images in task1
    elif args.task_id == 2:
        assert (args.frame_id < 45212),'Invalid frame id for Task 2!' # 45212 images in task2
    else:
        assert (args.task_id not in [1, 2]), 'Task-id should be one of the two tasks, [1,2]!'

    if args.use_mano:
        assert(exists(args.mano_model_path)), "MANO model file does not exist at %s. Please download the model from official MANO page." % args.mano_model_path
        assert (exists(args.mano_anno_path)), "Fitted MANO model annotations doesn't exist at %s." % args.mano_anno_path
    assert(exists(args.joint_anno_path)), "3D joint ground truth annotations doesn't exist at %s." % args.joint_anno_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HANDS19 MANO Renderer')
    parser.add_argument('--task-id', type=int, default=1, help='Task id of the challenge you want to see the rendering.  1, 2 or 3 for HANDS19')
    parser.add_argument('--frame-id', type=int, default=0, help='Frame id of the corresponding image, eg. frame_id=0 -> IMG_D00000000.png')
    parser.add_argument('--use-mano', type=int, default=1, help='Use MANO model for synthetic rendering or no rendering ')
    parser.add_argument('--mano-model-path', default='../MANO_RIGHT.pkl', help='Path to mano .pkl model file.')
    args = parser.parse_args()

    args.joint_anno_path = '../Task%d/training_joint_annotation.txt' % args.task_id
    if args.use_mano:
        args.mano_anno_path = '../Task%d/training_mano_annotation.txt' % args.task_id
    args.frame_root_path = '../Task%d/training_images' % args.task_id
    check_args(args)

    # read annotations: joints and mano
    frame_name, annotations = read_anno(args)
    joints3d_anno = annotations[0]

    # create MANO object, read model and initialize
    if args.use_mano:
        mano = MANO_SMPL(args.mano_model_path, task_id=args.task_id)
        # parse mano annotations
        mano_anno = annotations[1]
        mano_cam = mano_anno[:4][np.newaxis]
        mano_quat = mano_anno[4: 4 + 4][np.newaxis]
        mano_art = mano_anno[4 + 4: 4 + 4 + 45][np.newaxis]
        mano_shape = mano_anno[4 + 4 + 45:][np.newaxis]

    # read image
    frame_path = join(args.frame_root_path, frame_name)
    assert(exists(frame_path)), "Frame doesn't exist at %s" % frame_path
    img = read_image(frame_path)

    # xyz to uvd
    joints3d_anno = joints3d_anno.reshape(21,3)
    joints2d_anno = world2pixel(joints3d_anno[:,0], joints3d_anno[:,1], joints3d_anno[:,2], u0, v0, fx, fy)

    # crop input image with 3d gt joint annotations.
    # cropping is done by fitting a 3d bounding box of size bbsize around the MCP joint of the middle finger.
    cropped_img, joints2d_anno_crop, bb2d, success_crop = crop(img, joints3d_anno, u0=u0, v0=v0, fx=fx, fy=fy, bbsize=bbsize, center_joint=center_joint_id, offset=30)

    crop_img_w = cropped_img.shape[1]
    crop_img_h = cropped_img.shape[0]

    print('Joints 2D GT', joints2d_anno)
    print('Joints 3D GT', joints3d_anno)

    vis_cropped = showHandJoints(cropped_img, joints2d_anno_crop)
    vis_img = showHandJoints(img, joints2d_anno)
    cv2.imshow('Cropped Hand Image', vis_cropped)
    cv2.imshow('Input Image', vis_img)

    # prepare renderer, rendered image is same size as the cropped image
    if args.use_mano:
        renderer = HandRenderer(faces=mano.faces)
        renderer.init_buffers(image_w=crop_img_w, image_h=crop_img_h)

        # get mano vertices and joints with the current parameters
        vertices, joints_normed_ren = mano.get_mano_vertices(quat_or_euler=mano_quat, pose=mano_art, shape=mano_shape, cam=mano_cam)

        joints2d_ren_crop = joints_normed_ren[0].copy()
        joints2d_ren_crop[:, 0] *= crop_img_w
        joints2d_ren_crop[:, 1] *= crop_img_h
        joints2d_ren_crop[:, 2] *= bbsize

        # align cropped joints to original image space
        joints2d_ren_org = joints2d_ren_crop.copy()
        joints2d_ren_org[:, 0] += joints2d_anno[center_joint_id, 0] - crop_img_h // 2 # align with cropped center = (0,0)
        joints2d_ren_org[:, 1] += joints2d_anno[center_joint_id, 1] - crop_img_w // 2
        joints2d_ren_org[:, 2] += joints2d_anno[center_joint_id, 2] - joints2d_ren_crop[center_joint_id, 2] # recenter around center joint's depth

        # get xyz
        joints3d_ren_org = pixel2world(joints2d_ren_org[:, 0], joints2d_ren_org[:, 1], joints2d_ren_org[:, 2], u0, v0, fx, fy)

        # rendering
        rendered_img_crop = renderer.render_mano(vertices).copy()

        # paste the rendered image back into original image resolution.
        # and rearrange the depth values wrt center joint's depth
        # cropped image could have been obtained with padding, get the valid area
        rendered_img_org = np.ones(img.shape, img.dtype)*1500 #zFar=1500mm 
        rendered_img_org, success_paste = copypaste_crop2org(org_img=rendered_img_org, crop_img=rendered_img_crop,
                                                             center2d_org=joints2d_anno[center_joint_id], center2d_crop=joints2d_ren_crop[center_joint_id],
                                                             bbsize=bbsize)

        print('Joints 2D MANO', joints2d_ren_org)
        print('Joints 3D MANO', joints3d_ren_org)

        # visualize
        # blend real and synthetic images and plot joints
        if args.use_mano:
            canvas = blend_frames(img.copy(), rendered_img_org.copy(), joints2d_gt=joints2d_anno, joints2d_ren=joints2d_ren_org)

        # draw the bounding box from cropping
        canvas = draw_bbox2d(canvas, bb2d, color=(0, 0, 255))

        # draw joints
        vis_rendered_org = showHandJoints(rendered_img_org, joints2d_ren_org)
        vis_rendered_crop = showHandJoints(rendered_img_crop, joints2d_ren_crop)

        # show
        cv2.imshow('Rendered MANO', vis_rendered_org)
        cv2.imshow('Rendered MANO Crop', vis_rendered_crop)
        cv2.imshow('Blended Image Real+Synth', canvas)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Visualized Task#%d - %s" % (args.task_id, frame_name))


