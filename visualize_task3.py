# author: Shreyas Hampali
# contact: hampali@icg.tugraz.at
# modifier for HANDS19: Anil Armagan
# contact: aarmagan@ic.ac.uk

from os.path import join, exists
import numpy as np
import cv2
import argparse
from utils.MANO_SMPL import MANO_SMPL

from utils.reader import read_anno
from utils.vis import showHandJoints, showPalm

from opendr.renderer import DepthRenderer
from opendr.camera import ProjectPoints

# HO-3D camera parameters
fx = 617.343
fy = 617.343
u0 = 312.42
v0 = 241.42
img_w = 640
img_h = 480

def decodeDepthImg(inFileName, dsize=None):
    '''
    Decode the depth image to depth map in METERS
    :param inFileName: input file name
    :return: depth map (float) in meters
    '''
    depthScale = 0.00012498664727900177
    depthImg = cv2.imread(inFileName)
    if dsize is not None:
        depthImg = cv2.resize(depthImg, dsize, interpolation=cv2.INTER_CUBIC)

    dpt = depthImg[:, :, 0] + depthImg[:, :, 1] * 256
    dpt = dpt * depthScale
    return dpt

def project3DPoints(camMat, pts3D, isOpenGLCoords=True):
    '''
    Function for projecting 3d points to 2d
    :param camMat: camera matrix
    :param pts3D: 3D points in mm. Nx3
    :param isOpenGLCoords: If True, hand/object along negative z-axis. If False hand/object along positive z-axis
    :return:
    '''
    assert pts3D.shape[-1] == 3
    assert len(pts3D.shape) == 2
    # convert mm to meter
    pts3D /= 1000.

    # ho3d annotations are in OpenGL coordinate system
    if isOpenGLCoords:
        coordChangeMat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        pts3D = pts3D.dot(coordChangeMat.T)

    projPts = pts3D.dot(camMat.T)
    projPts = np.stack([projPts[:,0]/projPts[:,2], projPts[:,1]/projPts[:,2]],axis=1)

    assert len(projPts.shape) == 2

    return projPts

def showObjJoints(imgIn, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying object annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''
    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (255,255,0)
    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst  = (newCol, newCol, newCol)

    # draws lines connected using jointConns
    img = np.zeros((imgIn.shape[0], imgIn.shape[1], imgIn.shape[2]), dtype=np.uint8)
    img[:, :, :] = (imgIn).astype(np.uint8)

    img = cv2.resize(img, (upscale * imgIn.shape[1], upscale * imgIn.shape[0]), interpolation=cv2.INTER_CUBIC)
    if gtIn is not None:
        gt = gtIn.copy() * upscale
    if estIn is not None:
        est = estIn.copy() * upscale

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j+1]
            if gtIn is not None:
                cv2.line(img, (int(gt[jntC,0]), int(gt[jntC,1])), (int(gt[jntN,0]), int(gt[jntN,1])), jointColsGt, lineThickness)
            if estIn is not None:
                cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    if filename is not None:
        cv2.imwrite(filename, img)

    return img


if __name__ == '__main__':
    # parse the arguments
    parser = argparse.ArgumentParser(description='HANDS19 - Task#3 HO-3D Visualization')
    parser.add_argument('--frame-id', type=int, default=0, required=False,
                        help='Frame id of the corresponding image, eg. frame_id=0 -> IMG_C00000000.png')
    parser.add_argument('--train-test', type=int, default=0, required=False,
                        help='Visualize training or test set of task3 --train_test==0/1')
    parser.add_argument('--mano-model-path', default='../MANO_RIGHT.pkl', help='Path to mano model file.')
    args = parser.parse_args()
    args.task_id = 3

    if args.train_test == 0:# if train set
        args.joint_anno_path = '../Task3/training_joint_annotation.txt'
        args.object_anno_path = '../Task3/training_object_annotation.txt'
        args.mano_anno_path = '../Task3/training_mano_annotation.txt'
        args.frame_root_path = '../Task3/training_images'
        args.depth_frame_root_path = '../Task3/release/training_images_depth'
        args.models_path = '../Task3/object_models'
    else: # if test
        args.palm_anno_path = '../Task3/test_palm_annotation.txt'
        args.frame_root_path = '../Task3/test_images'

    # read annotations and images
    frame_name, annotations = read_anno(args)
    frame_path = join(args.frame_root_path, frame_name)
    print('frame_path',frame_path)
    assert (exists(frame_path)), "RGB frame doesn't exist at %s" % frame_path
    img = cv2.imread(frame_path)

    if args.train_test == 0:
        depth_frame_path = join(args.depth_frame_root_path, frame_name.replace('image_C', 'image_D'))
        assert (exists(depth_frame_path)), "Depth frame doesn't exist at %s" % depth_frame_path
        depth = decodeDepthImg(depth_frame_path)

    # camera properties
    camMat = np.array([[fx, 0., u0], [0., fy, v0], [0., 0., 1.]])

    # hand joint annotations
    if args.train_test == 0:
        joints3d_anno = annotations[0]
        joints3d_anno = joints3d_anno.reshape(21, 3)

        # object pose annotations
        obj_anno = annotations[2]
        obj_rot = obj_anno[:3]
        obj_trans = obj_anno[3:6]
        obj_id = annotations[3]

        # mano annotations
        mano_anno = annotations[1]
        mano_cam = np.zeros((1, 4), np.float32)
        mano_cam[0, 0] = 1
        mano_cam[0, 1:] = mano_anno[48:48 + 3]
        mano_euler = mano_anno[:3][np.newaxis]
        # print(mano_euler);assert()
        mano_art = mano_anno[3:48][np.newaxis]
        mano_shape = mano_anno[48 + 3:48 + 3 + 10][np.newaxis]

        mano = MANO_SMPL(args.mano_model_path, task_id=args.task_id)
        vertices, joints3d_mano = mano.get_mano_vertices(quat_or_euler=mano_euler, pose=mano_art, shape=mano_shape, cam=mano_cam)

        # apply pose on object corners
        objCornersFilename = join(args.models_path, obj_id, 'corners.npy')
        objCorners = np.load(objCornersFilename)
        # transform the object corners
        objCornersTrans = np.matmul(objCorners, cv2.Rodrigues(obj_rot)[0].T) + obj_trans

        # project 3D hand keypoints to image place
        handKps = project3DPoints(camMat, joints3d_anno, isOpenGLCoords=True)
        handKps_mano = project3DPoints(camMat, joints3d_mano[0], isOpenGLCoords=True)
        objKps = project3DPoints(camMat, objCornersTrans, isOpenGLCoords=True)

        # draw 2D hand keypoints and object corners
        imgAnno_gt = showHandJoints(img, handKps.copy(), lineThickness=2)
        imgAnno_gt = showObjJoints(imgAnno_gt, objKps.copy(), lineThickness=2)

        imgAnno_mano = showHandJoints(img, handKps_mano.copy(), lineThickness=2)
        imgAnno_mano = showObjJoints(imgAnno_mano, objKps.copy(), lineThickness=2)

        # render mano vertices
        rn = DepthRenderer()
        rn.camera = ProjectPoints(v=vertices[0], rt=np.zeros(3), t=np.zeros(3), f=np.array([fx, fy]), c=np.array([u0, v0]), k=np.zeros(5))
        rn.frustum = {'near': 0.01, 'far': 1.5, 'width': img_w, 'height': img_h}
        rn.set(v=vertices[0], f=mano.faces, bgcolor=np.zeros(3))
        # render
        mano_rendered = rn.r
        # show
        cv2.imshow('Depth Image', depth)
        cv2.imshow('MANO Rendered', mano_rendered)
        cv2.imshow('Image GT Annotation', imgAnno_gt)
        cv2.imshow('Image MANO Annotation', imgAnno_mano)

    else:
        palmAnno = annotations[-1][np.newaxis]
        palmKp = project3DPoints(camMat, palmAnno, isOpenGLCoords=True)
        imgPalm_gt = showPalm(img.copy(), palmKp.copy(), lineThickness=2)
        cv2.imshow('Test Image Palm Annotation', imgPalm_gt)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
