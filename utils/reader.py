# modifier: Anil Armagan
# contact: a.armagan@imperial.ac.uk
import numpy as np
import cv2

def read_image(img_path):
    # read depth image
    img = cv2.imread(img_path, 2).astype(np.float32)
    return img


# read file line by line
# args is a list of paths to read from
# [joint anno path, mano anno path, object anno path]
# returns: img_name, name of the frame
#          annos, list of read annotations
def read_anno(args):
    paths = []
    if hasattr(args, 'joint_anno_path'):
        paths = [args.joint_anno_path]
    if hasattr(args, 'mano_anno_path'):
        paths.append(args.mano_anno_path)
    if hasattr(args, 'object_anno_path'):
        paths.append(args.object_anno_path)
    if hasattr(args, 'palm_anno_path'):
        paths.append(args.palm_anno_path)
    frame_id = args.frame_id
    annos = []
    for path in paths:
        with open(path, 'r') as myfile:
            for i in range(frame_id):
                myfile.readline()
            anno = myfile.readline()
        if hasattr(args, 'palm_anno_path') and path == args.palm_anno_path:
            anno = anno.split(' ')
        else:
            anno = anno.split('\t')
        if(anno[-1] == '\n'):
            anno = anno[:-1]
        img_name = anno[0]

        if args.task_id==3 and hasattr(args, 'mano_anno_path') and path == args.object_anno_path:
            object_id = anno[1]
            anno = np.array(anno[2:]).astype(np.float32)
            annos.append(anno)
            annos.append(object_id)
        else:
            anno = np.array(anno[1:]).astype(np.float32)
            annos.append(anno)

    return img_name, annos

# read all row in annotation file
def read_all(path, dtype=np.float, cols=None):
    if cols is None:
        text = np.loadtxt(path, usecols=(np.arange(1,64)), dtype=dtype)
    else:
        text = np.loadtxt(path, usecols=cols,dtype=dtype)
    return text