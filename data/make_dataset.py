import scipy
import scipy.io as sio
from scipy import spatial
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import PIL.Image as Image
from matplotlib import cm as CM

def gaussian_filter_density(img, points):

    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    gt_count = len(points)
    if gt_count == 0:
        return density

    leafsize = 2048
    # build kdtree
    tree = spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=4)

    print('generate density...')
    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2. / 2.  # case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    print('done.')
    return density


# test code
if __name__ == "__main__":

    dirname = os.path.dirname(__file__)
    root = os.path.join(dirname, 'ShanghaiTech')

    part_A_train = os.path.join(root, 'part_A', 'train_data', 'images')
    part_A_test = os.path.join(root, 'part_A', 'test_data', 'images')
    # part_B_train = os.path.join(root, 'part_B_final', 'train_data', 'images')
    # part_B_test = os.path.join(root, 'part_B_final', 'test_data', 'images')
    path_sets = [part_A_train, part_A_test]
    #path_sets = [part_B_train, part_B_test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        img = plt.imread(img_path)
        mat_path = img_path.replace('.jpg', '.mat').replace('images', 'ground-truth').replace('IMG_', 'GT_IMG_')
        mat = sio.loadmat(mat_path, appendmat=False)
        k = np.zeros((img.shape[0], img.shape[1]))
        points = mat["image_info"][0, 0][0, 0][0]
        print("Number of points:", len(points))
        k = gaussian_filter_density(img, points)
        with open(img_path.replace('.jpg','.npy').replace('images','ground-truth-h5_1'), 'wb') as f:
            np.save(f, k)

    plt.imshow(Image.open(img_paths[0]))

    gt_file = np.load(img_paths[0].replace('.jpg','.npy').replace('images','ground_truth-h5_1'))
    plt.imshow(gt_file, cmap=CM.jet)