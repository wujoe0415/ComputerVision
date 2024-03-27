import cv2
import numpy as np
import open3d as o3d
import sys
import matplotlib.pyplot as plt
import scipy.sparse as sp
image_row = 120
image_col = 120

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.savefig("Mask")

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')
    plt.savefig("Normal")

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.savefig("Depth")

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image_row, image_col = image.shape
    return image

def read_light(filepath):
    L = np.zeros((6,3))
    file = open(filepath,'r')
    for i in range(6):
        line = file.readline()
        L[i] = list(map(float, line.split()[1][1:-1].split(',')))
        L[i] = L[i] / np.linalg.norm(L[i])
    return L

def pseudo_inverse(L, I):
    KN = (np.linalg.inv(np.transpose(L)@ L)) @ np.transpose(L) @ I
    return KN

# Enhance the depth map by adding the dot product of normal vector and light source
def enhance_depth(N, Z):
    image_row, image_col, _ = N.shape
    enhanced_Z = np.copy(Z)
    for i in range(image_row):
        for j in range(image_col):
            if np.linalg.norm(N[i][j]) != 0:
                enhanced_Z[i][j] += np.dot(N[i][j], np.array([0, 0, 1])) / np.linalg.norm(N[i][j])
    return enhanced_Z

# Remove Gaussian Noise
# Reference: https://stackoverflow.com/questions/62042172/how-to-remove-noise-in-image-opencv-python
def remove_noise(image):
    # 1. Morphological Transformations
    # 2. Divide the original image by the morphological transformed image
    # 3. Threshold the divided image
    # 4. Median Blur the thresholded image
    se=cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
    bg=cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
    out_gray=cv2.divide(image, bg, scale=255)
    out_binary=cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU )[1] 
    tempmask = cv2.medianBlur(np.float32(out_binary), 5)

    return image*(np.float32(tempmask)/255)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please provide the parameter (venus or noisy_venus)")
        sys.exit(1)
    
    parameter = sys.argv[1]
    
    if not parameter == 'bunny' and not parameter == 'star' and not parameter == 'venus' and not parameter == 'noisy_venus':
        print("Invalid parameter")
        sys.exit(1)
    
    tmp_pics = []
    for i in range(1,7):
        tmp_pics.append(read_bmp(f'./test/{parameter}/pic{i}.bmp'))
    pictures = np.zeros((6, image_row * image_col))
    for i in range(6):
        if parameter == 'noisy_venus':
            tmp_pics[i] = remove_noise(tmp_pics[i])
        pictures[i,:] = cv2.medianBlur(np.float32(tmp_pics[i]), 5).flatten()
        #cv2.imwrite(f'./test/{parameter}/pic{i}_filtered.bmp', pictures[i].reshape(image_row, image_col))

    light_sources = read_light(f'./test/{parameter}/LightSource.txt')

    N = pseudo_inverse(light_sources, pictures)
    N = np.transpose(N)
    N = np.reshape(N, (image_row, image_col, 3))
    N = cv2.medianBlur(np.float32(N), 5)
    
    mask = np.zeros((image_row, image_col))
    for i in range(image_row):
        for j in range(image_col):
            if np.linalg.norm(N[i][j]) == 0:
                N[i][j] = np.array([0,0,0])
            else:
                N[i][j] = N[i][j] / np.linalg.norm(N[i][j])
                mask[i][j] = 1
    Z = np.zeros((image_row, image_col))
    
    Z = enhance_depth(N, Z)
    
    # standardize Z
    Zmean = np.mean(Z)
    Zstd = np.std(Z)
    Z = (Z - Zmean) / Zstd * 10

    # post processing
    kernel = np.ones((5,5),np.float32)/25
    Z = cv2.filter2D(Z*mask, -1, kernel)
   
    filepath = f'./test/{parameter}/depth.ply'
    normal_visualization(N)
    mask_visualization(mask)
    depth_visualization(Z*mask)
    save_ply(Z*mask,filepath)
    show_ply(filepath)

    # showing the windows of all visualization function
    plt.show()