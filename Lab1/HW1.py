import cv2
import numpy as np
import open3d as o3d
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

def Integral(N):
    
    # Integral from 0, 0
    dfdx, dfdy = 0, 0
    for i in range(image_row):
        dfdx += N[i][0][0]/(N[i][0][2] + 1e-8) * (image_row - i) / image_row 
        for j in range(image_col):
            dfdy -= N[i][j][1]/(N[i][j][2] + 1e-8) * (image_col - j) / image_col
            Z[i][j] -= (dfdx + dfdy)
    dfdx, dfdy = 0, 0
    for j in range(image_col):
        dfdy -= N[0][j][1]/(N[0][j][2] + 1e-8) * (image_col - j) / image_col
        for i in range(image_row):
            dfdx -= N[i][j][0]/(N[i][j][2] + 1e-8) * (image_row - i) / image_row 
            Z[i][j] += (dfdy + dfdx)

    # # Integral from 0, image_col-1
    # dfdx, dfdy = 0, 0
    # for i in range(image_row):
    #     dfdx -= N[i][image_col-1][0]/(N[i][image_col-1][2] + 1e-8) * (image_row - i) / image_row 
    #     for j in range(image_col-1, -1, -1):
    #         dfdy += N[i][j][1]/(N[i][j][2] + 1e-8) * j  / image_col
    #         Z[i][j] += (dfdx + dfdy)
    # dfdx, dfdy = 0, 0
    # for j in range(image_col-1, -1, -1):
    #     dfdy -= N[0][j][1]/(N[0][j][2] + 1e-8) *j / image_col
    #     for i in range(image_row):
    #         dfdx += N[i][j][0]/(N[i][j][2] + 1e-8) * (image_row - i) / image_row 
    #         Z[i][j] += (dfdy + dfdx)

    # # Integral from image_row-1, 0
    # dfdx, dfdy = 0, 0
    # for i in range(image_row-1, -1, -1):
    #     dfdx += N[i][0][0]/(N[i][0][2] + 1e-8) * i / image_row
    #     for j in range(image_col):
    #         dfdy -= N[i][j][1]/(N[i][j][2] + 1e-8) * (image_col - j) / image_col
    #         Z[i][j] += (dfdx + dfdy)
    # dfdx, dfdy = 0, 0
    # for j in range(image_col):
    #     dfdy += N[image_row-1][j][1]/(N[image_row-1][j][2] + 1e-8) * (image_col - j) / image_col
    #     for i in range(image_row-1, -1, -1):
    #         dfdx -= N[i][j][0]/(N[i][j][2] + 1e-8) * i / image_row 
    #         Z[i][j] += (dfdy + dfdx)

    # #Integral from image_row-1, image_col-1
    # dfdx, dfdy = 0, 0
    # for i in range(image_row-1, -1, -1):
    #     dfdx += N[i][image_col-1][0]/(N[i][image_col-1][2] + 1e-8) * i / image_row 
    #     for j in range(image_col-1, -1, -1):
    #         dfdy += N[i][j][1]/(N[i][j][2] + 1e-8) * j / image_col
    #         Z[i][j] += (dfdx + dfdy)
    # dfdx, dfdy = 0, 0
    # for j in range(image_col-1, -1, -1):
    #     dfdy += N[image_row-1][j][1]/(N[image_row-1][j][2] + 1e-8)* j / image_col
    #     for i in range(image_row-1, -1, -1):
    #         dfdx += N[i][j][0]/(N[i][j][2] + 1e-8) * i / image_row 
    #         Z[i][j] += (dfdy + dfdx)

    
    for i in range(image_row):
        for j in range(image_col):
            Z[i][j] /= (80) 
    return Z
def lsqr(A, b):
    return sp.linalg.lsqr(A.T@A, A.T@b)[0]

if __name__ == '__main__':
    tmp_pics = []
    test = 0
    testcase = ['bunny', 'star', 'venus', 'noisy_venus']
    for i in range(1,7):
        tmp_pics.append(read_bmp(f'./test/{testcase[test]}/pic{i}.bmp'))
    pictures = np.zeros((6, image_row * image_col))
    for i in range(6):
        pictures[i,:] = tmp_pics[i].flatten()
    light_sources = read_light(f'./test/{testcase[test]}/LightSource.txt')
    N = pseudo_inverse(light_sources, pictures)
    N = np.transpose(N)
    N = np.reshape(N, (image_row, image_col, 3))
    mask = np.zeros((image_row, image_col))
    for i in range(image_row):
        for j in range(image_col):
            if np.linalg.norm(N[i][j]) == 0:
                N[i][j] = np.array([0,0,0])
            else:
                N[i][j] = N[i][j] / np.linalg.norm(N[i][j])
                mask[i][j] = 1
    size = image_row * image_col

    M = sp.lil_matrix((2*size, size))
    Z = np.zeros((size, 1))
    # V = np.zeros((2 * size, 1))
    # for i in range(image_row - 1):
    #     for j in range(image_col - 1):
    #         idx = i * image_col + j
    #         if mask[i,j] == 0:
    #             continue
    #         M[idx*2, idx] = -1
    #         M[idx*2 + 1, idx] = -1
    #         M[idx*2, (i+1) * image_col + j] = 1
    #         M[idx*2+1, i * image_col + j+1] = 1
    #         V[2*idx] = -N[i][j][0]/N[i][j][2]
    #         V[2*idx+1] = -N[i][j][1]/N[i][j][2]
    # Z = lsqr(M, V)
    Z.resize((image_row, image_col))

    Z = Integral(N)
    
    # normalize Z
    Zmin = np.min(Z)
    Zmax = np.max(Z)
    for i in range(image_row):
        for j in range(image_col):
            Z[i][j] = (Z[i][j] - Zmin) / (Zmax - Zmin) * 20

    # post processing
    kernel = np.ones((5,5),np.float32)/25
    Z = cv2.filter2D(Z*mask, -1, kernel)
   
    filepath = f'./test/{testcase[test]}/depth.ply'
    normal_visualization(N)
    mask_visualization(mask)
    depth_visualization(Z*mask)
    save_ply(Z*mask,filepath)
    show_ply(filepath)

    # showing the windows of all visualization function
    plt.show()