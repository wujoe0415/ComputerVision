import cv2
import numpy as np
import random
import math
import sys
import os
import matplotlib.pyplot as plt

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)#CyclindricalProjection(cv2.imread(path))
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def KNN_match(img1, kps1, des1, img2, kps2, des2, k):
    matches = []
    for i in range(len(des1)):
        dists = []
        for j in range(len(des2)):
            dist = np.linalg.norm(des1[i] - des2[j])
            dists.append((j, dist))
        dists.sort(key=lambda x: x[1])
        # lowe_ratio_test
        if dists[0][1] < 0.8 * dists[1][1]:
            matches.append((i, dists[0][0]))

    result = np.zeros((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    result[:img1.shape[0], :img1.shape[1]] = img1
    result[:img2.shape[0], img1.shape[1]:] = img2

    for match in matches:
        src_pt = np.float32([kps1[match[0]].pt]).reshape(-1, 1, 2)
        dst_pt = np.float32([kps2[match[1]].pt]).reshape(-1, 1, 2)
        dst_pt[0][0][0] += img1.shape[1]  # Adjust x-coordinate for the second image
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.line(result, tuple(src_pt[0][0].astype(int)), tuple(dst_pt[0][0].astype(int)), color, thickness=1)

    # Save the result image
    print('there')
    cv2.imwrite(f'match_result{random.randint(0, 100)}.jpg', result)
    print('finish')
    return matches

def expected_Homography(matches, kps1, kps2):
    src_pts = np.float32([kps1[m[0]].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps2[m[1]].pt for m in matches]).reshape(-1, 1, 2)
    expected_H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return expected_H

def Homography(pairs, kps1, kps2):
    A = []
    for pair in pairs:
        src_pt = np.float32([kps1[pair[0]].pt]).reshape(-1, 1, 2)
        dst_pt = np.float32([kps2[pair[1]].pt]).reshape(-1, 1, 2)
        x1, y1 = src_pt[0][0][0], src_pt[0][0][1]
        x2, y2 = dst_pt[0][0][0], dst_pt[0][0][1]

        A.append([-x1, -y1, -1, 0, 0, 0, x1*x2, y1*x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, x1*y2, y1*y2, y2])
    A.append([0, 0, 0, 0, 0, 0, 0, 1, 1])
    A = np.array(A)
    A = A.reshape(-1, 9)  # Reshape A to have a homogeneous shape
    U, S, V = np.linalg.svd(A)
    H = V[-1].reshape(3, 3)
    H /= H[2, 2] # Normalize H
    return H

def RANSAC(matches, kps1, kps2):
    #return expected_Homography(matches, kps1, kps2)
    max_inliers = 0
    threshold = 1
    best_H = None
    iteration = 1000

    for i in range(iteration):
        matches = random.sample(matches, k = len(matches))
        H = Homography(matches[:4], kps1, kps2)
        inliers = 0
        for j in range(len(matches)):
            src_pt = np.float32([kps1[matches[j][0]].pt]).reshape(-1, 1, 2)
            dst_pt = np.float32([kps2[matches[j][1]].pt]).reshape(-1, 1, 2)
            dst_pt_ = cv2.perspectiveTransform(src_pt, H)
            if np.linalg.norm(dst_pt - dst_pt_) < threshold:
                inliers += 1
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H
    #print("Best H: ", best_H)
    return best_H

def stitch_image(src, dest):

    image1, img_gray1 = src
    image2, img_gray2 = dest
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_gray1, None)
    kp2, des2 = sift.detectAndCompute(img_gray2, None)
    
    matches = KNN_match(image1, kp1, des1, image2, kp2, des2, 2)
    holo = RANSAC(matches, kp1, kp2).astype(np.float32)
    corners = np.array([[0, 0], [image1.shape[1], 0], [0, image1.shape[0]], [image1.shape[1], image1.shape[0]]], dtype=np.float32)
    img1_homo_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), holo)
    _minx = np.min(img1_homo_corners[:, :, 0])
    _maxx = np.max(img1_homo_corners[:, :, 0])
    _miny = np.min(img1_homo_corners[:, :, 1])
    _maxy = np.max(img1_homo_corners[:, :, 1])
    _deltaW =  int(abs(min(0, _minx)))
    _deltaH =  int(abs(min(0, _miny)))
    dWidth = int(max(image2.shape[1], _maxx) - min(0, _minx))
    dHeight = int(max(image2.shape[0], _maxy) - min(0, _miny))
    affine = np.array([[1, 0, _deltaW], [0, 1, _deltaH], [0, 0, 1]]).astype(np.float32)
    holo = np.matmul(affine, holo)
    holo = holo.astype(np.float32)  # Convert holo to the correct data type

    result = cv2.warpPerspective(image1, holo, (dWidth, dHeight))
    tmp = cv2.warpPerspective(image1, holo, (5000, 5000))
    _center = cv2.warpPerspective(image2, affine, (dWidth, dHeight))
    
    rand = random.randint(0, 100)
    linearBlending = LinearBlending(result, _center)
    cv2.imwrite(f'result{rand}.jpg', result)
    cv2.imwrite(f'_center{rand}.jpg', _center)
    # cv2.imwrite(f'tmp{rand}.jpg', tmp)
    cv2.imwrite(f'Blended_Image{rand}.jpg', linearBlending)
        
    return linearBlending

def stitch_images(images, img_grays):
    centerImageIndex = int(len(images) / 2 + 0.5)
    leftImages = list(images[:centerImageIndex])
    leftGrays = list(img_grays[:centerImageIndex])
    rightImages = list(images[centerImageIndex - 1:])
    rightGrays = list(img_grays[centerImageIndex - 1:])
    # rightImages.reverse()
    # rightGrays.reverse()

    while len(leftImages) > 1:
        dest = (leftImages.pop(), leftGrays.pop())
        src = (leftImages.pop(), leftGrays.pop())
        left_pano = stitch_image(src, dest)
        left_pano_gray = cv2.cvtColor(left_pano, cv2.COLOR_BGR2GRAY)
        leftImages.append((left_pano, left_pano_gray))
    while len(rightImages) > 1:
        src = (rightImages.pop(), rightGrays.pop())
        dest = (rightImages.pop(), rightGrays.pop())
        right_pano = stitch_image(src, dest)
        right_pano_gray = cv2.cvtColor(right_pano, cv2.COLOR_BGR2GRAY)
        rightImages.append((right_pano, right_pano_gray))
        
    src = (leftImages[0][0], leftImages[0][1])
    dest = (rightImages[0][0], rightImages[0][1])
    if leftImages[0][0].shape[1] >= rightImages[0][0].shape[1]:
        src = (rightImages[0][0], rightImages[0][1])
        dest = (leftImages[0][0], leftImages[0][1])
    result = stitch_image(src, dest)
    return result

def LinearBlending(img1, img2):
    img1_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    img2_mask = np.zeros(img2.shape[:2], dtype=np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if np.all(img1[i, j] != 0):
                img1_mask[i, j] = 1
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if np.all(img2[i, j] != 0):
                img2_mask[i, j] = 1
    overlap_mask = np.zeros(img1.shape[:2], dtype=np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            overlap_mask[i, j] = img1_mask[i, j] * img2_mask[i, j]
    alpha_mask = np.zeros(img1.shape[:2], dtype=np.float32)
    for i in range(img1.shape[0]):
        minIndex = maxIndex = -1
        for j in range(img1.shape[1]):
            if (overlap_mask[i, j] == 1 and minIndex == -1):
                minIndex = j
            if (overlap_mask[i, j] == 1):
                maxIndex = j
        if (minIndex == maxIndex):
            continue

        decrease_step = 1 / (maxIndex - minIndex)
        middleIdx = int((maxIndex + minIndex) / 2)
        for j in range(minIndex, middleIdx):
            alpha_mask[i, j] = 1 - (j - minIndex) * decrease_step
    
    # cv2.imwrite('alpha.jpg', alpha_mask * 255)
    result = np.zeros(img1.shape, dtype=np.uint8)
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (overlap_mask[i, j] != 0):
                result[i, j] = overlap_mask[i, j] * (alpha_mask[i, j] * img1[i, j] + (1 - alpha_mask[i, j]) * img2[i, j])
            elif (img1_mask[i, j] == 1):
                result[i, j] = img1[i, j]
            elif (img2_mask[i, j] == 1):
                result[i, j] = img2[i, j]
    
    return result

def CyclindricalProjection(img):
    h, w = img.shape[:2]
    f = 2#(w/2)/np.arctan(np.pi/8)

    cylinder = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            x_ = f * np.arctan((x - w / 2)) / f + f * np.arctan(w/2/f)
            y_ = f * (y - h / 2) / np.sqrt((x - w / 2) ** 2 + f ** 2) + h / 2
            x_ = int(x + 0.5)
            y_ = int(y + 0.5)
            if x_ >= 0 and x_ < w and y_ >= 0 and y_ < h:
                cylinder[y, x] = img[y_, x_]
    return cylinder

def GainCompensator(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    img_equalized = np.interp(img_gray.flatten(), bins[:-1], cdf_normalized)
    img_equalized = img_equalized.reshape(img_gray.shape)
    img_equalized = img_equalized.astype(np.uint8)
    img_compensated = cv2.cvtColor(img_equalized, cv2.COLOR_GRAY2BGR)
    return img_compensated

if __name__ == '__main__':
    imgs, img_grays = [], []
    if sys.argv[1] != 'Base' and sys.argv[1] != 'Challenge':
        print("The second argument must be 'Base' or 'Challenge'")
        sys.exit()
    
    for filename in os.listdir(f'Photos/{sys.argv[1]}'):
        img, img_gray = read_img(f'Photos/{sys.argv[1]}/{filename}')
        imgs.append(img)
        img_grays.append(img_gray)
    
    # result = stitch_image((imgs[1], img_grays[1]), (imgs[2], img_grays[2]))
    result = stitch_images(imgs, img_grays)

    # stich images
    # result = stitch_images(imgs, img_grays)
    cv2.imwrite('result.jpg', result)
    
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)