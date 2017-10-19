# -*- coding: utf-8 -*-
"""
TH 1 - Bài 4: Thực hiện Clustering cho dữ liệu Labeled Faces in the Wild

                  Thực hiện alignment cho face:
                                + Tìm landmarks (Sử dụng dlib) -> Rotate theo mắt sao cho 2 mắt nằm ngang
                                + Crop ảnh, chỉ lấy phần center của face.
                    
               Vì thử nghiệm ở đây sẽ sử dụng alignment theo Multi-Task CNN (https://kpzhang93.github.io/MTCNN_face_detection_alignment/index.html)
                    nên hàm alignment này hiện tại không sử dụng.

Language: Python 3.6.1 - Anaconda 4.4 (64-bit) + dlib + opencv
OS: Windows 10 x64
Created on Tue Oct 17 08:25:05 2017
Last Modified: Oct 18 10:55 PM

Special thanks to [REF]: kevinlu1211  (https://github.com/kevinlu1211/FacialClusteringPipeline)

"""

import numpy as np
import cv2
import dlib
import copy 
import math

################################### UTILITIES FUNCTIONS #########################################
def find_rotation_angle(landmarks):
    # Rotate image so that the eyes are aligned
    left_eye_corner = landmarks[36]
    right_eye_corner = landmarks[45] 
    opp = right_eye_corner[1] - left_eye_corner[1]
    adj = right_eye_corner[0] - left_eye_corner[0]
    angle = np.arctan(opp/adj)*180/np.pi
    return(angle)

def find_rotation_matrix(image, angle):
    h, w = image.shape[0:2]
    center = tuple(np.array([h, w])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    return(rot_mat)

def rotate_image(image, angle):
    h, w = image.shape[0:2]
    center = tuple(np.array([h, w])/2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)
    image = cv2.warpAffine(image, rot_mat, (w, h))
    return(image, rot_mat)

def transform_2dpoints(points, rot_mat):
#     points_rs = np.reshape(np.array(points), (68, 1, 2))
    num_points = len(points)
    points_rs = np.expand_dims(np.array(points), 1)
    points_t = cv2.transform(points_rs, rot_mat)
    points_t = np.float32(np.reshape(points_t, (num_points, 2)))
    return(points_t)

def find_bbox_points(image, landmarks, margin_ratio):
    h, w = image.shape[0:2]
    
    left = max(0, min([landmark[0] for landmark in landmarks]))
    right = min(w, max([landmark[0] for landmark in landmarks]))
    lr_diff = right - left
    left = max(0, left - lr_diff * margin_ratio)
    right = min(w, right + lr_diff * margin_ratio)
    
    top = max(0, min([landmark[1] for landmark in landmarks]))
    bottom = min(h, max([landmark[1] for landmark in landmarks]))
    tb_diff = bottom - top
    top = max(0, top - tb_diff * margin_ratio)
    bottom = min(h, bottom + tb_diff * margin_ratio)
    
    return (int(left), int(top)), (int(right), int(bottom))

def get_centers_for_landmarks(landmarks):
    # Center of image
    left_most = landmarks[np.argmin(landmarks[:, 0])]
    right_most = landmarks[np.argmax(landmarks[:, 0])]
  
    face_center = ((left_most+right_most)/2).astype(np.int32)
    
    # Get the center of the eyes and mouth
    eye_area = np.concatenate((landmarks[36:42], landmarks[42:48]), axis = 0)
    eye_center = np.mean(eye_area, axis = 0).astype(np.int32)
    
    mouth_area = landmarks[48:68]
    mouth_center = np.mean(mouth_area, axis = 0).astype(np.int32)
    return([face_center, eye_center, mouth_center])


def find_center_transform(image, face_center):
    h, w = image.shape[0:2]
    center_dx = int(w/2) - face_center[0] # if face_center[0] > w/2 then it means that image 
                                          # will shift left
    center_dy = int(h/2) - face_center[1] # if face_center[1] > h/2 then it means that image 
                                          # will shift 
    M = np.float32([[1,0,center_dx],[0,1,center_dy]])
    return(M)

def center_image_on_face(image, center_matrix):
    h, w = image.shape[0:2]
    # Centered image
    return(cv2.warpAffine(image, center_matrix, (w, h)))

def similarity_transform(inPoints, outPoints):
    s60 = math.sin(60*math.pi/180)
    c60 = math.cos(60*math.pi/180)


    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    # Find the point for an equilateral triangle
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1]
    
    inPts.append([np.int(xin), np.int(yin)])
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1]
    
    outPts.append([np.int(xout), np.int(yout)]);

    tform = cv2.estimateRigidTransform(np.array([inPts]).astype(np.float32), np.array([outPts]).astype(np.float32), False)
    
    return tform

def get_landmark_points(pose_landmarks, dlib_point = True):
    landmarks = []
    n = 68
    for i in range(n): 
        point = pose_landmarks.part(i)
        if dlib_point:
            landmark = point
        else:
            landmark = [point.x, point.y]
        landmarks.append(landmark) 
    return(landmarks)

def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        draw_point(image, landmark)
    return(image)


################################### MAIN FUNCTIONS #########################################
def face_alignment(image, predictor_model = "shape_predictor_68_face_landmarks.dat"):
    '''
        Use dlib landmarks to align image.
        Input: 1 image with 1 people face
        Output: centered image
    '''
    face_detector = dlib.get_frontal_face_detector() 
    face_pose_predictor = dlib.shape_predictor(predictor_model) 
    
    detected_faces = face_detector(image)

    # Get the bounding box for the face
    for i, face_rect in enumerate(detected_faces):
    #     if i == 1:
    #         continue
        temp_image = copy.deepcopy(image)
        h, w = temp_image.shape[0:2]

        # Get the landmarks
        pose_landmarks = face_pose_predictor(temp_image, face_rect)
        landmarks = get_landmark_points(pose_landmarks, dlib_point=False)
    #     left, right, top, bottom = find_bbox_points(landmarks)
    #     temp_image = temp_image[top:bottom, left:right]

        # Rotate image so that the eyes are aligned
        angle = find_rotation_angle(landmarks)
        rot_mat = find_rotation_matrix(image, angle)
        temp_image = cv2.warpAffine(temp_image, rot_mat, (w, h))

        # Rotate the facial features to create a bbox for the face
        landmarks = transform_2dpoints(landmarks, rot_mat)   
            
        # Fidn the centers for the face
        center_points_src = get_centers_for_landmarks(landmarks)

    #     # Do affine transform to center the image
        center_transform = find_center_transform(temp_image, center_points_src[0]) 
        temp_image = center_image_on_face(temp_image, center_transform)

        # Update the center points and landmark points
        center_points_src = transform_2dpoints(center_points_src, center_transform)
        landmarks = transform_2dpoints(landmarks, center_transform)

        # Get the target points for the affine transform to make the eye and mouth at same position
        # for all the images
        eye_center_dst = [float(center_points_src[1][0]), h * 0.45]
        mouth_center_dst = [float(center_points_src[2][0]), h * 0.75]

        alignment_transform = similarity_transform([center_points_src[1], center_points_src[2]], [eye_center_dst, mouth_center_dst])
        temp_image = cv2.warpAffine(temp_image, alignment_transform, (w, h)) 
        landmarks = transform_2dpoints(landmarks, alignment_transform) 
        center_points_src = transform_2dpoints(center_points_src, alignment_transform)    

        # Update the center points and landmark points
    #     landmarks = transform_2dpoints(landmarks, alignment_transform)

        top_corner, bottom_corner = find_bbox_points(temp_image, landmarks, 0.1)

    #     cv2.rectangle(temp_image, top_corner, bottom_corner, (255, 0, 0), 3)   
        
        face_image = temp_image[top_corner[1]:bottom_corner[1], top_corner[0]:bottom_corner[0]]
        face_resized = cv2.resize(face_image, (160, 160))
        
        #name_of_face = image_path.split("/")[6]
            
        return face_resized



################################### TESTING #########################################

#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt

#image = mpimg.imread('./face_utils/test_facealign_1.jpg').astype(np.uint8)[:,:,0:3]
##image = mpimg.imread('./face_utils/test_facealign_2.jpg').astype(np.uint8)[:,:,0:3]

#image_align = face_alignment(image)

#plt.figure(1)
#plt.imshow(image)
#plt.figure(2)
#plt.imshow(image_align)

#plt.show()