"""
YawnDD dataset pre-processing
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib
import torch
import os

TOTAL_FRAME_FLAG = cv2.CAP_PROP_FRAME_COUNT
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

detector = dlib.get_frontal_face_detector()

def get_landmarks(im):
    rects = detector(im, 1)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im

def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50,53):
        top_lip_pts.append(landmarks[i])
    for i in range(61,64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:,1])

def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65,68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56,59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:,1])

def mouth_open(image):
    landmarks = get_landmarks(image)
    print(f'shape of landmarks: {len(landmarks)}')
    # print(f'type of landmarks: {type(landmarks)}')

    if landmarks == "error":
        return image, 0, 0, 0
    
    image_with_landmarks = annotate_landmarks(image, landmarks)
    top_lip_center = top_lip(landmarks)
    bottom_lip_center = bottom_lip(landmarks)
    lip_distance = abs(top_lip_center - bottom_lip_center)
    
    xmouthpoints = [landmarks[x,0] for x in range(48,67)]
    ymouthpoints = [landmarks[x,1] for x in range(48,67)]

    maxx = max(xmouthpoints)
    minx = min(xmouthpoints)
    maxy = max(ymouthpoints)
    miny = min(ymouthpoints) 

    pad = 5

    crop_image = image[miny-pad:maxy+pad,minx-pad:maxx+pad]

    return image_with_landmarks, landmarks, lip_distance, crop_image

def lip_distance(face):
    top_lip = face[52:55]
    top_lip = np.concatenate((top_lip, face[63:67]))

    low_lip = face[56:59]
    low_lip = np.concatenate((low_lip, face[65:68]))
    return top_lip, low_lip

def get_frames(path, total_num_frames=None, dataset=[], target=[]):
    """
    extract frames from the given video
    """
    plot_landmarks = False
    cap=cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_FPS, 30)

    yawn_count = 0
    close_count = 0

    if total_num_frames is None:
        total_num_frames = int(cap.get(TOTAL_FRAME_FLAG))    # get total number of frames

    print(f'Total number of frames: {total_num_frames}')
    for ii in range(total_num_frames):
        success,image = cap.read()
        print(f'frame: {ii}')
        image = cv2.resize(image, (320, 320))
        cv2.imwrite(f"./frames/frames_face_{ii}.jpg", image)
        
        # cropped_face = facial_detection(image)
        
        image_landmarks, landmarks, lip_distance, crop_mouth = mouth_open(image)

        if np.sum(landmarks) == 0:
            continue
        
        if plot_landmarks:
            image_2 = image.copy()
            for p in landmarks:
                image_2[p[0,1]-1:p[0,1]+1, p[0,0]-1:p[0,0]+1, :] = (255, 255, 255)
                plt.imshow(image_2)

        # print(crop_mouth.shape)
        crop_mouth = cv2.resize(crop_mouth, (32, 32))
        if lip_distance >= 15:
            print(f'Yawning!!! yawn_count={yawn_count}')
            label = [0,1]

            print('saving....')
            dataset.append(crop_mouth)
            target.append(label)

            yawn_count+=1
        else:
            label = [0,0]
            print(f'lip distance: {lip_distance}')
            if close_count < yawn_count:
                dataset.append(crop_mouth)
                target.append(label)
                print(f'Non-yawning label: {label}')
                close_count+=1

        print(f'Number of yawning: {yawn_count}, number of close: {close_count}')
    return dataset, target


def main():
    dataset = []
    target = []
    for kk in os.listdir('./videos'):
        if '.avi' in kk:
            print(f'Extracting {kk}....')
            data, label = get_frames('./videos/'+kk, total_num_frames=None)
            print(f'shape of the dataset: {len(data)}')
            dataset = dataset + data
            target = target + label

    dataset = torch.Tensor(dataset)
    target = torch.Tensor(target)
    
    print(f'shape of the final dataset: {list(dataset.size())} | shape of the final target: {list(target.size())}')

    torch.save(dataset, './yawnDD_image.pt')
    torch.save(target, './yawnDD_label.pt')




if __name__ == '__main__':
    main()