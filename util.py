import numpy as np
import cv2

def min_distance_index(sperm, crds):
    sperm_x = sperm.position_x
    sperm_y = sperm.position_y
    distances = []

    for crd in crds:
        dist = np.sqrt((sperm_x - crd[0].item()) ** 2 + (sperm_y - crd[1].item()) ** 2)
        distances.append(dist)
    
    return distances.index(min(distances)), min(distances)

def draw_lines(history, frame):
    for i in range(len(history) - 1):
        cv2.line(frame, (int(history[i][0]),int(history[i][1])), (int(history[i+1][0]),int(history[i+1][1])), (0, 0, 150+i*2), 1)
    return frame