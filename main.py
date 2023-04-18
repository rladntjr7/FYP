# load dependencies
import cv2
from ultralytics import YOLO
from sperm import Sperm
from util import *
import time
import os

# main function
def main():
    # load the video and the trained model
    video = "BlurrySample"
    output = video + "Output.avi"
    video_path = os.path.join("testing videos", video+".wmv")
    cap = cv2.VideoCapture(video_path)
    model = YOLO("Model.pt")

    # read the first frame for initial assigning of sperms
    ret, frame = cap.read()
    
    # get the video properties
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)+0.5)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)+0.5)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Video Writer 
    video_writer = cv2.VideoWriter((os.path.join("video output", output)), fourcc, 30, (width, height)) 

    # if the video loading fails, exits the program
    if not ret:
        print("Error: Video not found")
        exit()

    # detects sperms from the first frame
    results = model.predict(frame, conf=0.45)
    boxes = results[0].boxes
    crds = boxes.xywh

    # if no sperms are detected, keep detecting sperms until sperms are detected
    while (len(crds) == 0):
        ret, frame = cap.read()
        results = model.predict(frame, conf=0.45)
        boxes = results[0].boxes
        crds = boxes.xywh

    # make an empty list to store sperms
    sperm_list = []

    # initialize the id count
    id_count = 0

    # add sperms to the list
    for crd in crds:
        sperm_list.append(Sperm(id_count, crd[0].item(), crd[1].item(), crd[2].item(), crd[3].item()))
        id_count += 1

    # Write sperm count on the frame
    cv2.putText(frame, "sperm count: " + str(len(crds)), (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    # draw the sperms on the frame
    for sperm in sperm_list:
        cv2.putText(frame, str(sperm.id), (int(sperm.position_x), int(sperm.position_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # show the frame
    cv2.imshow('Sperm Tracking', frame)

    # write the frame to the video
    video_writer.write(frame)

    # initialize the time for FPS tracking
    previous=time.time()
    next=0


    # start of the loop
    while True:
        # read the next frame
        ret, frame = cap.read()

        # if the video ends, exit the program
        if not ret:
            break

        # predict the sperms in the frame
        results = model.predict(frame, conf=0.45)
        boxes = results[0].boxes
        crds = boxes.xywh.cpu().numpy()

        # start looping through the sperms
        for sperm in sperm_list:
            # predict the priori position of sperm
            sperm.predict()
            # if the sperm is out of the frame, deactivate the sperm
            if sperm.position_x < 0 or sperm.position_x > 640 or sperm.position_y < 0 or sperm.position_y > 480:
                sperm.active = False
                continue

            # find the closest detection with the sperm
            try:
                index, dist = min_distance_index(sperm, crds)

            # if there is no detection left, the sperm unmatched count increases
            except ValueError:
                sperm.missing()
                continue

            # if the distance is less than 10, update the posteriori position and remove the detection from the list
            if dist < 10:
                sperm.update(np.matrix([[crds[index][0]], [crds[index][1]]]))
                crds = np.delete(crds, index, 0)
                sperm.found()
                sperm.num_unmatched = 0
                
            # if the distance is greater than 10, the sperm unmatched count increases
            else:
                sperm.missing()

        # start looping through the remaining detections and add them as sperms
        for crd in crds:
            sperm_list.append(Sperm(id_count, crd[0], crd[1], crd[2], crd[3]))
            id_count += 1

        # initialize the healthy sperm count
        healthy_sperm = 0

        # draw the id and the path of the sperms on the frame
        for sperm in sperm_list:
            if sperm.active:
                cv2.putText(frame, str(sperm.id), (int(sperm.position_x)-10, int(sperm.position_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                history = sperm.get_coor_history()
                frame = draw_lines(history, frame)

                # if the sperm is healthy, increase the healthy sperm count
                if sperm.healthy():
                    healthy_sperm += 1

        # write the sperm count and the healthy sperm count on the frame
        cv2.putText(frame, "sperm count: " + str(len(list(sperm for sperm in sperm_list if sperm.active))), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "healthy sperm count: " + str(healthy_sperm), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        
        # write the fertility status on the frame
        if healthy_sperm / (len(list(sperm for sperm in sperm_list if sperm.active))+0.00001) < 0.4:
            cv2.putText(frame, "Infertile Specimen", (0,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Fertile Specimen", (0,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
            
        # update the current time and calculate the FPS
        next=time.time()
        cv2.putText(frame, "FPS: " + str(int(1/(next-previous))), (0,11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        previous=next

        # show the frame
        cv2.imshow('Sperm Tracking', frame)

        # write the frame to the video
        video_writer.write(frame)

        # if the user presses q, exit the program
        if cv2.waitKey(1) == ord('q'):
            break

    # release the video and the camera
    cap.release()
    cv2.destroyAllWindows()

    # release the video writer
    video_writer.release()

# when the program is executed, run the main function
if __name__ == '__main__':
    main()