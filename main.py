import cv2
from ultralytics import YOLO
from sperm import Sperm
from util import *
import time
import os


def main():
    # load the video and the trained model
    video = "BlurrySample"
    output = video + "Output.avi"
    video_path = os.path.join("testing videos", video+".wmv")
    cap = cv2.VideoCapture(video_path)
    model = YOLO("Model.pt")
    # read the first frame for initial assigning of sperms
    ret, frame = cap.read()
    
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
    while (len(crds) == 0):
        ret, frame = cap.read()
        results = model.predict(frame, conf=0.45)
        boxes = results[0].boxes
        crds = boxes.xywh
    # make an empty list to store sperms
    sperm_list = []
    id_count = 0

    # add sperms to the list
    for crd in crds:
        sperm_list.append(Sperm(id_count, crd[0].item(), crd[1].item(), crd[2].item(), crd[3].item()))
        id_count += 1

    cv2.putText(frame, "sperm count: " + str(len(crds)), (0, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

    for sperm in sperm_list:
        cv2.putText(frame, str(sperm.id), (int(sperm.position_x), int(sperm.position_y)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


    cv2.imshow('Sperm Tracking', frame)
    video_writer.write(frame)

    previous=time.time()
    next=0


    # start of the loop
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(frame, conf=0.45)
        boxes = results[0].boxes
        crds = boxes.xywh.cpu().numpy()

        for sperm in sperm_list:
            sperm.predict()
            if sperm.position_x < 0 or sperm.position_x > 640 or sperm.position_y < 0 or sperm.position_y > 480:
                sperm.active = False
            try:
                index, dist = min_distance_index(sperm, crds)
            except ValueError:
                sperm.missing()
                continue
            if dist < 10:
                sperm.update(np.matrix([[crds[index][0]], [crds[index][1]]]))
                crds = np.delete(crds, index, 0)
                sperm.found()
                sperm.num_unmatched = 0
                
            else:
                sperm.missing()

        for crd in crds:
            sperm_list.append(Sperm(id_count, crd[0], crd[1], crd[2], crd[3]))
            id_count += 1

        healthy_sperm = 0

        for sperm in sperm_list:
            if sperm.active:
                cv2.putText(frame, str(sperm.id), (int(sperm.position_x)-10, int(sperm.position_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                history = sperm.get_coor_history()
                frame = draw_lines(history, frame)
                if sperm.healthy():
                    healthy_sperm += 1

        cv2.putText(frame, "sperm count: " + str(len(list(sperm for sperm in sperm_list if sperm.active))), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "healthy sperm count: " + str(healthy_sperm), (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv2.LINE_AA)
        if healthy_sperm / (len(list(sperm for sperm in sperm_list if sperm.active))+0.00001) < 0.4:
            cv2.putText(frame, "Infertile Specimen", (0,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Fertile Specimen", (0,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
            
        next=time.time()
        cv2.putText(frame, "FPS: " + str(int(1/(next-previous))), (0,11), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        previous=next

        cv2.imshow('Sperm Tracking', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    video_writer.release()


if __name__ == '__main__':
    main()