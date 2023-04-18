import numpy as np


class Sperm:
    def __init__(self,
                 sperm_id,  # ids are given to each sperm at the time of declaration
                 position_x,  # initial x_coordinate when the sperm is found
                 position_y,  # initial y_coordinate when the sperm is found
                 width,  # width of the bounding box
                 height,  # height of the bounding box
                 dt=1 / 29.97,  # time interval between measurements --29.97 fps
                 u_x=0,  # acceleration in x direction.
                 u_y=0,  # acceleration in y direction.
                 std_acc=20,  # standard deviation of acceleration
                 std_meas=0.1,  # standard deviation of measurement
                 velocity=0,  # initial velocity
                 ):
        self.id = sperm_id # id of the sperm
        self.position_x = position_x # x coordinate of the sperm
        self.position_y = position_y # y coordinate of the sperm
        self.width = width # width of the bounding box
        self.height = height # height of the bounding box
        self.velocity = velocity # velocity of the sperm
        self.num_unmatched = 0 # number of unmatched frames
        self.max_unmatched = 3 # maximum number of unmatched frames
        self.num_matched = 1 # number of matched frames
        self.min_matched = 3 # minimum number of matched frames
        self.active = False # if the sperm is active or not
        self.health = False # if the sperm is healthy or not
        self.coor_history = [(position_x, position_y)] # history of the coordinates of the sperm, initialized with the initial coordinates
        self.dt = dt # time interval between measurements (29.97 FPS)
        self.u = np.matrix([[u_x], [u_y]]) # acceleration of the sperm
        self.X = np.matrix([[self.position_x], [self.position_y], [self.velocity], [self.velocity]]) # state vector
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]) # state transition matrix
        self.B = np.matrix([[(self.dt ** 2) / 2, 0],
                            [0, (self.dt ** 2) / 2],
                            [self.dt, 0],
                            [0, self.dt]]) # control input matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]]) # measurement matrix
        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * std_acc ** 2 # process noise covariance matrix
        self.R = np.matrix([[std_meas ** 2, 0],
                            [0, std_meas ** 2]]) # measurement noise covariance matrix
        self.P = np.eye(self.A.shape[1]) # error covariance matrix

    def predict(self):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.u) # state prediction
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q # error covariance prediction
        self.position_x = self.X[0,0] # update x coordinate of the sperm
        self.position_y = self.X[1,0] # update y coordinate of the sperm
        self.coor_history.append((self.position_x, self.position_y)) # append the coordinates to the history

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R # innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain
        self.X = np.round(self.X + np.dot(K, (z - np.dot(self.H, self.X)))) # state update
        I = np.eye(self.H.shape[1]) # identity matrix
        self.P = (I - (K * self.H)) * self.P # error covariance update
        self.position_x = self.X[0, 0] # update x coordinate of the sperm
        self.position_y = self.X[1, 0] # update y coordinate of the sperm
        self.coor_history[-1]=(self.position_x, self.position_y) # update the coordinates in the history

    def missing(self):
        self.num_unmatched += 1 # increment the number of unmatched frames
        if self.num_unmatched > self.max_unmatched:
            self.active = False
        
    def found(self):
        self.num_matched += 1 # increment the number of matched frames
        if self.num_matched > self.min_matched:
            self.active = True

    def get_box(self):
        return self.position_x, self.position_y, self.width, self.height 
    
    def get_coor_history(self):
        return self.coor_history[-60:]
    
    def healthy(self):
        history = self.get_coor_history() # get the last 60 coordinates
        x1, y1 = history[0] # get the first coordinate
        x2, y2 = history[-1] # get the last coordinate
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) # calculate the distance between the first and last coordinates
        velocity = dist / len(history) # calculate the velocity
        if velocity > 1:
            self.health = True
        else:
            self.health = False
        return self.health
