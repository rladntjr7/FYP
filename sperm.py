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
        self.id = sperm_id
        self.position_x = position_x
        self.position_y = position_y
        self.width = width
        self.height = height
        self.velocity = velocity
        self.num_unmatched = 0
        self.max_unmatched = 3
        self.num_matched = 1
        self.min_matched = 3
        self.active = False
        self.health = False
        self.coor_history = [(position_x, position_y)]
        self.dt = dt
        self.u = np.matrix([[u_x], [u_y]])
        self.X = np.matrix([[self.position_x], [self.position_y], [self.velocity], [self.velocity]])
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        self.B = np.matrix([[(self.dt ** 2) / 2, 0],
                            [0, (self.dt ** 2) / 2],
                            [self.dt, 0],
                            [0, self.dt]])
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * std_acc ** 2
        self.R = np.matrix([[std_meas ** 2, 0],
                            [0, std_meas ** 2]])
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.X = np.dot(self.A, self.X) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.position_x = self.X[0,0]
        self.position_y = self.X[1,0]
        self.coor_history.append((self.position_x, self.position_y))

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.X = np.round(self.X + np.dot(K, (z - np.dot(self.H, self.X))))
        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P
        self.position_x = self.X[0, 0]
        self.position_y = self.X[1, 0]
        self.coor_history[-1]=(self.position_x, self.position_y)

    def missing(self):
        self.num_unmatched += 1
        if self.num_unmatched > self.max_unmatched:
            self.active = False
        
    def found(self):
        self.num_matched += 1
        if self.num_matched > self.min_matched:
            self.active = True

    def get_box(self):
        return self.position_x, self.position_y, self.width, self.height
    
    def get_coor_history(self):
        return self.coor_history[-60:]
    
    def healthy(self):
        history = self.get_coor_history()
        x1, y1 = history[0]
        x2, y2 = history[-1]
        dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        velocity = dist / len(history)
        if velocity > 1:
            self.health = True
        else:
            self.health = False
        return self.health
