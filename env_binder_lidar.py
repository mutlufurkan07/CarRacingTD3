import copy
import math

import gym
import cv2
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
gym.logger.set_level(50)


class car_racer:
    def __init__(self, num_of_lidar_points=7, render_opencv=False):
        self.str_env_name_ = "CarRacing-v0"
        self.str_window_name = f"env"
        if render_opencv:
            cv2.namedWindow(self.str_window_name, cv2.WINDOW_FREERATIO)
            self.render_opencv = True
        else:
            self.render_opencv = False
        self.env = gym.make(self.str_env_name_)
        logging.info(f" {self.str_env_name_} env created")

        self.car_position_x = 0.0
        self.car_position_y = 0.0
        self.car_velocity_x = 0.0
        self.car_velocity_y = 0.0
        self.car_angularDamping = 0.0
        self.car_pos_angle = 0.0
        self.car_angularVelocity = 0.0
        self.car_linearDamping = 0.0
        self.num_of_lidar_points = num_of_lidar_points

    @staticmethod
    def __set_point_value(s_, point, list_=False):
        if not list_:
            s_[point[1], point[0], :] = 255
            return s_
        else:
            for a_point in point:
                s_[a_point[1], a_point[0], :] = 255
            return s_

    @staticmethod
    def __calculate_circle_points(radius, center_point, num_of_points=7):
        assert num_of_points % 2 == 1

        initial_upright_point = [center_point[0], center_point[1] - radius]
        deg_interval = math.pi / (num_of_points - 1)
        rotation_values_rad = []
        for i in range(num_of_points):
            rotation_values_rad.append((i * deg_interval) - math.pi / 2)

        rotated_points = []
        # translate point back to origin
        initial_upright_point[0] -= center_point[0]
        initial_upright_point[1] -= center_point[1]

        for rotation_value_rad in rotation_values_rad:
            # rotate point
            c = math.cos(rotation_value_rad)
            s = math.sin(rotation_value_rad)

            x1 = initial_upright_point[0] * c - initial_upright_point[1] * s
            y1 = initial_upright_point[0] * s + initial_upright_point[1] * c
            # translate point back
            rotated_points.append([round(x1 + center_point[0]), round(y1 + center_point[1])])

        return rotated_points

    def __get_lidar_features_img(self, s_):
        self.__get_car_kinematics()
        y_ = 66
        x1_ = 47
        x2_ = 48
        r1 = 10
        car_angle = self.car_pos_angle
        car_angle += math.pi / 2
        # print(car_angle*180/math.pi)

        # point1_upright_left = [x1_, y_ - r1]
        # point1_upright_right = [x2_, y_ - r1]
        # self.__set_point_value(s_, point1_upright_left)
        # self.__set_point_value(s_, point1_upright_right)
        rotated_circle_1 = self.__calculate_circle_points(radius=r1,
                                                          center_point=[x1_, y_],
                                                          num_of_points=self.num_of_lidar_points)
        # self.__set_point_value(s_, rotated_circle_1, list=True)
        distance_list_1 = []
        for a_point in rotated_circle_1:
            x_rate = ((a_point[0] - x1_) / r1) * 2
            y_rate = ((a_point[1] - y_) / r1) * 2

            found_lane_flag = False
            next_point = [x1_, y_]
            while not found_lane_flag:
                next_point[0] = round(next_point[0] + x_rate)
                next_point[1] = round(next_point[1] + y_rate)
                if next_point[0] > 95 or next_point[1] > 95:
                    break
                pixel_value = s_[next_point[1], next_point[0]]
                if not (((pixel_value > [30, 30, 30]).all()) and (pixel_value < [150, 150, 150]).all()):
                    break

                self.__set_point_value(s_, next_point)

            distance_list_1.append(math.sqrt((next_point[0] - x1_) ** 2 + (next_point[1] - y_) ** 2))

        s_[y_, x1_, :] = 255
        s_[y_, x2_, :] = 255
        return distance_list_1, s_

    def __get_car_kinematics(self):
        self.car_position_x = self.env.car.hull.position.x
        self.car_position_y = self.env.car.hull.position.y
        self.car_pos_angle = self.env.car.hull.angle
        # print(self.car_pos_angle)
        self.car_angularDamping = self.env.car.hull.angularDamping
        self.car_angularVelocity = self.env.car.hull.angularVelocity
        self.car_velocity_x = self.env.car.hull.linearVelocity.x
        self.car_velocity_y = self.env.car.hull.linearVelocity.y
        self.car_linearDamping = self.env.car.hull.linearDamping

    def __create_state(self, lidar_features):
        assert self.num_of_lidar_points == len(lidar_features)
        """
        states are 
            1 -> lidar_distances normalized to 60m
            2 -> car_pos_Angle
            3 -> car_angular_damping
            4 -> car_AngularVelocity
            5 -> car_velocity_x
            6 -> car_velocity_y
            7 -> car_linearDamping
        """
        state = np.zeros(self.num_of_lidar_points + 6, dtype=np.float32)
        lidar_features = np.asarray(lidar_features, dtype=np.float32)
        additional_done_condition = (lidar_features < 3).all()
        lidar_features = np.clip(lidar_features, 0, 60) / 60
        state[0:self.num_of_lidar_points] = lidar_features
        state[self.num_of_lidar_points] = self.car_pos_angle
        state[self.num_of_lidar_points + 1] = self.car_angularDamping
        state[self.num_of_lidar_points + 2] = self.car_angularVelocity
        state[self.num_of_lidar_points + 3] = self.car_velocity_x
        state[self.num_of_lidar_points + 4] = self.car_velocity_y
        state[self.num_of_lidar_points + 5] = self.car_linearDamping
        return state, additional_done_condition

    def reset(self):
        self.env.env.t = 1.1
        lidar_features, rgb_frame = self.__get_lidar_features_img(self.env.reset())
        state, another_done = self.__create_state(lidar_features=lidar_features)
        if self.render_opencv:
            cv2.imshow(self.str_window_name, rgb_frame)
            cv2.waitKey(1)

        return state

    @staticmethod
    def sample_random_action():
        return np.random.uniform(-1, 1, size=2)

    def step(self, a_, test=False):
        """
        :param a_:  a[0] -> steering
                    a[1] -> gas
                    a[2] -> brake
        :return: rgb_frame, reward, done, info
        """
        # a_[0] = 0.3
        # a_[1] = 0.1
        # a_[2] = 0.0
        if not a_.shape[0] == 3:
            if a_[1] > 0.0:
                gas = a_[1]
                brake = 0.0
            else:
                gas = 0.0
                brake = -a_[1]
            action_arr = np.asarray([a_[0], gas, brake])
        else:
            action_arr = a_
        rgb_frame, reward, done, info = self.env.step(action=action_arr)
        lidar_features, rgb_frame = self.__get_lidar_features_img(rgb_frame)
        state, another_done = self.__create_state(lidar_features=lidar_features)

        if self.render_opencv:
            cv2.imshow(self.str_window_name, rgb_frame)
            cv2.waitKey(1)
        if test:
            final_done = done
        else:
            final_done = done or another_done
        return state, reward, final_done, info

    def render(self):
        self.env.render()
