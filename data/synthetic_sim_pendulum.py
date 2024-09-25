import numpy as np
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
import argparse


class PendulumSim(object):
    def __init__(self, n_sticks=3, stick_length=1, stick_mass=1, loc_std=.1, vel_norm=.5,
                noise_var=0.):
        self.n_sticks = n_sticks
        self.stick_length = stick_length
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        
        self.noise_var = noise_var
        self.stick_mass = stick_mass
        
        self._delta_T = 0.0001
        self.g = 9.8


    def _energy(self, loc, vel):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            U=-self.stick_mass*self.stick_length*self.g/2*(5*np.cos(loc[0])+3*np.cos(loc[1])+1*np.cos(loc[2]))
            K=self.stick_mass*self.stick_length*self.stick_length/6*(9*vel[1]*vel[0]*np.cos(loc[0]-loc[1])+3*vel[2]*vel[0]*np.cos(loc[0]-loc[2])+3*vel[2]*vel[1]*np.cos(loc[1]-loc[2])+7*vel[0]*vel[0]+4*vel[1]*vel[1]+1*vel[2]*vel[2])

            print('U: ', U)
            print('K: ', K)
            print('energy:',U+K)

            return U + K, U ,K
    def generate_static_graph(self):
        # Sample edges: without self-loop
        edges = np.eye(self.n_sticks, k=1) + np.eye(self.n_sticks, k=-1)

        return edges


    def calculate_angular_speed(self, loc_next, p_next):
        vel_next = np.zeros((1, self.n_sticks))
        vel_next[0,0] = 6 * (9*p_next[0,0]*np.cos(2*(loc_next[0,1]-loc_next[0,2])) + 27*p_next[0,1]*np.cos(loc_next[0,0]-loc_next[0,1]) - 9*p_next[0,1]*np.cos(loc_next[0,0]+loc_next[0,1]-2*loc_next[0,2]) + 21*p_next[0,2]*np.cos(loc_next[0,0]-loc_next[0,2])- 27*p_next[0,2]*np.cos(loc_next[0,0]-2*loc_next[0,1]+loc_next[0,2]) - p_next[0,0]*23) / (self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169))
        vel_next[0,1] = 6 * (27*p_next[0,0]*np.cos(loc_next[0,0]-loc_next[0,1]) -9* p_next[0,0]*np.cos(loc_next[0,0]+loc_next[0,1]-2*loc_next[0,2]) + 9*p_next[0,1]*np.cos(2*(loc_next[0,0]-loc_next[0,2])) - 27*p_next[0,2]*np.cos(2*loc_next[0,0]-loc_next[0,1]-loc_next[0,2]) + 57*p_next[0,2]*np.cos(loc_next[0,1]-loc_next[0,2]) - p_next[0,1]*47) / (self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169))
        vel_next[0,2] = 6 * (21*p_next[0,0]*np.cos(loc_next[0,0]-loc_next[0,2]) - 27*p_next[0,0]*np.cos(loc_next[0,0]-2*loc_next[0,1]+loc_next[0,2]) - 27*p_next[0,1]*np.cos(2*loc_next[0,0]-loc_next[0,1]-loc_next[0,2])+ 57*p_next[0,1]*np.cos(loc_next[0,1]-loc_next[0,2])+ 81*p_next[0,2]*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - p_next[0,2]*143)  / (self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169))
        up=6 * (9*p_next[0,0]*np.cos(2*(loc_next[0,1]-loc_next[0,2])) + 27*p_next[0,1]*np.cos(loc_next[0,0]-loc_next[0,1]) - 9*p_next[0,1]*np.cos(loc_next[0,0]+loc_next[0,1]-2*loc_next[0,2]) + 21*p_next[0,2]*np.cos(loc_next[0,0]-loc_next[0,2])- 27*p_next[0,2]*np.cos(loc_next[0,0]-2*loc_next[0,1]+loc_next[0,2]) - p_next[0,0]*23)
        down=self.stick_mass*self.stick_length*self.stick_length*(81*np.cos(2*(loc_next[0,0]-loc_next[0,1])) - 9*np.cos(2*(loc_next[0,0]-loc_next[0,2])) + 45*np.cos(2*(loc_next[0,1]-loc_next[0,2]))- 169)
        return vel_next

    def calculate_p_dot(self, loc_next, vel_next):
        p_dot = np.zeros((1, self.n_sticks))
        p_dot[0,0] = -1/2 * self.stick_mass * self.stick_length * (3*vel_next[0,1]*vel_next[0,0]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,1]) + vel_next[0,0]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,2]) + 5*self.g*np.sin(loc_next[0,0])) 
        p_dot[0,1] = -1/2 * self.stick_mass * self.stick_length * (-3*vel_next[0,1]*vel_next[0,0]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,1]) + vel_next[0,1]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,1]-loc_next[0,2])+ 3*self.g*np.sin(loc_next[0,1]))
        p_dot[0,2] = -1/2 * self.stick_mass * self.stick_length * (vel_next[0,0]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,0]-loc_next[0,2]) + vel_next[0,1]*vel_next[0,2]*self.stick_length*np.sin(loc_next[0,1]-loc_next[0,2]) - self.g*np.sin(loc_next[0,2]))

        return p_dot

    def sample_trajectory_static_graph_irregular_difflength_each(self, args, edges, isTrain = True):
        '''
        every node have different observations
        train observation length [ob_min, ob_max]
        :param args:
        :param edges:
        :param isTrain:
        :param sample_freq:
        :param step_train:
        :param step_test:
        :return:
        '''
        sample_freq = args.sample_freq
        ode_step = args.ode
        max_ob = ode_step//sample_freq

        num_test_box = args.num_test_box
        num_test_extra  = args.num_test_extra

        ob_max = args.ob_max
        ob_min = args.ob_min


        #########Modified sample_trajectory with static graph input, irregular timestamps.

        n = self.n_sticks

        if isTrain:
            T = ode_step
        else:
            T = ode_step * (1 + num_test_box)

        step = T//sample_freq



        counter = 1 #reserve initial point
        # Initialize location and velocity
        loc_theta = np.zeros((step, 1, n))
        vel_theta = np.zeros((step, 1, n))
        #
        loc_next = np.random.uniform(0, np.pi/2, (1, 3))
        loc_next = np.mod(loc_next, 2*np.pi)

        print('initial loc:', loc_next)
        p_next =np.zeros((1, 3))
        print('initial p: ', p_next)
        loc_dot = np.zeros((1, 3))
        print('initial vel: ', loc_dot )
        # initial_energy = sim._energy(loc_next[0], loc_dot[0])
        # print('initial_energy:', initial_energy)
        loc_theta[0, :, :], vel_theta[0, :, :] =loc_next,loc_dot
        #---------RK4 solver--------------
        # with np.errstate(divide='ignore'):
        #     for i in range(1, T):
        #         # print('rk4: ',i)
        #         d_loc_1=self.calculate_angular_speed(loc_next, p_next)*self._delta_T
        #         d_p_1 = self.calculate_p_dot(loc_next, loc_dot)*self._delta_T
        #         loc_dot_1 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1, p_next + 1 / 2 * d_p_1)
        #
        #         d_loc_2 =  loc_dot_1 * self._delta_T
        #         d_p_2 = self.calculate_p_dot(loc_next+1/2*d_loc_1, loc_dot_1) * self._delta_T
        #         loc_dot_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2, p_next + 1 / 2 * d_p_2)
        #
        #         d_loc_3 = loc_dot_2 * self._delta_T
        #         d_p_3 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_2, loc_dot_2) * self._delta_T
        #         loc_dot_3 = self.calculate_angular_speed(loc_next + d_loc_3, p_next +  d_p_3)
        #
        #         d_loc_4 = loc_dot_3 * self._delta_T
        #         d_p_4 = self.calculate_p_dot(loc_next +  d_loc_3, loc_dot_3) * self._delta_T
        #
        #         d_loc=(1/6)*(d_loc_1+2*d_loc_2+2*d_loc_3+d_loc_4)
        #         d_p=(1/6)*(d_p_1+2*d_p_2+2*d_p_3+d_p_4)
        #         loc_next +=d_loc
        #         p_next +=d_p
        #         loc_dot=self.calculate_angular_speed(loc_next, p_next)
        #
        #         if i % sample_freq == 0:
        #             loc_theta[counter, :, :], vel_theta[counter, :, :] = loc_next, loc_dot
        #             counter += 1
        with np.errstate(divide='ignore'):
            for i in range(1, T):
                d_loc_1 = self.calculate_angular_speed(loc_next, p_next) * self._delta_T
                d_p_1 = self.calculate_p_dot(loc_next, loc_dot) * self._delta_T

                # loc_dot_1 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1, p_next + 1 / 2 * d_p_1)
                d_loc_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1,
                                                       p_next + 1 / 2 * d_p_1) * self._delta_T
                d_p_2 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_1, d_loc_1 / self._delta_T) * self._delta_T
                # loc_dot_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2, p_next + 1 / 2 * d_p_2)

                d_loc_3 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2,
                                                       p_next + 1 / 2 * d_p_2) * self._delta_T
                d_p_3 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_2, d_loc_2 / self._delta_T) * self._delta_T
                # loc_dot_3 = self.calculate_angular_speed(loc_next + d_loc_3, p_next +  d_p_3)

                d_loc_4 = self.calculate_angular_speed(loc_next + d_loc_3, p_next + d_p_3) * self._delta_T
                d_p_4 = self.calculate_p_dot(loc_next + d_loc_3, d_loc_3 / self._delta_T) * self._delta_T

                d_loc = (1 / 6) * (d_loc_1 + 2 * d_loc_2 + 2 * d_loc_3 + d_loc_4)
                d_p = (1 / 6) * (d_p_1 + 2 * d_p_2 + 2 * d_p_3 + d_p_4)
                loc_next += d_loc
                p_next += d_p
                loc_dot = self.calculate_angular_speed(loc_next, p_next)

                if i % sample_freq == 0:
                    loc_theta[counter, :, :], vel_theta[counter, :, :] = loc_next, loc_dot
                    counter += 1


        #-----Leapfrog solver--------
        # disables division by zero warning, since I fix it with fill_diagonal
        # with np.errstate(divide='ignore'):
        #     vel_next = np.zeros((1, 3))
        #     print('initial vel: ', vel_next)
        #     initial_energy = sim._energy(loc_next[0], vel_next[0])
        #     print('initial_energy:', initial_energy)
        #
        #     for i in range(1, T):
        #         p_dot = self.calculate_p_dot(loc_next, vel_next)
        #         p_mid = p_next + 1 / 2 * self._delta_T * p_dot
        #         vel_next = self.calculate_angular_speed(loc_next, p_mid)
        #         loc_next += self._delta_T * vel_next
        #         loc_next = np.mod(loc_next, 2*np.pi)
        #         p_dot = self.calculate_p_dot(loc_next, vel_next)
        #         p_next=p_mid+1 / 2 * self._delta_T * p_dot
        #         vel_next = self.calculate_angular_speed(loc_next, p_next)
        #         if i % sample_freq == 0:
        #             loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
        #             counter += 1

        # ----------Euler solver-----------
        # with np.errstate(divide='ignore'):
        #     vel_next = self.calculate_angular_speed(loc_next, p_next)
        #     # run leapfrog
        #     for i in range(1, T):
        #         loc_next += self._delta_T * vel_next
        #         loc_next = np.mod(loc_next, 2*np.pi)
        #
        #         if i % sample_freq == 0:
        #             loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
        #             counter += 1
        #
        #         p_dot = self.calculate_p_dot(loc_next, vel_next)
        #         p_next += self._delta_T * p_dot
        #         print('p_dot: ', p_dot)
        #         print('p_next: ', p_next)
        #         vel_next = self.calculate_angular_speed(loc_next, p_next)
            # turn to x y
            loc = np.zeros((step, 2, n))
            vel = np.zeros((step, 2, n))
            #记录端点坐标
            x0 = self.stick_length  * np.sin(loc_theta[:, :, 0])
            y0 = -self.stick_length* np.cos(loc_theta[:, :, 0])
            x1 =  self.stick_length* (np.sin(loc_theta[:, :, 0])+np.sin(loc_theta[:, :, 1]))
            y1 = -self.stick_length* (np.cos(loc_theta[:, :, 0])+np.cos(loc_theta[:, :, 1]))
            x2 = self.stick_length * (np.sin(loc_theta[:, :, 0]) + np.sin(loc_theta[:, :, 1])+ np.sin(loc_theta[:, :, 2]))
            y2 = -self.stick_length * (np.cos(loc_theta[:, :, 0]) + np.cos(loc_theta[:, :, 1])+  np.cos(loc_theta[:, :, 2]))
            #记录端点速度
            v_x0=self.stick_length* np.cos(loc_theta[:, :, 0])*vel_theta[:, :, 0]
            v_y0=self.stick_length* np.sin(loc_theta[:, :, 0])*vel_theta[:, :, 0]
            v_x1 =self.stick_length* (np.cos(loc_theta[:, :, 0])*vel_theta[:, :, 0]+np.cos(loc_theta[:, :, 1])*vel_theta[:, :, 1])
            v_y1 =self.stick_length* (np.sin(loc_theta[:, :, 0])*vel_theta[:, :, 0]+np.sin(loc_theta[:, :, 1])*vel_theta[:, :, 1])
            v_x2 =self.stick_length * (np.cos(loc_theta[:, :, 0]) *vel_theta[:, :, 0]+ np.cos(loc_theta[:, :, 1])*vel_theta[:, :, 1]+  np.cos(loc_theta[:, :, 2])*vel_theta[:, :, 2])
            v_y2 =self.stick_length * (np.sin(loc_theta[:, :, 0])*vel_theta[:, :, 0] + np.sin(loc_theta[:, :, 1])*vel_theta[:, :, 1]+ np.sin(loc_theta[:, :, 2])*vel_theta[:, :, 2])

            loc[:, 0, 0] = x0.squeeze()
            loc[:, 1, 0] = y0.squeeze()
            loc[:, 0, 1] = x1.squeeze()
            loc[:, 1, 1] = y1.squeeze()
            loc[:, 0, 2] = x2.squeeze()
            loc[:, 1, 2] = y2.squeeze()


            vel[:, 0, 0] = v_x0.squeeze()
            vel[:, 1, 0] = v_y0.squeeze()
            vel[:, 0, 1] = v_x1.squeeze()
            vel[:, 1, 1] = v_y1.squeeze()
            vel[:, 0, 2] = v_x2.squeeze()
            vel[:, 1, 2] = v_y2.squeeze()
            # Add noise to observations
            loc += np.random.randn(step, 2, self.n_sticks) * self.noise_var
            vel += np.random.randn(step, 2, self.n_sticks) * self.noise_var

            # sampling

            loc_sample = []
            vel_sample = []
            loc_theta_sample = []
            vel_theta_sample = []
            time_sample = []

            if isTrain:

                for i in range(n):
                    # number of timesteps
                    num_steps = np.random.randint(low=ob_min, high=ob_max + 1, size=1)[0]
                    # value of timesteps
                    Ts_ball = self.sample_timestamps_with_initial(num_steps, 0, max_ob)

                    loc_sample.append(loc[Ts_ball,:,i])
                    vel_sample.append(vel[Ts_ball, :, i])
                    loc_theta_sample.append(loc_theta[Ts_ball, :, i])
                    vel_theta_sample.append(vel_theta[Ts_ball, :, i])
                    time_sample.append(Ts_ball)


            else:
                for i in range(n):
                    # number of timesteps
                    num_steps = np.random.randint(low=ob_min, high=ob_max, size=1)[0]
                    # value of timesteps
                    Ts_ball = self.sample_timestamps_with_initial(num_steps, 0, max_ob)

                    for j in range(num_test_box):
                        start = max_ob + j*max_ob
                        end  = min(T//sample_freq,max_ob + (j+1)*max_ob)
                        Ts_append = self.sample_timestamps_with_initial(num_test_extra,start,end)
                        Ts_ball = np.append(Ts_ball,Ts_append)

                    loc_sample.append(loc[Ts_ball,:,i])
                    vel_sample.append(vel[Ts_ball, :, i])
                    loc_theta_sample.append(loc_theta[Ts_ball, :, i])
                    vel_theta_sample.append(vel_theta[Ts_ball, :, i])
                    time_sample.append(Ts_ball)
            # print('initial theta:', loc_theta_sample[0])
            # print('initial theta:', loc_theta_sample[0][0])
            # print('initial theta sin:', np.sin(loc_theta_sample[0][0]))
            # print('initial theta cos:', np.cos(loc_theta_sample[0][0]))
            #
            # print('initial loc:', loc_sample[0])


            return loc_sample, vel_sample ,loc_theta_sample,vel_theta_sample,time_sample


    def sample_timestamps_with_initial(self, num_sample, start, end):
        times = set()
        assert(num_sample<=(end-start-1))
        times.add(start)
        while len(times) < num_sample:
            times.add(int(np.random.randint(low=start+1, high=end, size=1)[0]))
        times = np.sort(np.asarray(list(times)))
        return times

    def sample_trajectory(self, T=10000, sample_freq=10,loc_0=0,p_0=0,loc_dot_0=0,
                          spring_prob=[1. / 2, 0, 1. / 2]):

        n = self.n_sticks
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Def edges
        edges = np.eye(self.n_sticks, k=1) + np.eye(self.n_sticks, k=-1)

        # Initialize location and velocity
        loc_theta = np.zeros((T_save, 1, n))
        vel_theta = np.zeros((T_save, 1, n))
        #
        # loc_next = np.random.uniform(0, np.pi / 2, (1, 3)) * self.loc_std
        loc_next = loc_0
        # loc_next = np.full((1, 3), np.pi / 2)
        # loc_next = np.mod(loc_next, 2*np.pi)
        print('initial loc:', loc_next)
        # p_next = np.zeros((1, 3))
        p_next=p_0
        print('initial p: ', p_next)
        # loc_dot = np.zeros((1, 3))
        loc_dot = loc_dot_0
        print('initial vel: ', loc_dot)
        # initial_energy = sim._energy(loc_next[0], loc_dot[0])
        # print('initial_energy:', initial_energy)
        loc_theta[0, :, :],vel_theta[0, :, :]=loc_next, loc_dot

        # ---------RK4 solver--------------
        # with np.errstate(divide='ignore'):
        #     for i in range(1, T):
        #         # print('rk4: ',i)
        #         d_loc_1 = self.calculate_angular_speed(loc_next, p_next) * self._delta_T
        #         d_p_1 = self.calculate_p_dot(loc_next, loc_dot) * self._delta_T
        #         loc_dot_1 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1, p_next + 1 / 2 * d_p_1)
        #
        #         d_loc_2 = loc_dot_1 * self._delta_T
        #         d_p_2 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_1, loc_dot_1) * self._delta_T
        #         loc_dot_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2, p_next + 1 / 2 * d_p_2)
        #
        #         d_loc_3 = loc_dot_2 * self._delta_T
        #         d_p_3 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_2, loc_dot_2) * self._delta_T
        #         loc_dot_3 = self.calculate_angular_speed(loc_next + d_loc_3, p_next + d_p_3)
        #
        #         d_loc_4 = loc_dot_3 * self._delta_T
        #         d_p_4 = self.calculate_p_dot(loc_next + d_loc_3, loc_dot_3) * self._delta_T
        #
        #         d_loc = (1 / 6) * (d_loc_1 + 2 * d_loc_2 + 2 * d_loc_3 + d_loc_4)
        #         d_p = (1 / 6) * (d_p_1 + 2 * d_p_2 + 2 * d_p_3 + d_p_4)
        #         loc_next += d_loc
        #         p_next += d_p
        #         loc_dot = self.calculate_angular_speed(loc_next, p_next)
        #         print('RK:',i)
        #         if i % sample_freq == 0:
        #             loc_theta[counter, :, :], vel_theta[counter, :, :] = loc_next, loc_dot
        #             counter += 1

        with np.errstate(divide='ignore'):
            for i in range(1, T):
                d_loc_1 = self.calculate_angular_speed(loc_next, p_next) * self._delta_T
                d_p_1 = self.calculate_p_dot(loc_next, loc_dot) * self._delta_T

                # loc_dot_1 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1, p_next + 1 / 2 * d_p_1)
                d_loc_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_1,
                                                       p_next + 1 / 2 * d_p_1) * self._delta_T
                d_p_2 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_1, d_loc_1 / self._delta_T) * self._delta_T
                # loc_dot_2 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2, p_next + 1 / 2 * d_p_2)

                d_loc_3 = self.calculate_angular_speed(loc_next + 1 / 2 * d_loc_2,
                                                       p_next + 1 / 2 * d_p_2) * self._delta_T
                d_p_3 = self.calculate_p_dot(loc_next + 1 / 2 * d_loc_2, d_loc_2 / self._delta_T) * self._delta_T
                # loc_dot_3 = self.calculate_angular_speed(loc_next + d_loc_3, p_next +  d_p_3)

                d_loc_4 = self.calculate_angular_speed(loc_next + d_loc_3, p_next + d_p_3) * self._delta_T
                d_p_4 = self.calculate_p_dot(loc_next + d_loc_3, d_loc_3 / self._delta_T) * self._delta_T

                d_loc = (1 / 6) * (d_loc_1 + 2 * d_loc_2 + 2 * d_loc_3 + d_loc_4)
                d_p = (1 / 6) * (d_p_1 + 2 * d_p_2 + 2 * d_p_3 + d_p_4)
                loc_next += d_loc
                p_next += d_p
                loc_dot = self.calculate_angular_speed(loc_next, p_next)

                # print(loc_theta[0, :, :])
                # print(vel_theta[0, :, :])
                if i % sample_freq == 0:
                    loc_theta[counter, :, :], vel_theta[counter, :, :] = loc_next, loc_dot

                    counter += 1

            # -----Leapfrog solver--------
            # disables division by zero warning, since I fix it with fill_diagonal
            # with np.errstate(divide='ignore'):
            #     vel_next = np.zeros((1, 3))
            #     print('initial vel: ', vel_next)
            #     initial_energy = sim._energy(loc_next[0], vel_next[0])
            #     print('initial_energy:', initial_energy)
            #
            #     for i in range(1, T):
            #         p_dot = self.calculate_p_dot(loc_next, vel_next)
            #         p_mid = p_next + 1 / 2 * self._delta_T * p_dot
            #         vel_next = self.calculate_angular_speed(loc_next, p_mid)
            #         loc_next += self._delta_T * vel_next
            #         loc_next = np.mod(loc_next, 2*np.pi)
            #         p_dot = self.calculate_p_dot(loc_next, vel_next)
            #         p_next=p_mid+1 / 2 * self._delta_T * p_dot
            #         vel_next = self.calculate_angular_speed(loc_next, p_next)
            #         if i % sample_freq == 0:
            #             loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
            #             counter += 1

            # ----------Euler solver-----------
        # with np.errstate(divide='ignore'):
        #
        #     # run leapfrog
        #     for i in range(1, T):
        #         loc_next += self._delta_T * loc_dot
        #         loc_next = np.mod(loc_next, 2*np.pi)
        #         p_dot = self.calculate_p_dot(loc_next, loc_dot)
        #         p_next += self._delta_T * p_dot
        #         loc_dot = self.calculate_angular_speed(loc_next, p_next)
        #
        #         if i % sample_freq == 0:
        #             loc_theta[counter, :, :], vel_theta[counter, :, :] = loc_next, loc_dot
        #             counter += 1
        #         print('Euler')
                #
                # print('p_dot: ', p_dot)
                # print('p_next: ', p_next)


            # turn to x y
            loc = np.zeros((T_save, 2, n))

            vel = np.zeros((T_save, 2, n))
            # 记录端点坐标
            x0 = self.stick_length * np.sin(loc_theta[:, :, 0])
            y0 = -self.stick_length * np.cos(loc_theta[:, :, 0])
            x1 = self.stick_length * (np.sin(loc_theta[:, :, 0]) + np.sin(loc_theta[:, :, 1]))
            y1 = -self.stick_length * (np.cos(loc_theta[:, :, 0]) + np.cos(loc_theta[:, :, 1]))
            x2 = self.stick_length * (
                        np.sin(loc_theta[:, :, 0]) + np.sin(loc_theta[:, :, 1]) + np.sin(loc_theta[:, :, 2]))
            y2 = -self.stick_length * (
                        np.cos(loc_theta[:, :, 0]) + np.cos(loc_theta[:, :, 1]) + np.cos(loc_theta[:, :, 2]))
            # 记录端点速度
            v_x0 = self.stick_length * np.cos(loc_theta[:, :, 0]) * vel_theta[:, :, 0]
            v_y0 = self.stick_length * np.sin(loc_theta[:, :, 0]) * vel_theta[:, :, 0]
            v_x1 = self.stick_length * (
                        np.cos(loc_theta[:, :, 0]) * vel_theta[:, :, 0] + np.cos(loc_theta[:, :, 1]) * vel_theta[:, :,
                                                                                                       1])
            v_y1 = self.stick_length * (
                        np.sin(loc_theta[:, :, 0]) * vel_theta[:, :, 0] + np.sin(loc_theta[:, :, 1]) * vel_theta[:, :,
                                                                                                       1])
            v_x2 = self.stick_length * (
                        np.cos(loc_theta[:, :, 0]) * vel_theta[:, :, 0] + np.cos(loc_theta[:, :, 1]) * vel_theta[:, :,
                                                                                                       1] + np.cos(
                    loc_theta[:, :, 2]) * vel_theta[:, :, 2])
            v_y2 = self.stick_length * (
                        np.sin(loc_theta[:, :, 0]) * vel_theta[:, :, 0] + np.sin(loc_theta[:, :, 1]) * vel_theta[:, :,
                                                                                                       1] + np.sin(
                    loc_theta[:, :, 2]) * vel_theta[:, :, 2])

            loc[:, 0, 0] = x0.squeeze()
            loc[:, 1, 0] = y0.squeeze()
            loc[:, 0, 1] = x1.squeeze()
            loc[:, 1, 1] = y1.squeeze()
            loc[:, 0, 2] = x2.squeeze()
            loc[:, 1, 2] = y2.squeeze()

            vel[:, 0, 0] = v_x0.squeeze()
            vel[:, 1, 0] = v_y0.squeeze()
            vel[:, 0, 1] = v_x1.squeeze()
            vel[:, 1, 1] = v_y1.squeeze()
            vel[:, 0, 2] = v_x2.squeeze()
            vel[:, 1, 2] = v_y2.squeeze()
            # Add noise to observations
            # loc += np.random.randn(T_save, 2, self.n_sticks) * self.noise_var
            # vel += np.random.randn(T_save, 2, self.n_sticks) * self.noise_var
            return loc, vel,loc_theta,vel_theta, edges


if __name__ == '__main__':

    sim = PendulumSim()
    num=10
    loc_all = list()
    vel_all = list()
    edges = list()
    timestamps = list()

    for i in range(num):
        noise = np.random.randn(1, 3) * 0.0001
        t = time.time()
        T = 6000
        loc_0 = np.full((1, 3), np.pi / 2)
        loc_0 = np.mod(loc_0, 2 * np.pi)+noise
        print('initial loc:', loc_0)
        p_0 = np.zeros((1, 3))
        print('initial p: ', p_0)
        loc_dot_0 = np.zeros((1, 3))
        print('initial vel: ', loc_dot_0)


        loc, vel,loc_theta,vel_theta, edges = sim.sample_trajectory(T=T, sample_freq=100,loc_0=loc_0,p_0=p_0,loc_dot_0=loc_dot_0)
        print("Simulation time: {}".format(time.time() - t))
        vel_norm = np.sqrt((vel ** 2).sum(axis=1))
        print(loc_theta[0, :,:] )
        print(vel_theta[0, :, :])
        # print(loc)
        # print(vel)
        print("shape : ", loc.shape)
        print("shape : ", vel.shape)
        loc_all.append(loc)  # [49,2,5]
        vel_all.append(vel)  # [49,2,5]
        # edges.append(edges)

    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    vel_all = np.asarray(vel_all)
    # edges = np.stack(edges)
    suffix = "_pendulum"
    dir = os.path.join('MLE')
    np.save(os.path.join(dir, 'loc' + suffix + '.npy'), loc_all)
    np.save(os.path.join(dir, 'vel' + suffix + '.npy'), loc_all)




    # plt.figure()
    # axes = plt.gca()
    # axes.set_xlim([-5., 5.])
    # axes.set_ylim([-5., 5.])

    # 提取角度
    # theta_1 = loc_theta[:, 0, 0]
    # theta_2 = loc_theta[:, 0, 1]
    # theta_3 = loc_theta[:, 0, 2]
    # print('loc shape:',loc.shape)
    # print('theta shape:',theta_3.shape)
    # tN=loc.shape[0]
    # print('tN:',tN)
    # l=1
    # # 计算出关节坐标
    # CX1_A = np.zeros((1, tN))
    # # print('CX1_A shape:',CX1_A.shape)
    # CX1_B = CX1_A + l * np.sin(theta_1)
    # # print('CX1_B shape:',CX1_B.shape)
    # CY1_A = np.zeros((1, tN))
    # CY1_B = CY1_A - l * np.cos(theta_1)
    #
    # CX2_A = CX1_B
    # CX2_B = CX2_A + l * np.sin(theta_2)
    # CY2_A = CY1_B
    # CY2_B = CY2_A - l * np.cos(theta_2)
    #
    # CX3_A = CX2_B
    # CX3_B = CX3_A + l * np.sin(theta_3)
    # CY3_A = CY2_B
    # CY3_B = CY3_A - l * np.cos(theta_3)
    #
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # # 假设CX1_A, CX1_B, CY1_A, CY1_B等都是已经定义好的列表或numpy数组
    # # 这里只是一个示例，所以我没有定义这些变量
    #
    # n = 1
    # # fig, ax = plt.subplots()
    #
    # # plt.figure(figsize=(5, 4))  # 设置图形大小
    # plt.figure()  # Create a new figure
    # # for k in range(0, 49, 1):
    # # plt.clf()
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 0])
    # print(CX1_B[0])
    # print('shape:',CX1_A[0].shape)
    # # print('1: ',CX1_A[0][k], CX1_B[0][k])
    # # print('2: ',CX2_A[0][k], CX2_B[0][k])
    # # print('3: ',CX3_A[0][k], CX3_B[0][k])
    # # print('1: ', CY1_A[0][k], CY1_B[0][k])
    # # print('2: ', CY2_A[0][k], CY2_B[0][k])
    # # print('3: ', CY3_A[0][k], CY3_B[0][k])
    # # l1=(CX1_B[0][k]-CX1_A[0][k])**2+(CY1_B[0][k]-CY1_A[0][k])**2
    # # l2 = (CX2_B[0][k] - CX2_A[0][k]) ** 2 + (CY2_B[0][k] - CY2_A[0][k]) ** 2
    # # l3 = (CX3_B[0][k] - CX3_A[0][k]) ** 2 + (CY3_B[0][k] - CY3_A[0][k]) ** 2
    # # print('l1: ',l1)
    # # print('l2: ',l2)
    # # print('l3: ',l3)
    # plt.scatter(CX1_B[0],CY1_B[0])
    # plt.scatter(CX2_B[0],CY2_B[0])
    # plt.scatter(CX3_B[0],CY3_B[0])
    #
    # # ax.scatter(CX2_B[0][k], CY2_B[0][k])
    # # ax.scatter(CX3_B[0][k], CY3_B[0][k])
    #
    # for k in range(T-1):
    #     plt.plot([CX1_A[0][k], CX1_B[0][k]], [CY1_A[0][k], CY1_B[0][k]], color='k', linewidth=1)
    #     plt.plot([CX2_A[0][k], CX2_B[0][k]], [CY2_A[0][k], CY2_B[0][k]], color='k', linewidth=1)
    #     plt.plot([CX3_A[0][k], CX3_B[0][k]], [CY3_A[0][k], CY3_B[0][k]], color='k', linewidth=1)
    # for i in range(loc.shape[-1]):
    #     # plt.scatter(loc[:, 0, i], loc[:, 1, i])
    #     plt.plot(loc[0, 0, i], loc[0, 1, i], 'd')
    #     # ax.set_aspect('equal')
    # plt.grid(True)
    # plt.show()
    # #
    # plt.figure()
    # plt.gca()
    # plt.clf()
    # energies=[]
    # u=[]
    # k=[]
    # energies= [sim._energy(loc_theta[i][0], vel_theta[i][0])[0] for i in
    #             range(loc_theta.shape[0])]
    # u = [sim._energy(loc_theta[i][0], vel_theta[i][0])[1] for i in
    #             range(loc_theta.shape[0])]
    # k = [sim._energy(loc_theta[i][0], vel_theta[i][0])[2] for i in
    #             range(loc_theta.shape[0])]
    #
    # plt.plot(energies)
    # plt.plot(u)
    # plt.plot(k)
    #
    #
    # plt.show()










