import pickle
import numpy as np
import os

pkl_filename = "./motion_dance.pkl"
suffix="_motion_dance"
dir="./motion_dance"

with open(pkl_filename, "rb") as file:
    Edges, X = pickle.load(file)
    X = np.array(X, dtype=object)
    
    V = [] 
    for i in range(len(X)):
        V.append(X[i][1:] - X[i][:-1])    
        X[i] = X[i][:-1]

    for i in range(len(X)):
        X[i] = np.swapaxes(X[i], 0, 1)
        V[i] = np.swapaxes(V[i], 0, 1)
    
    X = np.array(X, dtype=object)

    x_lengths = [element.shape[1] for element in X] 
    longest_indices = np.argsort(x_lengths)[-7:]
    loc_test = [X[i] for i in longest_indices]
    vel_test = [V[i] for i in longest_indices]
    train_indices = np.setdiff1d(np.arange(len(X)), longest_indices)
    loc_train = [X[i] for i in train_indices]
    vel_train = [V[i] for i in train_indices]


    split_train_size = 50
    loc_train_set = []
    vel_train_set = []

    for loc_traj, vel_traj in zip(loc_train, vel_train):
        num_splits = loc_traj.shape[1] // split_train_size
        remainder = loc_traj.shape[1] % split_train_size


        for i in range(num_splits):
            loc_train_set.append(loc_traj[:, i*split_train_size:(i+1)*split_train_size, :])
            vel_train_set.append(vel_traj[:, i*split_train_size:(i+1)*split_train_size, :])
        
   
        if remainder >= 49:
            loc_train_set.append(loc_traj[:, num_splits*split_train_size:, :])
            vel_train_set.append(vel_traj[:, num_splits*split_train_size:, :])


    split_test_size = 100
    loc_test_set = []
    vel_test_set = []

    for loc_traj, vel_traj in zip(loc_test, vel_test):
        num_splits = loc_traj.shape[1] // split_test_size
        remainder = loc_traj.shape[1] % split_test_size

  
        for i in range(num_splits):
            loc_test_set.append(loc_traj[:, i*split_test_size:(i+1)*split_test_size, :])
            vel_test_set.append(vel_traj[:, i*split_test_size:(i+1)*split_test_size, :])
        

        if remainder >= 99:
            loc_test_set.append(loc_traj[:, num_splits*split_test_size:, :])
            vel_test_set.append(vel_traj[:, num_splits*split_test_size:, :])


    def sample_segments(loc_segments, vel_segments, max_len):
        sampled_loc_segments = []
        sampled_vel_segments = []
        sampled_indices_list = []
        for log_traj, vel_traj in zip(loc_segments, vel_segments):
            sampled_loc_node_traj=[]
            sampled_vel_node_traj=[]
            time_node_traj=[]
            for loc_node_traj,vel_node_traj in zip(log_traj, vel_traj):
                n = np.random.randint(30, 42) 
                sampled_indices = np.random.choice(max_len, n, replace=False)
                sampled_indices=np.sort(sampled_indices)
                # print(sampled_indices)
                sampled_loc_node_traj.append(loc_node_traj[sampled_indices,:])
                sampled_vel_node_traj.append(vel_node_traj[sampled_indices,:])
                time_node_traj.append(sampled_indices)
            sampled_loc_segments.append(sampled_loc_node_traj)
            sampled_vel_segments.append(sampled_vel_node_traj)
            sampled_indices_list.append(time_node_traj)
        return sampled_loc_segments, sampled_vel_segments, sampled_indices_list


    


    def sample_and_concat_test_segments(loc_segments, vel_segments):
        sampled_loc_segments = []
        sampled_vel_segments = []
        sampled_indices_list = []
        for log_traj, vel_traj in zip(loc_segments, vel_segments):
            sampled_loc_node_traj=[]
            sampled_vel_node_traj=[]
            time_node_traj=[]
            for loc_node_traj,vel_node_traj in zip(log_traj, vel_traj):
                n = np.random.randint(30, 42) 
                first_sampled_indices = np.random.choice(50, n, replace=False)
                first_sampled_indices=np.sort(first_sampled_indices)
                last_sampled_indices=np.random.choice(50, 40, replace=False)+50
                last_sampled_indices=np.sort(last_sampled_indices)
                combined_indices = np.concatenate((first_sampled_indices, last_sampled_indices))
               
                sampled_loc_node_traj.append(loc_node_traj[combined_indices,:])
                sampled_vel_node_traj.append(vel_node_traj[combined_indices,:])
                time_node_traj.append(combined_indices)
            sampled_loc_segments.append(sampled_loc_node_traj)
            sampled_vel_segments.append(sampled_vel_node_traj)
            sampled_indices_list.append(time_node_traj)
                

        return sampled_loc_segments, sampled_vel_segments, sampled_indices_list

   
    sampled_loc_test_set, sampled_vel_test_set, test_times = sample_and_concat_test_segments(loc_test_set, vel_test_set)
    sampled_loc_train_set, sampled_vel_train_set, train_times = sample_segments(loc_train_set, vel_train_set, split_train_size)

    adj_matrix = np.zeros((31, 31), dtype=float)  
    for edge in Edges:     
        adj_matrix[edge[0], edge[1]] = 1
    # print(adj_matrix)  
    train_num=len(sampled_loc_train_set)
    train_edges= np.zeros((train_num, 31, 31), dtype=int)
    for i in range(train_num):     
            train_edges[i] = adj_matrix



    test_num=len(sampled_loc_test_set)
    test_edges= np.zeros((test_num, 31, 31), dtype=int)
    for i in range(test_num):     
        test_edges[i] = adj_matrix

    np.save(os.path.join(dir,"loc_test"+suffix+".npy"),sampled_loc_test_set)
    np.save(os.path.join(dir,"vel_test"+suffix+".npy"),sampled_vel_test_set)
    np.save(os.path.join(dir,"times_test"+suffix+".npy"),test_times)
    np.save(os.path.join(dir,"edges_test"+suffix+".npy"),test_edges)
    np.save(os.path.join(dir,"loc_train"+suffix+".npy"),sampled_loc_train_set)
    np.save(os.path.join(dir,"vel_train"+suffix+".npy"),sampled_vel_train_set)
    np.save(os.path.join(dir,"times_train"+suffix+".npy"),train_times)
    np.save(os.path.join(dir,"edges_train"+suffix+".npy"),train_edges)
    

  
