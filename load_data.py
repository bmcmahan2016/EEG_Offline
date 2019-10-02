#!/usr/bin/python
 
import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt
import numpy as np
import struct
import msgpack
from scipy.signal import butter, lfilter
from scipy import signal
#import eegclean


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None 

def butter_bandpass(lowcut, highcut, fs, order=5):
    '''
    constructs a bandpass filter

    PARAMETERS
    -lowcut: the lowest frequency to be allowed
    -highcut: the highest fequency to bel allowed
    -fs: sampling rate of the signal
    -order: by default this is 5
    '''
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def Notch60(fs):
    '''
    Notch60 will construct filter coefficients to remove 60Hz noise from data 
    PARAMETERS
    -fs: sampling frequency of signal to be filtered
    -f_stop: frequency of 
    RETURNS:
    -b, a: filter coefficients
    '''
    f_stop = 60.0        # remove 60Hz noise from signal
    Q = 30               # quality factor
    w0 = f_stop/(fs/2)
    b, a = signal.iirnotch(w0, Q)
    return b, a

def EEGFilter(data, lowcut=5.0, highcut=50.0, fs=125.0, order=5):
    '''
    EEGFilter will construct a notch filter to remove 60Hz noise
    from the EEG signal and will then construct a bandpass to 
    filter the EEG signal between 5 and 50 Hz
    '''
    # first notch out 60Hz noise
    b, a = Notch60(fs)
    y_notched = lfilter(b, a, data)
    # next perform bandpass filering on the data
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y_banded = lfilter(b, a, y_notched)
    return y_banded

def Binarize(data):
    trace_max = np.max(data)
    trace_min = np.min(data)
    median = ( trace_max + trace_min ) / 2
    data -= median
    A = ( trace_max - trace_min ) / 2
    data /= A
    data = np.round(data)
    return data

def SerializeTrials(trial_idxs):
    '''
    SerializeTrials will enumerate all reach trials for a given direction. The 
    output will be a numpy array with shape (time_steps, 1) where the value at 
    each timestep is equal to zero if a reach in the specified direction is not 
    occuring at that instant in time or will be an integer, n, corresponding to 
    the trial number for how many reaches have occured in that direction.
    PAREMETERS
    -trial_idxs: a 1D NumPy array with length equal to the number of time steps 
        in the data. This array should be one at every point in time that belongs 
        to a reach trial in the direction of interest and zero at all other points 
        in time.
    RETURNS
    -enumerated_trials: A 1D NumPy array with length equal to the number of time 
        steps in the eeg data. The values of EnumeratedTrials will be zero at 
        time steps during which the reach is not in the specified direction, or 
        will be a positive integer at time steps where a reach in the specified 
        direction is occuring. The integer is determined by the number of previous 
        reach trials in the same direction. 
    '''
    trial_counter = 0           # will count the number of trials corresponding to a specific reach
    last_idx_in_trial = False   #flag indiciating if the last time step was in a trial
    num_t_steps = len(trial_idxs)
    # enumerated_trials will indicate the trial number each time step belongs to or zero for 
    # time steps corresponding to reaches in other directions
    enumerated_trials = np.zeros((num_t_steps))
    #loop over all timesteps in data
    for t_step in range(num_t_steps):
        # if the current time step in the data is part of a reach trial then
        # add this to the serialized trial
        if trial_idxs[t_step] == 1:
            enumerated_trials[t_step] = trial_idxs[t_step] + trial_counter
            last_idx_in_trial = True 
        # if the current time step is not part of a reach trial but 
        # the past time step was increment the trial counter
        elif ( trial_idxs[t_step] == 0 ) and ( last_idx_in_trial ):
            trial_counter += 1
            last_idx_in_trial = False
    return enumerated_trials

def UnpackBLOBS(data):
    '''
    Will unpack data stored as a blob. Incoming data has 
    shape (num_time_steps, -1)
    returned data will be a numpy array with shape (num_time_steps, -1)
    '''
    # determine number of time steps in the data
    num_t_steps = len(data)
    unpacked_data = []           # will hold the unpacked data
    #loop through 
    for _ in range(num_t_steps):
        unpacked_data.append(msgpack.unpackb(data[_][0]))
    unpacked_data = np.array(unpacked_data) #list w/ shape (num_t_steps,) --> array w/ shape (num_t_steps, -1)
    return unpacked_data

def FilterEEG(eeg1, eeg2):
    '''
    Will preform an offline preprocessing of EEG data to remove
    60Hz noise and filter data with bandpass of 5 to 50Hz
    RETURNS
    eeg_data: NumPy array containing eeg data with shape (num_time_steps, 16)
    '''

    #downsample EEG data from 1,000Hz to 125Hz
    eeg1 = eeg1[::8]
    eeg2 = eeg2[::8]

    # filter the EEG data
    # first remove 60Hz noise
    eeg1_filt = []
    eeg2_filt = []
    for _ in range(8):
        eeg1_filt.append(EEGFilter(eeg1[:,_]))
        eeg2_filt.append(EEGFilter(eeg2[:,_]))

    # concatenate all 16 EEG channels into a single NumPy array
    eeg_data = np.zeros((len(eeg1_filt[0]), 16))
    eeg_data[:, :8] = np.array(eeg1_filt).T 
    eeg_data[:, 8:] = np.array(eeg2_filt).T
    
    return eeg_data

def PartitionTrials(target_pos):
    '''
    PartitionTrials will partition the continous EEG data into discrete trials 
    of length 100 time stamps that correspond to a reach to one of four possible 
    targets: up, down, left, or right
    PAREMETERS
    -target_pos is a numpy array of shape (num_timesteps, 2) where the first
        column corresponds to the x-position of the on screen target and the 
        second column corresponds to the y-position
    '''
    # downsample target position so it lines up with EEG data
    target_pos = target_pos[::8]
    # binarize the target position so it is either +/-1
    target_pos_x = Binarize(target_pos[:,0])
    target_pos_y = Binarize(target_pos[:,1])
    # create a state variable that maps each reach direction to an integer
    # mapping from reach direction to integer:
    # up=2, down=-2, left=-1, and right=1
    reach_state = target_pos_x + 2*target_pos_y

    # create a dictionary to serialize trials
    enumerated_trials = {}
    down_idxs = (reach_state==-2)
    enumerated_trials['down'] = SerializeTrials(down_idxs)
    left_idxs = (reach_state==-1)
    enumerated_trials['left'] = SerializeTrials(left_idxs)
    right_idxs = (reach_state==1)
    enumerated_trials['right'] = SerializeTrials(right_idxs)
    up_idxs = (reach_state==2)
    enumerated_trials['up'] = SerializeTrials(up_idxs)

    return enumerated_trials, reach_state

def MakeTrainingData(enumerated_trials, eeg_data, reach_direction='down', Verbose=False):
    '''
    MakeTrainingData will create a Training set of EEG data for a given reach direction (to be used
    as the trianing label).
    PARAMETERS
    -enumerated_trials: dictionary containing NumPy arrays, the keys should be labels
    -eeg_data: the downsampled and filtered EEG data
    -Verbose: if true then EEG data is plotted. Will plot a figure for each data point
        so this should only be set to true for small data sets where debugging is desired
    RETURNS
    -training_data: NumPy array
    '''
    # plot seperate EEG trials under each condition
    # number of down reach trials
    num_down_trials = int( np.max(enumerated_trials[reach_direction]) ) + 1
    num_channels = 16
    len_of_trial = len(eeg_data[enumerated_trials[reach_direction]==1,0])

    training_data = np.zeros((num_down_trials, len_of_trial, num_channels))
    for trial_num in range(1, num_down_trials):
        training_data[trial_num, :] = eeg_data[enumerated_trials[reach_direction]==trial_num,:]
        if Verbose:
            plt.figure()
            for channel in range(16):
                plt.plot(eeg_data[enumerated_trials[reach_direction]==trial_num,channel])
            plt.title(str(reach_direction) + 'Trial #' + str(trial_num))
    return training_data


def main():
    # experiment12.db is the file containing the sample experimental data
    database = "experiment12.db"
    # create a database connection to load the data
    conn = create_connection(database)
    cur = conn.cursor()
    with conn:
        cols = cur.execute("PRAGMA table_info(signals_table)").fetchall()

        # select the binary eeg data from the database
        eeg_i_b = cur.execute("SELECT eegsignali FROM signals_table").fetchall()
        eeg_ii_b = cur.execute("SELECT eegsignalii FROM signals_table").fetchall()
        # select the binary reach target position from the database
        target_pos_b = cur.execute("SELECT pos_target FROM signals_table").fetchall()

        #unpack the binary data to NumPy arrays
        eeg_i = UnpackBLOBS(eeg_i_b)
        eeg_ii = UnpackBLOBS(eeg_ii_b)
        target_pos = UnpackBLOBS(target_pos_b)

        #filter the EEG data
        eeg_data = FilterEEG(eeg_i, eeg_ii)

        #partition the EEG data into trials based on target position
        enumerated_trials, state = PartitionTrials(target_pos)

        #plot some results
        plt.figure()
        plt.plot(state)
        plt.plot(enumerated_trials['up'])
        plt.plot(enumerated_trials['down'])
        plt.legend(['state', 'Up Reach Trials (Enumerated)', 'Down Reach Trials (Enumerated)'])
        plt.title('Enumeration of EEG Reach Trials')


        # below code will create training data 
        training_data_tmp = []
        for reach_direction in ['down', 'up', 'left', 'right']:
            training_data_tmp.append(MakeTrainingData(enumerated_trials, eeg_data, reach_direction=reach_direction))

        num_data_points = 0
        for _ in range(len(training_data_tmp)):
            num_data_points += len(training_data_tmp[_])
            len_of_trial = len(training_data_tmp[_][0])
            print(len_of_trial)

        training_data = np.zeros((num_data_points, len_of_trial, 16))

        print('training_data', training_data.shape)
        plt.show()


if __name__ == '__main__':
    main()