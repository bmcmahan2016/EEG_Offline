#!/usr/bin/python
 
import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt
import numpy as np
import struct
import msgpack
from scipy.signal import butter, lfilter
from scipy import signal
import eegclean


def new_func():
    print('I am a new function')
 
 
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
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Binarize(data):
    trace_max = np.max(data)
    trace_min = np.min(data)
    median = ( trace_max + trace_min ) / 2
    data -= median
    A = ( trace_max - trace_min ) / 2
    data /= A
    data = np.round(data)
    return data

def SerializeTrials(trial_idx):
    trial_counter = 0
    last_idx_in_trial = False
    serialized_trial = np.zeros((len(trial_idx)))
    for _ in range(len(trial_idx)):
        if trial_idx[_] == 1:
            serialized_trial[_] = trial_idx[_] + trial_counter
            last_idx_in_trial = True 
        elif ( trial_idx[_] == 0 ) and ( last_idx_in_trial ):
            trial_counter += 1
            last_idx_in_trial = False
    return serialized_trial

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
    Will preform an offline preprocessing of EEG data
    '''
    # filter parameters
    fs       = 125.0     # sample frequency
    lowcut   = 5         # lower cut-off frequency
    highcut  = 50        # higher cut-off frequency
    f0       = 60.0      # frequency to be removed from signal
    Q = 30               # quality factor
    w0 = f0/(fs/2)

    #downsample EEG data to 125Hz
    eeg1 = eeg1[::8]
    eeg2 = eeg2[::8]

    # filter the EEG data
    # first remove 60Hz noise
    eeg1_filt = []
    eeg2_filt = []
    b, a = signal.iirnotch(w0, Q)
    for _ in range(8):
        eeg1_filt.append(100*_+butter_bandpass_filter(lfilter(b, a, eeg1[:,_]), lowcut, highcut, fs, order=5))    # will have shape (8,_)
        eeg2_filt.append((100*_+50)+butter_bandpass_filter(lfilter(b, a, eeg2[:,_]), lowcut, highcut, fs, order=5))    # will have shape (8,_)
    '''
    eeg1_filt = lfilter(b, a, eeg1)
    eeg2_filt = lfilter(b, a, eeg2)
    eeg1_filt_filt = butter_bandpass_filter(eeg1_filt, lowcut, highcut, fs, order=5)
    eeg2_filt_filt = butter_bandpass_filter(eeg2_filt, lowcut, highcut, fs, order=5)
    '''

    eeg_data = np.zeros((len(eeg1_filt[0]), 16))
    eeg_data[:, :8] = np.array(eeg1_filt).T #np.array(eeg1)
    eeg_data[:, 8:] = np.array(eeg2_filt).T#rials#np.array(eeg2)
    

    return eeg_data

def PartitionTrials(pos_cursor):
    # downsample cursor position so it lines up with EEG data
    pos_cursor = pos_cursor[::8]
    x_pos = Binarize(pos_cursor[:,0])
    y_pos = Binarize(pos_cursor[:,1])
    state = x_pos + 2*y_pos

    # extract serialized trials for each condition of interest
    serialized_trials = {}
    idx_of_interest = (state==-2)
    serialized_trials['down'] = SerializeTrials(idx_of_interest)
    idx_of_interest = (state==-1)
    serialized_trials['left'] = SerializeTrials(idx_of_interest)
    idx_of_interest = (state==1)
    serialized_trials['right'] = SerializeTrials(idx_of_interest)
    idx_of_interest = (state==2)
    serialized_trials['up'] = SerializeTrials(idx_of_interest)

    return serialized_trials, state




def main():
    # file containing the data
    database = "experiment12.db"#"experiment9_10.db"
    # create a database connection
    conn = create_connection(database)
    cur = conn.cursor()
    with conn:
        cols = cur.execute("PRAGMA table_info(signals_table)").fetchall()
        signals = {}    #dictionary to hold symbols
        for id in cols:
            print(id, '\n')
            signals[id[1]] = ()
        print('signals', signals)

        eeg1 = []
        eeg2 = []
        save_tag = []
        
        eeg_i = cur.execute("SELECT eegsignali FROM signals_table").fetchall()
        eeg_ii = cur.execute("SELECT eegsignalii FROM signals_table").fetchall()
        pos_cursor_b = cur.execute("SELECT pos_target FROM signals_table").fetchall()

        eeg_i = UnpackBLOBS(eeg_i)
        eeg_ii = UnpackBLOBS(eeg_ii)
        pos_cursor_b = UnpackBLOBS(pos_cursor_b)
        
        eeg_data = FilterEEG(eeg_i, eeg_ii)

        serialized_trials, state = PartitionTrials(pos_cursor_b)


        plt.plot(state)
        plt.plot(serialized_trials['up'])
        plt.plot(serialized_trials['down'])
        plt.legend(['state', 'Up Reach Trials (Serialzed)', 'Down Reach Trials (Serialzed)'])
        num_this_trial = int( np.max(serialized_trials['down']) ) + 1
        for _ in range(1, num_this_trial):
            plt.figure()
            for channel in range(16):
                plt.plot(eeg_data[serialized_trials['down']==_,channel])
            plt.title('Up Trial #'+str(_))

        plt.figure()
        time_stamps = np.array([0.008*i for i in range(len(eeg_data))])
        for channel in range(16):
            plt.plot(time_stamps, eeg_data[:,channel])
        plt.title('All EEG data acquired')
        
        plt.show()



        #print(int.from_bytes(our_thing, byteorder='little'))
        print('#'*50)
        assert False
        plt.show()
        #TEMPORARY STOP HERE

        #print(eeg_signal[0].decode('utf-8'))
        #curr_trial = np.array(np.squeeze(cur.execute("SELECT r_i1_curr_trial FROM signals").fetchall()))
        time_stamps = np.array([0.008*i for i in range(len(eeg_signal))])
        eeg_data_flag = np.abs(np.diff(eeg_signal))
        eeg_signal_downsampled = eeg_signal[:-1][eeg_data_flag>0]
        time_stamps_downsampled = time_stamps[:-1][eeg_data_flag>0]
    #plt.plot(np.linspace(0,60, len(eeg_signal_downsampled)), eeg_signal_downsampled)
    
    #EEG DATA SIGNAL
    plt.figure()
    plt.plot(time_stamps, eeg_signal)
    plt.plot(time_stamps_downsampled, eeg_signal_downsampled)
    plt.title('EEG Signal')
    plt.legend([ 'Raw EEG from Logger', 'Downsampled EEG'])
    plt.xlabel('Time (seconds)')

    #EEG DATA BY TRIAL
    plt.figure()
    plt.plot(time_stamps_downsampled, eeg_signal_downsampled)
    #plt.plot(time_stamps, curr_trial)
    plt.title('EEG During Trials')
    plt.show()


if __name__ == '__main__':
    main()