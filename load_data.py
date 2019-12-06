#!/usr/bin/python
 
import sqlite3
from sqlite3 import Error
import matplotlib.pyplot as plt
import numpy as np
import struct
import msgpack
from scipy.signal import butter, lfilter
from scipy import signal
import os
from sklearn.model_selection import train_test_split

class DataManager():
    CLASS_MAP = {0 : 'Down', # map numerical classes to reach directions
                1 : 'Left',
                2 : 'Up',
                3 : 'Right'}

    # pass all global parameters to instantion of class
    def __init__(self, collection_type="non_gel", lowcut=5.0, highcut=50.0, fs=125.0,
        filter_order=5, bin_size=1, trial_delay=0, include_center=False, sliding_window=False,
        equalize_proportions=False, include_classes=["Down", "Left", "Up", "Right"], combine_features=False):
        # filter params
        self.collection_type = collection_type
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.filter_order = filter_order

        # data set params
        self.bin_size = bin_size
        self.trial_delay = trial_delay
        self.include_center = include_center
        self.sliding_window = sliding_window
        self.equalize_proportions = equalize_proportions
        self.include_classes = include_classes
        self.combine_features = combine_features

    def ButterBandpass(self, lowcut, highcut):
        '''
        constructs a bandpass filter

        PARAMETERS
        -lowcut: the lowest frequency to be allowed
        -highcut: the highest fequency to bel allowed
        -fs: sampling rate of the signal
        -order: by default this is 5
        '''
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(self.filter_order, [low, high], btype='band')
        return b, a

    def EEGFilter(self, data, lowcut, highcut):
        '''
        EEGFilter will construct a notch filter to remove 60Hz noise
        from the EEG signal and will then construct a bandpass to 
        filter the EEG signal between 5 and 50 Hz
        '''

        # perform bandpass filering on the data
        band_b, band_a = self.ButterBandpass(lowcut, highcut)
        self.band_z = signal.lfilter_zi(band_b, 1) # state is used for sample-by-sample filtering

        result = np.zeros(data.size)
        for i, x in enumerate(data):
            result[i], self.band_z = lfilter(band_b, band_a, [x], zi=self.band_z)

        return result

    def PlotFreqSpectrum(self, eeg_data):
        for c in range(16):
            plt.figure("fourier")
            plt.magnitude_spectrum(eeg_data[:,c], Fs=self.fs)
            plt.title("Fourier Transform")
            plt.figure("welch")
            win = 4*self.fs
            freqs, psd = signal.welch(eeg_data[:,c], fs=self.fs, nperseg=win)
            plt.plot(freqs, psd)
            plt.title("Power Spectral Density")
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power spectral density (V^2 / Hz)')
        
        plt.show()

    def FilterEEG(self, eeg1, eeg2, lowcut, highcut, plot_freq=False):
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
        eeg1_filt = []
        eeg2_filt = []
        for c in range(8):
            eeg1_filt.append(self.EEGFilter(eeg1[:,c], lowcut, highcut))
            eeg2_filt.append(self.EEGFilter(eeg2[:,c], lowcut, highcut))

        # concatenate all 16 EEG channels into a single NumPy array
        eeg_data = np.zeros((len(eeg1_filt[0]), 16))
        eeg_data[:, :8] = np.array(eeg1_filt).T 
        eeg_data[:, 8:] = np.array(eeg2_filt).T
        
        if (plot_freq):
            self.PlotFreqSpectrum(eeg_data)

        return eeg_data

    def Binarize(self, data):
        trace_max = np.max(data)
        trace_min = np.min(data)
        median = ( trace_max + trace_min ) / 2
        data -= median
        A = ( trace_max - trace_min ) / 2
        if A != 0:
            data /= A
            data = np.round(data)
        return data

    def PartitionTrials(self, target_pos):
        '''
        PartitionTrials will partition the continous EEG data into discrete trials 
        of length 100 time stamps that correspond to a reach to one of four possible 
        targets: up, down, left, or right
        PAREMETERS
        -target_pos is a numpy array of shape (num_timesteps, 2) where the first
            column corresponds to the x-position of the on screen target and the 
            second column corresponds to the y-position
        RETURNS
        -training_classes: the target class for each time tick
        '''
        # downsample target position so it lines up with EEG data
        target_pos = target_pos[::8]
        # binarize the target position so it is either +/-1
        target_pos_x = self.Binarize(target_pos[:,0])
        target_pos_y = self.Binarize(target_pos[:,1])
        # create a state variable that maps each reach direction to an integer
        # mapping from reach direction to integer:
        # down=0, left=1, up=2, and right=3
        reach_state_mapping = {(0, -1) : 0, # down
                            (-1, 0) : 1, # left
                            (0, 1) : 2,  # up
                            (1, 0) : 3,  # right
                            (0, 0) : 4}  # center
        reach_state = np.array([reach_state_mapping[(x, y)] for x, y in zip(target_pos_x, target_pos_y)])

        return reach_state

    def GetExperimentData(self, eeg_data, target_classes):
        '''
        GetExperimentData will return training data and classes in specified bin sizes for a single experiment trial
        PARAMETERS
        -eeg_data: filtered eeg data with shape (num_time_steps, 16)
        -target_classes: the target class for each time step with shape (num_time_steps, 16)
        -bin_size: the number of time_steps to include in one data sample for classifying a reach direction
        -trial_delay: the amount of time in ms to skip after the start of a new reach direction
        -include_center: include reaches back to the center in training data if true, otherwise discard them
        RETURNS:
        -training_data: binned eeg_data with reaches to the center removed - has shape (n, bin_size, 16)
        -training_classes: the class associated with each row of training data - has shape (n,)
        '''
        eeg_data = eeg_data[15*125:-5*125] # remove first 15 seconds and last 5 seconds of data
        target_classes = target_classes[15*125:-5*125]

        training_data = []
        training_classes = []

        center_reach_map = {0:2, 1:3, 2:0, 3:1}

        b = self.bin_size
        inc = 5 if self.sliding_window else b
        cur_target = target_classes[0]
        prev_target = cur_target
        start_idx = 0
        skip_zeros = 0
        # while skip_zeros < len(target_classes) and np.sum(eeg_data[skip_zeros]) == 0:
        #     skip_zeros += 1
        # print(skip_zeros)
        for i in range(skip_zeros, len(target_classes)):
            if target_classes[i] != cur_target:
                true_target = cur_target
                if cur_target == 4:
                    if not self.include_center:
                        prev_target = cur_target
                        cur_target = target_classes[i]
                        start_idx = i
                        continue # skip all reaches to center
                    true_target = center_reach_map[prev_target]

                x = eeg_data[start_idx:i]
                s = self.trial_delay
                next_sample = []
                next_class = []
                while s + b <= x.shape[0]:
                    # data_sample = x[s:s+b]
                    # if np.sum(data_sample) == 0:
                    #     print(start_idx)
                    next_sample.append(x[s:s+b])
                    next_class.append(true_target)
                    s += inc

                if len(next_sample) != 0:
                    training_data.append(np.array(next_sample))
                    training_classes.append(np.array(next_class))

                prev_target = cur_target
                cur_target = target_classes[i]
                start_idx = i

        return np.array(training_data), np.array(training_classes)

    def UnpackBLOBS(self, data):
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

    def CreateDBConnection(self, db_file):
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

    def GetData(self, num_experiments, plot_freq=False):
        '''
        GetData will return training data and classes for a specified number of experiments
        PARAMETERS
        -num_experiments: the number of experiments to use for training data
        -bin_size: the number of time_steps to include in one data sample for classifying a reach direction
        -trial_delay: the amount of time in ms to skip after the start of a new reach direction
        -include_center: include reaches back to the center in training data if true, otherwise discard them
        RETURNS:
        -training_data: binned eeg_data with reaches to the center removed - has shape (n, bin_size, 16)
        -training_classes: the class associated with each row of training data - has shape (n,)
        '''
        training_data = []
        training_classes = []

        data_folder = os.path.join('data', self.collection_type)
        num_db_files = len(os.listdir(data_folder)) # remove one for readme file

        # this assumes that db files are labeled experiment_1.db, experiment_2.db, etc
        for db_num in range(min(num_db_files, num_experiments)):
            database = os.path.join(data_folder, 'experiment_' + str(db_num + 1) + '.db')

            # create a database connection to load the data
            conn = self.CreateDBConnection(database)
            cur = conn.cursor()
            with conn:
                # select the binary eeg data from the database
                eeg_i_b = cur.execute("SELECT eegsignali FROM signals_table").fetchall()
                eeg_ii_b = cur.execute("SELECT eegsignalii FROM signals_table").fetchall()
                
                # select the binary reach target position from the database
                target_pos_b = cur.execute("SELECT pos_target FROM signals_table").fetchall()

                # unpack the binary data to NumPy arrays
                eeg_i = self.UnpackBLOBS(eeg_i_b)
                eeg_ii = self.UnpackBLOBS(eeg_ii_b)
                target_pos = self.UnpackBLOBS(target_pos_b)

                # filter the EEG data
                if not self.combine_features:
                    eeg_data = self.FilterEEG(eeg_i, eeg_ii, self.lowcut, self.highcut, plot_freq=plot_freq)
                else:
                    eeg_data_1 = self.FilterEEG(eeg_i, eeg_ii, 8.0, 12.0, plot_freq=plot_freq)
                    eeg_data_2 = self.FilterEEG(eeg_i, eeg_ii, 12.0, 30.0, plot_freq=plot_freq)
                    eeg_data_3 = self.FilterEEG(eeg_i, eeg_ii, 30.0, 60.0, plot_freq=plot_freq)
                    eeg_data = np.stack([eeg_data_1, eeg_data_2, eeg_data_3], axis=1)

                # partition the EEG data into trials based on target position
                target_classes = self.PartitionTrials(target_pos)

                data, classes = self.GetExperimentData(eeg_data, target_classes)
                if len(training_data) == 0:
                    training_data = data
                    training_classes = classes
                else:
                    training_data = np.concatenate((training_data, data))
                    training_classes = np.concatenate((training_classes, classes))
        
        X_train, X_test, y_train, y_test = train_test_split(training_data, training_classes, test_size=0.2, random_state=0) 
        X_train, y_train, class_map = self.UpdateClassificationTask(np.vstack(X_train), np.hstack(y_train))
        X_test, y_test, _ = self.UpdateClassificationTask(np.vstack(X_test), np.hstack(y_test))
        if self.combine_features:
            X_train = np.swapaxes(X_train, 1, 2)
            X_test = np.swapaxes(X_test, 1, 2)
        return X_train, X_test, y_train, y_test, class_map

    def UpdateClassificationTask(self, training_data, training_classes):
        include_classes = set(self.include_classes)
        class_map = {}
        id_map = {}
        class_count = {}
        for idx, name in enumerate(include_classes):
            class_map[name] = idx
            id_map[idx] = name
            class_count[name] = 0

        include_indices = []
        for idx in range(len(training_classes)):
            name = DataManager.CLASS_MAP[training_classes[idx]]
            if name in include_classes:
                class_count[name] += 1
                if (len(include_classes) < 4):
                    training_classes[idx] = class_map[name]
                    include_indices.append(idx)

        if (len(include_classes) < 4): # update classification task to include only these classes
            training_data = training_data[include_indices]
            training_classes = training_classes[include_indices]
        else:
            id_map = self.CLASS_MAP

        if self.equalize_proportions:
            include_indices = []
            nums = [val for _, val in class_count.items()]
            min_count = min(nums)
            for key in class_count:
                class_count[key] = 0
            for idx in range(len(training_classes)):
                if class_count[id_map[training_classes[idx]]] < min_count:
                    class_count[id_map[training_classes[idx]]] += 1
                    include_indices.append(idx)

            training_data = training_data[include_indices]
            training_classes = training_classes[include_indices]

        return training_data, training_classes, id_map

def PlotData(num_rows, num_cols, plot_data, plot_labels):
    '''
    PlotData will plot subplots of the provided data
    PARAMETERS
    -num_rows: the number of rows in the subplot
    -num_cols: the number of columns in the subplot
    -plot_data: the data to be plotted in each subplot
    -plot_labels: the class label for each data sample in plot_data, must be the same length as plot_data
    '''
    i = 0
    stop_var = False
    while (i < len(plot_labels)):
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True, sharey=True)
        for row in ax:
            for col in row:
                if i >= len(plot_labels):
                    stop_var = True
                    break
                col.plot(plot_data[i])
                col.title.set_text("Reach direction: {}".format(DataManager.CLASS_MAP[plot_labels[i]]))
                i += 1
            if stop_var:
                break

        plt.tight_layout()
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time ticks")
        plt.ylabel("EEG Recording per Channel")
        plt.show()

def main():
    # experiment*.db is the file containing the sample experimental data
    db_file_limit = 1 # set temporary limit on number of experiments to use

    data = DataManager(bin_size=50)
    training_data, training_classes = data.GetData(db_file_limit, plot_freq=True)
    print("Training data shape", training_data.shape)
    print("Training class shape", training_classes.shape)

    uniques, counts = np.unique(training_classes, return_counts=True)
    total_trials = np.sum(counts)
    trial_freq = {DataManager.CLASS_MAP[uniques[0]] : (counts[0], float(counts[0])/total_trials),
                  DataManager.CLASS_MAP[uniques[1]] : (counts[1], float(counts[1])/total_trials),
                  DataManager.CLASS_MAP[uniques[2]] : (counts[2], float(counts[2])/total_trials),
                  DataManager.CLASS_MAP[uniques[3]] : (counts[3], float(counts[3])/total_trials)}
    print("Relative trial frequency", trial_freq)

    plot_map = {0 : False, 1 : False, 2 : False, 3 : False}
    plot_data = []
    plot_labels = []
    for i in range(len(training_classes)):
        if plot_map[training_classes[i]]:
            continue
        plot_data.append(training_data[i])
        plot_labels.append(training_classes[i])
        plot_map[training_classes[i]] = True
        if len(plot_data) == 4:
            break

    PlotData(2, 2, plot_data, plot_labels)

if __name__ == '__main__':
    main()