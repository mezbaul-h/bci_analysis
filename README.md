# UnicornBenchmarkingDataset
These EEG data were collected using wet electrodes of 8 channel unicorn device on a rest-movement paradigm of a motor imagery BCI

What is inside this dataset?

    There are 20 subjects in this dataset. Each subject completed at least 3 runs of the experiment. The data for each subject was recorded in a single session, i.e. in session 1. Only for the first subject, i.e.       P001 the second session (S002) data was given while the session 1 (S001) data was discarded due to noise. This information is coded in the filename itself. For example, if we take filename: 'sub-P007_ses-           S001_task-Ronak_run-004_eeg.xdf'; this means subject P007, session S001, run 4. 
    
    Now we can see that the data files are in .xdf format. This can be read in MATLAB using the command: load_xdf('filename.xdf'); where 'filename' needs to be replaced by the actual filename. Similarly, it can         also be read in Python using the command: lsl_stream, _ = pyxdf.load_xdf(filename, stream_to_load, synchronize_clocks=True)

Here is how pyxdf.load_xdf works:

      pyxdf: This refers to a Python library or module (pyxdf) used for handling XDF (Extensible Data Format) files, which are commonly used in neuroscientific data recording and analysis.
      
      load_xdf: load_xdf is a function provided by the pyxdf module. It is used to load data from an XDF file (filename) and return it in a structured format suitable for further processing.
      
      Arguments:
      
      filename: This is the path to the XDF file you want to load.
      stream_to_load: This specifies which streams from the XDF file you want to load. It can be a string specifying the stream name or a list of stream names.
      synchronize_clocks=True: This optional argument indicates whether to synchronize clocks based on the timestamps in the XDF file. When set to True, it attempts to align timestamps across different streams.
      Return Values:
      
      lsl_stream: This variable will contain the loaded data stream(s) from the XDF file. The exact structure and content of lsl_stream depend on the data format and organization within the XDF file.
      _: In Python convention, _ is often used as a placeholder variable for values that are intentionally ignored or not needed. In this case, it indicates that we're not interested in the second return value            (which might be metadata or additional information).

If you load the .xdf file in MATLAB you will see there is a cell array of 1x2 dimension named 'streams' in the workspace. One of the cells inside 'streams' contains the information about the event markers of the experiment and the other cell contains the information about the sensor data such as EEG and other sensors. To see which type of cell it is you need to go inside it where you will find a variable called 'info'. Now if you go inside 'info' there is a variable called 'type'. If the 'type' is 'Markers' then it contains event markers, and if the 'type' is 'Data' then it contains sensor information. The variable 'info' will contain a lot of information which can be ignored if it is not relevant.

So, if the cell is event marker there will be two other variables inside called 'time_series' and 'time_stamps' both of which are roughly  1x152 dimensions. The 'time_series' is a cell array containing the named markers of the experiment such as trial_start 0, cue_start 1, cue_stop, trial_start 1, cue_start 2, cue_stop, and so on. This is how you can understand them. Wherever you see a trial_start that is the start of the trial followed by the number indicating the index of that trial for example trial_start 0 is the 1st trial, trial_start 9 is the 10th trial, and so on. Wherever, you see a cue_start it shows the start of the 'cue' followed by the number indicating the class or activity type: cue_start 1 is the motor-attempt trial, and cue_start 2 is the resting state trial. 'cue_stop' means the end of the cue or the end of the trial as in the case of training runs cue is displayed until the end of a trial. There will be 50 trials ('trial_start 0' to 'trial_start 49') which are equally divided between 25 'cue_start 1' and 25 'cue_start 2'. Now the other variable 'time_stamps' contains the time stamps (time of appearance) of these markers (such as trial_start 0, cue_start 1, cue_start 2, cue_stop, etc.). These time stamps are important because you can extract the sensor data between a particular trial using these time stamps. For example, if you want to get the sensor data of trial 10. Then find the time stamp for 'trial_start 10' and the time stamp of 'cue_stop' which follows 'trial_start 10' and then grab the sensor data between them. 

To grab the sensor data corresponding to a trial you need the other cell inside 'streams' whose type is 'Data'. Inside this cell, there are two important variables called 'time_series' and 'time_stamps'. The 'time_series' has 17 rows or channels where the first 8 channels are for EEG channels of the 'Hybrid Unicorn Black' system and the rest of the channels are for other sensors. The time stamps corresponding to each samples in this matrix are inside the 'time_stamps' variable. Please note that the number of columns of 'time_series' is the total number of samples recorded during an experimental run.

Please note: To match the time stamp of a particular event marker with the time stamp of the sensor data rely on the closest possible gap between the two if you don't get an exact match.
      
