#task-switching paradigm experiment with EEG

debugging = False
running = True

#%% import all the modules and other script
import sys
import os
import random
import numpy as np
import pandas as pd
import json
from datetime import datetime
from numpy import random as rdm
import serial  # Import for USB trigger interface

if running == True: # if running the experiment
    from psychopy import visual, core, event, gui, logging
    cwd = os.path.abspath(os.path.dirname("__file__")) # the path of the current working directory
else:
    cwd = 'D:\\Ghent_Braem\\Experiment\\1st_fMRI\\Experiment_script'

# print(cwd)
sys.path.append(cwd)
from imageList import stimulusList, taskList, subj_random
import genCon

#%% functions to be defined first
# Function to send triggers to EEG via USB trigger interface
try:
    port = serial.Serial('COM4', baudrate=115200)  # Set the correct COM port for your system
except serial.SerialException:
    port = None
    print("Serial port not found. Triggers will not be sent.") #Handling Failure to Connect

def send_trigger(code):
    if port:
        port.write(code.to_bytes(1, 'big'))
        core.wait(0.01)  # Keep trigger high for 10 ms
        port.write((0).to_bytes(1, 'big'))

# Function to draw custom fixation point
from genCon import draw_custom_fixation

#%% collect demographic info
# Dialogue box
info = {'Subject No.':  'NaN', # the subject number
        'Run No. Start':'1',
        'Run No. End':  '8'} 

if running == True:
    dlg = gui.DlgFromDict(dictionary = info, title = 'Experiment Setup', order = ['Subject No.', 'Run No. Start', 'Run No. End'])
    if not dlg.OK: core.quit()
    if info['Subject No.'] == '': core.quit()
    subjNo = int(info['Subject No.'])
    runNo_start = int(info['Run No. Start'])
    runNo_end = int(info['Run No. End'])
else:
    subjNo = random.randrange(1, 10)
    runNo_start = 3
    runNo_end = 4

subjID = "subj_" + str(subjNo)
n_runs = runNo_end - runNo_start + 1 # number of runs

#%% defining all the important global variables
globalVars = {}
globalVars["taskDims"] = ["stim", "rule"]
globalVars["stimFullList"] = ["animal", "place", "vehicle"]
globalVars["ruleFullList"] = ["age", "size", "location"]
globalVars["nTrialsBlock"] = 54
globalVars["nBlocks"] = 8
globalVars["intervalTaskCue"] = 1000 # 1000 ms
globalVars["intervalTaskCueTran"] = 750
globalVars["intervalStim"] = 2000
globalVars["ddlResponse"] = 2000
globalVars["intervalTrialFb"] = 750
globalVars["threBlockFb"] = 0.8      # the accuracy threshold, below which the block-wise feedback will show
globalVars["CTIs"] = np.array([1250, 
             	   			2125, 2250, 2375, 2500, 2625, 2750, 
             	   			2808, 2865, 2923, 2981, 3038, 3096, 3154, 3212, 3269, 3327, 3385, 3442, 3500, 
             	   			3625, 3750, 3875, 4000, 4125, 4250, 
             	   			5000, 
             	   			5001, 
             	   			5875, 6000, 6125, 6250, 6375, 6500, 
             	   			6558, 6615, 6673, 6731, 6788, 6846, 6904, 6962, 7019, 7077, 7135, 7192, 7250, 
             	   			7375, 7500, 7625, 7750, 7875, 8000, 
             	   			8750]) # 54 CTIs, mean is 5041 ms.
globalVars["n_CTIs_long"] = int(len(globalVars["CTIs"]) * 0.5) # fMRI pilot setting, half of them are long, 27 trials
globalVars["numTran"] = 13 # or 14
globalVars["ITIs"] = np.array([1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 1750, 
                               2150, 2150, 2150, 2150, 2150, 2150, 2150, 2150, 2150, 2150, 2150, 
                               2550, 2550, 2550, 2550, 2550, 2550, 2550, 
                               2950, 2950, 2950, 2950, 2950, 2950, 
                               3350, 3350, 3350, 3350, 3350, 3350, 
                               3750, 3750, 3750, 3750, 3750, 
                               4150, 4150, 4150, 
                               4550, 4550, 
                               4950, 
                               5350]) # 54 intervals

globalVars["leftKey"] = "f"
globalVars["rightKey"] = "j"
keyList = ['f','j','escape']

#%% defining all the between subject variables
subj_param = genCon.fromInttoDict(subjNo, subj_random, "subjNo")

# fixation colors of 2 block types
#%% to be filled %%

# response mappings
responseMappings = tuple(map(int, subj_param["respMaps"].split(', ')))
respMap_age = int(responseMappings[0])
respMap_size = int(responseMappings[1])
respMap_location = int(responseMappings[2])

if responseMappings == (0,0,0):
    respMap_inst = "resp_map_0_0_0.jpg"
elif responseMappings == (0,0,1):
    respMap_inst = "resp_map_0_0_1.jpg"
elif responseMappings == (0,1,0):
    respMap_inst = "resp_map_0_1_0.jpg"
elif responseMappings == (1,0,0):
    respMap_inst = "resp_map_1_0_0.jpg"
elif responseMappings == (0,1,1):
    respMap_inst = "resp_map_0_1_1.jpg"
elif responseMappings == (1,1,0):
    respMap_inst = "resp_map_1_1_0.jpg"
elif responseMappings == (1,0,1):
    respMap_inst = "resp_map_1_0_1.jpg"
else:
    respMap_inst = "resp_map_1_1_1.jpg"
    
#%% create all the experiment psychopy objects
win = visual.Window(color=(1,1,1), fullscr = True) 
win.setMouseVisible(False)

instr_obj = visual.TextStim(win, text = '', wrapWidth = 1, font = 'monospace', color=(-1,-1,-1))
instr_img = visual.ImageStim(win, image = None, pos = (0, 0), size = 2, units = "norm")
instr_respMap = visual.ImageStim(win, image = respMap_inst, pos = (0, -0.4), size = (0.55, 0.45), units = "norm")

taskCue_frame = visual.Rect(win, width=0.36, height=0.32, lineColor=(-1,-1,-1))
taskCue_obj = visual.TextStim(win, text='', height = 0.072, pos = (0,0),  bold=True, color=(-1,-1,-1))

stimUp_obj = visual.ImageStim(win, image = None, size = (0.36, 0.24), pos = (0, 0.25), units = "height")
stimMid_obj = visual.ImageStim(win, image = None, size = (0.36, 0.24), pos = (0, 0), units = "height")
stimLow_obj = visual.ImageStim(win, image = None, size = (0.36, 0.24), pos = (0, -0.25), units = "height")

fb_obj = visual.TextStim(win, text='', height = 0.07, pos = (0,0), bold=True, color=(-1,-1,-1))

ITI_obj = visual.TextStim(win, text='+', height=0.07, pos = (0, 0),  bold=True, color=(-1,-1,-1))
block_end = visual.TextStim(win, text='+', height=0.06, pos = (0, 0),  bold=False, color=(-1,-1,-1))

# Close if 'Escape' is pressed at any moment:
escKeyMessage = visual.TextStim(win,
                                text='ESC was pressed.\nClosing the experiment.',
                                height=0.07,
                                color=(-1, -1, -1))
def EscExit():
    print('Experiment terminated by the user (ESC)')
    # present a message for 2 seconds (120 frames)
    for frameN in range(120):
        escKeyMessage.draw()
        win.flip()
    win.close()
    core.quit()
event.globalKeys.add(key='escape', func=EscExit)

#%% defining the clocks and st up triggers
clock_global = core.Clock()  # the time since the start of the script
clock_cti = core.Clock()
clock_stim = core.Clock()

#%% loop over each block and trial, present stimulus, record and save response for each trial
# read the json file and reconstruct the blocktrialiterator
with open(cwd +"\\iterators\\exp_iterator_subj" + str(subjNo) + ".json", "r") as read_it:
      blockTrialIterator = json.load(read_it)

# defining all the blocks to be run
if debugging == True:
    blockIDs = blockIDs[0:3] # only run 3 blocks for debugging purpose
else:
    blockIDs = blockIDs[runNo_start - 1:runNo_end] # including all the runs defined at the top of script

# loop over each block and each trial
for blockIdx, blockID in enumerate(blockIDs, runNo_start - 1):
    runID = "run_" + str(blockIdx + 1)
    trialIterator = blockTrialIterator[blockIdx]
    
    if debugging == True:
        trialIterator = trialIterator[0:5] # only run 5 trials per block for debugging purpose
        
    random.shuffle(globalVars["ITIs"]) # shuffle the ITI

#############################################
    ####### create data frame using panda #######
    #############################################
    # define columns #
    columns = ['subject', 'date',
               'reg_color', 'start_block', 'resp_map_age', 'resp_map_size', 'resp_map_location',
               'run_id', 'block_type', 'block_id',
               'trial_num', 'trial_id', 'trial_nature', 
               'task_id','stim', 'rule', 'image_up_id', 'image_mid_id', 'image_low_id', 'image_target_id', 'CTI',
               'dim_tran', 'task_tran', 'image_target_tran',
               'answer', 'correct_key', 
               'response', 'rt', 'accuracy', 
               ######### the following information all are the timing information for the fMRI GLM modelling ########
               ######################## all the timing information are the post-trigger time ########################
               'ITI_onset','ITI_duration','ITI_offset',
               'cue_onset','cue_duration','cue_offset',
               'CTI_onset','CTI_duration','CTI_offset',
               'tranCue_onset','tranCue_duration','tranCue_offset',
               'stim_onset','stim_duration','stim_offset',
               'resp_onset',
               'fb_onset','fb_duration','fb_offset',
               'run_time', 'time_elapsed']
    data = pd.DataFrame(columns = columns)
    dateTimeNow = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    
    # data file directory
    datafile_path = cwd + '\\data\\'
    print(datafile_path)
    
    # datafile name
    datafile_name = "{}behavioral_data_{}_{}.csv".format(datafile_path, str(subjID), str(runID))


    ######################################################################
    ############# block instruction and researcher start the #############
    ######################################################################
    # Instruction at the start of the block
    if blockID[0:2] == "RG":
        send_trigger(10)  # Marker for the beginning of a regular block
        instr_img.setImage("regular_block_inst.jpg")
    else:
        send_trigger(20)  # Marker for the beginning of a transform block
        instr_img.setImage("transform_block_inst.jpg")
       
    instr_img.draw(); instr_respMap.draw(); win.flip()
    start_block = event.waitKeys(keyList=["space", "escape"])

    # Start of the block
    clock_global.reset()  # reset the global clock at the beginning of the block
    
    # Loop over each trial
    for trialIdx, trialDict in enumerate(trialIterator):
        trialNum = trialIdx + 1 # since the index starts from zero
        trialID = str(subjID) + "_" + str(blockID) + "_" + str(trialNum)
        
        # intertrial interval (ITI)
        send_trigger(30)  # Marker for the beginning of the ITI
        ITI_obj.draw(); win.flip(); 
        core.wait(globalVars["ITIs"][trialIdx]/1000)
        
        # update and show task cue
        send_trigger(40)  # Marker for the beginning of the task cue
        taskCue = "{}\n{}".format(trialDict["task"]["stim"], trialDict["rule_cue"])
        taskCue_obj.setText(taskCue); taskCue_frame.draw(); taskCue_obj.draw(); win.flip()
        core.wait(globalVars["intervalTaskCue"]/1000)
        
        # CTI presentation
        send_trigger(50)  # Marker for the beginning of the CTI
        clock_cti.reset()
        draw_custom_fixation(win, duration=trialDict["CTI"] / 1000)  # Draw the custom fixation point with dynamic duration during CTI
        
        # show transformed task cue if it's a transform trial
        if trialDict["trial_type"] == "TF":
            send_trigger(60)  # Marker for the beginning of transform cue presentation
            tranCue = "{}\n{}".format(trialDict["task_tran"]["stim"], trialDict["rule_cue_tran"])
            taskCue_obj.setText(tranCue); taskCue_frame.draw(); taskCue_obj.draw(); win.flip() 
            core.wait(globalVars["intervalTaskCueTran"]/1000)
        
        # Show stimulus
        send_trigger(70)  # Marker for the beginning of the stimulus presentation
        stimUp_obj.setImage(trialDict["img_up"]["full_name"])
        stimMid_obj.setImage(trialDict["img_mid"]["full_name"])
        stimLow_obj.setImage(trialDict["img_low"]["full_name"])
        stimUp_obj.draw(); stimMid_obj.draw(); stimLow_obj.draw(); win.flip()
        
        # time and record participants' response
        clock_stim.reset()
        response = event.waitKeys(maxWait=globalVars["ddlResponse"]/1000, keyList=keyList)
        send_trigger(80)  # Marker for the beginning of the response
        rt = clock_stim.getTime()
        
        # retrieve response, evaluate accuracy, and show feedback
        if response:
            respKey = response[0]
            if respKey in keyList[0:2]:  # valid response
                if respKey == trialDict['correct_key']:
                    acc = True
                    fb_obj.setText(' ')
                else:
                    acc = False
                    fb_obj.setText('Incorrect!')
            else:
                acc = False
                fb_obj.setText('too slow!')
        else:  # no response
            respKey = "null"
            rt = "null"
            acc = False
            fb_obj.setText('too slow!')
        
        fb_obj.draw(); win.flip(); 
        send_trigger(90)  # Marker for the beginning of the feedback
        core.wait(globalVars["intervalTrialFb"]/1000)
        
        # save trial data
        data = data.append({
            'subject': subjID, 
            'date': dateTimeNow,
            'run_id': runID,
            'block_type': blockID[0:2], 
            'trial_num': trialNum,
            'task_id' : trialDict["task"]["task_id"],
            'stim': trialDict["task"]["stim"], 
            'rule': trialDict["task"]["rule"],
            'response': respKey, 
            'rt': rt, 
            'accuracy': acc}, ignore_index=True)

    ######################################
    ########## save data !!!!!!! #########
    ######################################    
    data.to_csv(datafile_name)    
    
    ################################################################################################
    ### show block-wise feedback based on the accuracy rate of both regular and transform trials ###
    ################################################################################################
    if blockIdx < 7:
        blockFb = ""
        ACC_reg = data.loc[(data["block_type"] == "RG") & (data["accuracy"] == True)]["accuracy"].mean()
        ACC_tran = data.loc[(data["block_type"] == "TF") & (data["accuracy"] == True)]["accuracy"].mean()    
        if ACC_reg < globalVars["threBlockFb"]:
            ACC_reg_percent = int(ACC_reg*100)
            blockFb += "In the previous block, you have achieved {} percent of accuracy on the regular trials, which is, unfortunately, lower than other participants\'performance, please try harder to perform better.\n\n".format(ACC_reg_percent)
        if (np.isnan(ACC_tran) == False) & (ACC_tran < globalVars["threBlockFb"]):
            ACC_tran_percent = int(ACC_tran*100)
            blockFb += "In the previous block, you have achieved {} percent of accuracy on the transform trials, which is, unfortunately, lower than other participants\'performance, please try harder to perform better.\n\n".format(ACC_tran_percent)       
    
        instr_obj.setText(blockFb); instr_obj.draw(); win.flip()
    else:
        instr_goodbye = "Now you've finished the experiment ! \n\n Thanks a lot for your participation, now please wait for the researcher to come to you."
        instr_obj.setText(instr_goodbye); instr_obj.draw(); win.flip()
               
    core.wait(16)

#%% close the whole experiment 
win.close()
