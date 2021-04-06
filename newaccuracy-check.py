#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:21:45 2021

@author: akhilachowdarykolla
"""

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
import os

## accuracy function
def Accuracy(predicated_y, target_y):
    if (np.logical_and(target_y[0] - predicated_y <= 0,
                         predicated_y - target_y[1] <= 0)):
        return 1
    else:
        return 0


# function to split data
def SplitFolder(labels, folders, fold_id):
    bool_suq = folders == fold_id
    train_label = labels[~bool_suq][:, 1:]
    test_label = labels[bool_suq][:, 1:]

    return train_label, test_label

# get command line argument length.

#dir_path = "neuroblastoma-data/data/systematic/cv/sequenceID"
dir_path = sys.argv[1]
print(dir_path)
## load the realating csv file
dir_path_split = dir_path.split("/cv/")
print(dir_path_split[1])
labels_path = dir_path_split[0] + "/outputs.csv.xz"
folds_path = dir_path + "/folds.csv"
input_path = dir_path + "/testFolds"
output_path = dir_path
main_path_split = dir_path_split[0].split("/data/")

labels = pd.read_csv(labels_path)
folds = pd.read_csv(folds_path)

labels = labels.values
folds = np.array(folds)
_, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
folds_sorted = folds[cor_index] # use for first split

spp_type_list =["convNet_2_cnv_fc","convNet_2_act_2_cnv_fc","convNet_3cnv_2fc","convNet_3conv_act_2fc",
 "convNet_4_cnv_fc","convNet_4con_act_2fc","convNet_5con_2fc()","convNet_5con_act_2fc", 
                 "convNet_2conv_1fc","convNet_2_3fc","convNet_even_hidden","convNet_1000_hidden"]


model_name_list = []
# get name of all the model
for py in glob.glob( input_path + "/1/randomTrainOrderings/1/models/*"):
    #get model name
    file_name = os.path.basename(py)
    name, _ = os.path.splitext(file_name)
   # print(name)
    if name == "spp_test":
      spp_types = glob.glob( input_path + "/1/randomTrainOrderings/1/models/spp_test/*")
      for i in spp_types:
          #print(os.path.basename(i))
         # print(spp_type_list[int(os.path.basename(i))])
          model_name_list.append("spp_"+spp_type_list[int(os.path.basename(i))])
          #print("-----")
    else:
        model_name_list.append(name)

num_model = len(model_name_list)
print("Model list")
print(model_name_list)



# model_list = []
# for name in model_name_list:
#     file_list = []
#     if(name.startswith("spp_")) :  
#         print(name)
#         print(spp_type_list.index(name.replace("spp_","")))
#         path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/spp_test/*/" 
#                               + "/predictions.csv")
#         print(path)
#         print("\n")
#     else:
#         path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/" 
#                               + name + "/predictions.csv")
#         print(path)
#         print("\n")
            
model_list = []
for name in model_name_list:
    file_list = []
    if(name.startswith("spp_")) : 
          path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/spp_test/" +
            str(spp_type_list.index(name.replace("spp_","")))  + "/predictions.csv")
    else:
          path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/" 
                          + name + "/predictions.csv")
    for file_path in path:
        #print(file_path)
        #get last column of each file
        df = pd.read_csv(file_path).iloc[:, -1].values
        #print(df)
        file_list.append(df)
    num_test = len(file_list)
    #print(num_test)
    #print(file_list)   
    accuracy_list = []
    # calculate accuracy
    for fold_num in range(num_test):
        _, fold_lab = SplitFolder(labels, folds_sorted[:, 1], fold_num + 1)
        acc = 0
        for (data, label) in zip(file_list[fold_num], fold_lab):
           # print(data)
            label = label.reshape(2)
            acc += Accuracy(data, label)
        
        num = fold_lab.shape[0]
        accuracy_list.append(acc/num * 100)
    
    average = sum(accuracy_list)/len(accuracy_list)
    model_component = [accuracy_list, name, average]
    model_list.append(model_component)
    #print(model_list)
model_list.sort(key = lambda model_list: model_list[2]) 
model_list = np.array(model_list,dtype=object)
model_accuracy = model_list[:, 0]
model_name = model_list[:, 1]    


for index in range(num_model):
    print(index)
    plt.scatter(model_accuracy[index], num_test * [model_name[index]], color = "black")
plt.xlabel("accuracy.percent %")
plt.ylabel("algorithm")
plt.tight_layout()
main_name = main_path_split[1]
sub_name = dir_path_split[1]
print('plot_folder/' + main_name + '_' + sub_name + '.png')
plt.savefig('plot_folder/' + main_name + '_' + sub_name + '.png')
#plt.savefig("SS_linear_accuracy.png")
plt.title(main_name + '_' + sub_name)

print(num_model)
