#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 07:26:06 2021

@author: akhilachowdarykolla
"""

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
from function import *
from sklearn import preprocessing
from spp_model import *


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


## load the realating csv file
# get command line argument length.
split_list = []
accurays = []
for i in range(0,9):
    dir_path = sys.argv[1]#"/Users/akhilachowdarykolla/Desktop/neuroblastoma-data/data/detailed/cv/sequenceID/testFolds/1"
    model_id = str(i)   
        ## load the realating csv file
    dir_path_split = dir_path.split("cv")
    fold_path_split = dir_path.split("/testFolds/")
    profiles_path = dir_path_split[0] + "profiles.csv.xz"
    labels_path = dir_path_split[0] + "outputs.csv.xz"
    folds_path = fold_path_split[0] + "/folds.csv"
    fold_num = int(fold_path_split[1])
    outputs_path = dir_path + "/randomTrainOrderings/1/models_test/Cnn_spp_test/" + model_id
    input_path = dir_path.split("testFolds")[0] + "testFolds"
    
    #init the model parameter
    criterion = SquareHingeLoss()
    step = 5e-5
    epoch = 1
    eps = epoch
    model_id_int = int(model_id)
    model = model_list[model_id_int]
    optimizer = optim.Adam(model.parameters(),  lr= step)
    min_feature = 500
    
    #save the init model
    model_path = "model_path/" + dir_path + "/" + model_id
    if not os.path.exists(model_path):
        os.makedirs(model_path) 
    PATH = model_path + 'cifar_net.pth'
    torch.save(model.state_dict(), PATH)
    
    ## load the file
    dtypes = { "sequenceID": "category"}
    profiles = pd.read_csv(profiles_path, dtype=dtypes)
    labels = pd.read_csv(labels_path)
    
    ## extract all sequence id
    sequenceID = labels["sequenceID"]
    seq_data_list = []
    # loop through all 
    for id in sequenceID:
        #extract all data from profiels using same id
        one_object = profiles.loc[profiles["sequenceID"] == id]
        one_feature = np.array(one_object["signal"].tolist())
        one_feature = preprocessing.scale(one_feature)
        #padding data less than 500
        N = one_feature.shape[0]
        if N < 500:
            padding_num = min_feature-N
            one_feature = np.pad(one_feature, (0, padding_num), 'constant')
            N = 500
        #transfter the data type
        one_feature = torch.from_numpy(one_feature.astype(float)).view(1, 1, N)
        one_feature = one_feature.type(torch.FloatTensor)
        one_feature = Variable(one_feature).to(device)
        #add to list
        seq_data_list.append(one_feature)
    inputs = seq_data_list
    
    ## get folder
    labels = labels.values
    folds = pd.read_csv(folds_path)
    folds = np.array(folds)
    _, cor_index = np.where(labels[:, 0, None] == folds[:, 0])
    folds_sorted = folds[cor_index] # use for first split
    
    ## transfer label type
    labels = torch.from_numpy(labels[:, 1:].astype(float))
    labels = labels.to(device).float()
    
    ## split train and test data
    bool_flag = folds_sorted[:, 1] == fold_num
    train_data = [a for i,a in enumerate(inputs) if not bool_flag[i]]
    test_data = [a for i,a in enumerate(inputs) if bool_flag[i]]
    train_label = labels[~bool_flag]
    test_label = labels[bool_flag]
    num_test = len(test_data)
    
    ## do early stop learning, get best epoch
    #split validation and subtraining data
    num_sed_fold = len(train_data)
    sed_fold = np.repeat([1,2,3,4,5], num_sed_fold/5)
    left = np.arange(num_sed_fold % 5) + 1
    sed_fold = np.concatenate((sed_fold, left), axis=0)
    np.random.shuffle(sed_fold)
    bool_flag = sed_fold == 1
    subtrain_data = [a for i,a in enumerate(train_data) if not bool_flag[i]]
    valid_data = [a for i,a in enumerate(train_data) if bool_flag[i]]
    subtrain_label = train_label[~bool_flag]
    valid_label = train_label[bool_flag]
    
    # do stochastic gradient descent
    best_output_list = []
    valid_losses = []
    subtrain_losses = []
    avg_subtrain_loss =[]
    avg_valid_loss = []
    ## train the network
    for epoch in range(epoch):  # loop over the dataset multiple times
        for index, (data, label) in enumerate(zip(subtrain_data, subtrain_label)):
            model.train()  
            # zero the parameter gradients
            optimizer.zero_grad()
            # do SGD
            #print(data)
            #print(data.shape)
            outputs = model(data)        
            loss = criterion(outputs, label)
            #print("Subtrain Loss : ",loss)
            loss.backward()
            optimizer.step()
            subtrain_losses.append(loss.cpu().data.numpy())
        
        subtrain_loss = np.average(subtrain_losses)
        avg_subtrain_loss.append(subtrain_loss)
        
        with torch.no_grad():
            for index, (data, label) in enumerate(zip(valid_data, valid_label)):
                model.eval()
                outputs = model(data)                    
                loss = criterion(outputs, label)
                #print("VAlidation Loss : ",loss)
                valid_losses.append(loss.cpu().data.numpy())
        
        valid_loss = np.average(valid_losses)
        avg_valid_loss.append(valid_loss)
    
    
    
    min_loss_subtrain = min(avg_subtrain_loss)
    #print("Min loss",min_loss_valid)
    best_parameter_subtrain= avg_subtrain_loss.index(min_loss_subtrain)
    
    print("best parameter subtrains",best_parameter_subtrain)
    
    #get best parameter
    min_loss_valid = min(avg_valid_loss)
    #print("Min loss",min_loss_valid)
    best_parameter = avg_valid_loss.index(min_loss_valid)
    
    print("best parameter",best_parameter)
    
    # init variables for model
    model = model_list[model_id_int]
    model.load_state_dict(torch.load(PATH))
    optimizer = optim.Adam(model.parameters(),  lr= step)
    
    ## train the network using best epoch
    for epoch in range(best_parameter + 1): 
        for index, (data, label) in enumerate(zip(train_data, train_label)):
            model.train()  
            # zero the parameter gradients
            optimizer.zero_grad()
            # do SGD
            outputs = model(data)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
    test_losses = []
    test_outputs = []
    with torch.no_grad():
        for data in test_data:
            output = model(data).cpu().data.numpy().reshape(-1)
            test_outputs.append(output)
        for index, (data, label) in enumerate(zip(test_data, test_label)):
                model.eval()
                outputs = model(data)                    
                loss = criterion(outputs, label)
                print("numpy : ",loss.cpu().data.numpy())
                test_losses.append(loss.cpu().data.numpy())
    
        print(np.average(test_losses))
        print("min vale:",min(test_losses))    
        test_outputs = np.array(test_outputs)
        
        
         # choose the min value from valid list
    min_loss_train = min(subtrain_losses)
    min_train_index = subtrain_losses.index(min(subtrain_losses))
    min_loss_valid = min(valid_losses)
    best_parameter_value = valid_losses.index(min(valid_losses))
    best_output = test_outputs[best_parameter_value]
    best_output_list.append(best_output)     

        # plot
    plt.plot(subtrain_losses, label = 'Training loss')
    plt.plot(valid_losses, label = 'Validation loss')
    plt.scatter(min_train_index, min_loss_train, label = 'min train value', color='green')
    plt.scatter(best_parameter_value, min_loss_valid, label = 'min valid value', color='black')
    plt.legend(frameon=False)
    plt.xlabel("step of every 10 min-bath")
    plt.ylabel("loss")
    plt.show()
    main_path_split = dir_path_split[0].split("/data/")
    main_name = main_path_split[1]
    sub_name = dir_path.split("cv/")[1].split("/testFolds")[0]
    plt.savefig('plot_folder/' + str(eps)+ "itr-loss" + '_' + "main_name" +  "_" + "sub_name" + '.png')
            
#     # test data
#     with torch.no_grad():
#         accuracy_subtrain = 0
#         accuracy_valid = 0
#         for index in range(num_test):
#             accuracy_subtrain = accuracy_subtrain + Accuracy(subtrain_losses[index], subtrain_label[index].cpu().data.numpy())
#             accuracy_valid = accuracy_valid+ Accuracy(valid_losses[index], valid_label[index].cpu().data.numpy())
    
#     #print(accuracy/num_test * 100)
#     print(accuracy_subtrain/num_test )
#     print(accuracy_valid/num_test)
    
#     # this fucntion output the csv file
#     cnn_output_subtrain = pd.DataFrame(subtrain_losses)
#     cnn_output_valid = pd.DataFrame(valid_losses)
#     if not os.path.exists(outputs_path):
#         os.makedirs(outputs_path) 
#     cnn_output_subtrain.to_csv(outputs_path + '/subtrain_losses.csv') 
#     cnn_output_valid.to_csv(outputs_path + '/valid_losses.csv') 
    
    
#     spp_type_list =["convNet_2_cnv_fc","convNet_2_act_2_cnv_fc","convNet_3cnv_2fc","convNet_3conv_act_2fc",
#       "convNet_4_cnv_fc","convNet_4con_act_2fc","convNet_5con_2fc()","convNet_5con_act_2fc", 
#                       "convNet_2conv_1fc","convNet_2_3fc","convNet_even_hidden","convNet_1000_hidden"]
#     #model_name_list=["Cnn_spp_test"]  
# #print(model_name_list)
#     model_name_list = []
#     # get name of all the model
#     print(input_path)
#     print(glob.glob(input_path + "/1/randomTrainOrderings/1/models_test/*"))
#     for py in glob.glob( input_path + "/1/randomTrainOrderings/1/models_test/*"):
#         #get model name
#         file_name = os.path.basename(py)
#         name, _ = os.path.splitext(file_name)
#         print(py)
#         print("here")
#         if name == "Cnn_spp_test":
#           spp_types = glob.glob( input_path + "/1/randomTrainOrderings/1/models_test/Cnn_spp_test/*")
#           for i in spp_types:
#               #print(os.path.basename(i))
#              # print(spp_type_list[int(os.path.basename(i))])
#               model_name_list.append("spp_"+spp_type_list[int(os.path.basename(i))])
#               #print("-----")
#         else:
#             model_name_list.append(name)
    
#     num_model = len(model_name_list)
#     print("Model list")
#     print(model_name_list)


                
#     model_list = []
#     for name in model_name_list:
#         file_list = []
#         if(name.startswith("spp_")) : 
#               path = glob.glob( input_path + "/*/randomTrainOrderings/1/models_test/Cnn_spp_test/" +
#                 str(spp_type_list.index(name.replace("spp_","")))  + "/subtrain_losses.csv")
      
#         for file_path in path:
#             #print(file_path)
#             #get last column of each file
#             df = pd.read_csv(file_path).iloc[:, -1].values
#             #print(df)
#             file_list.append(df)
#         num_test = len(file_list)
#         #print(num_test)
#         #print(file_list)   
#         accuracy_list = []
#         # calculate accuracy
#         for fold_num in range(num_test):
#             _, fold_lab = SplitFolder(labels, folds_sorted[:, 1], fold_num + 1)
#             acc = 0
#             for (data, label) in zip(file_list[fold_num], fold_lab):
#                 print(label)
#                 #label = label.reshape(2)
#                 acc += Accuracy(data, label)
            
#             num = fold_lab.shape[0]
#             accuracy_list.append(acc/num * 100)
        
#         average = sum(accuracy_list)/len(accuracy_list)
#         model_component = [accuracy_list, name, average]
#         model_list.append(model_component)
#         #print(model_list)
#     model_list.sort(key = lambda model_list: model_list[2]) 
#     model_list = np.array(model_list,dtype=object)
#     model_accuracy = model_list[:, 0]
#     model_name = model_list[:, 1]    
    
    
     
#     # model_list = []
#     # for name in model_name_list:
#     #     file_list = []
#     #     if(name.startswith("spp_")) : 
#     #           path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/spp_test/" +
#     #             str(spp_type_list.index(name.replace("spp_","")))  + "/subtrain_losses.csv")
#     #     else:
#     #           path = glob.glob( input_path + "/*/randomTrainOrderings/1/models/" 
#     #                           + name + "/subtrain_losses.csv")
#     #     for file_path in path:
#     #         #print(file_path)
#     #         #get last column of each file
#     #         df = pd.read_csv(file_path).iloc[:, -1].values
#     #         #print(df)
#     #         file_list.append(df)
#     #     num_test = len(file_list)
#     #     #print(num_test)
#     #     #print(file_list)   
#     #     accuracy_list = []
#     #     # calculate accuracy
#     #     for fold_num in range(num_test):
#     #         _, fold_lab = SplitFolder(labels, folds_sorted[:, 1], fold_num + 1)
#     #         acc = 0
#     #         for (data, label) in zip(file_list[fold_num], fold_lab):
#     #            # print(data)
#     #             label = label.reshape(2)
#     #             acc += Accuracy(data, label)
            
#     #         num = fold_lab.shape[0]
#     #         accuracy_list.append(acc/num * 100)
        
#     #     average = sum(accuracy_list)/len(accuracy_list)
#     #     model_component = [accuracy_list, name, average]
#     #     model_list.append(model_component)
#     #     #print(model_list)
#     # model_list.sort(key = lambda model_list: model_list[2]) 
#     # model_list = np.array(model_list,dtype=object)
#     # model_accuracy = model_list[:, 0]
#     # model_name = model_list[:, 1]    
    
#     print(num_model)
#     for index in range(num_model):
#         print(index)    
#         plt.scatter(model_accuracy[index], num_test * [model_name[index]], color = "black")
#     plt.xlabel("accuracy.percent %")
#     plt.ylabel("algorithm")
#     plt.tight_layout()
#     main_name = main_path_split[1]
#     sub_name = dir_path_split[1]
#     plt.savefig('/Users/akhilachowdarykolla/Desktop/plot_folder/' + "subtrain_" +main_name + '_' + sub_name + '.png')
#     #plt.savefig("SS_linear_accuracy.png")
#     plt.title(main_name + '_' + sub_name)

