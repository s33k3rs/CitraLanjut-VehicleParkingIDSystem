import csv
import glob
import os
import numpy as np
import fnmatch

Base_Path = '/home/ihsan/Documents/MIK/PCL_Project/'
Dataset_Path = Base_Path + 'Data/'
Training_Set_Path = Dataset_Path + 'Train'
Test_Set_Path = Dataset_Path + 'Test'

Processed_Training_Set_Path = Dataset_Path + 'Processed/Train'
Processed_Test_Set_Path = Dataset_Path + 'Processed/Test'

FoldersToLoad=[0,0] #TRAIN THEN TEST.

if __name__=="__main__":


    # JUST DIAGNOSTICS nb_testing_samples = len(fnmatch.filter(os.listdir(Test_Set_Path), '*.jpg'))
    # print (nb_testing_samples)
    #
    # for i in PP_Method_List:
    #     ResultMatrixPath = Base_Path + 'Results/'
    #     FoldersToLoad[0] = str(FolderPair[0] + i)
    #     FoldersToLoad[1] = str(FolderPair[1] + i)
    #     print('FoldersToLoad[0]: {} and FoldersToLoad[1]: {}'.format(FoldersToLoad[0], FoldersToLoad[1]))
    #     ResultMatrixPath += ('Result'+ i + '.mat')
    #     print('ResultMatrixPath: {}'.format(ResultMatrixPath))

    #
    #build list of folder names (folder hierarchy aka labels)
    FolderNames = []
    FolderNames = os.listdir(Dataset_Path + 'Train/') #doesn't matter since both Test and Train have the same hierarchy..
    pp_method_list= ['_PPd_RGB', '_PPd_G', '_PPd_G_Canny', '_PPd_Sobel_Each','_PPd_Sobel_HSV']
    dir_type_list = ['Train','Test']

    #Training_Set_Path
    for dirtype in dir_type_list:
        for ppmethod in pp_method_list:
            with open(Dataset_Path + dirtype + ppmethod + '/' + dirtype + '_Set_Labels.csv', 'wb') as csvfile:
                print (Dataset_Path + dirtype + ppmethod + '/' + dirtype + '_Set_Labels.csv')
                for i in FolderNames:
                    print i #could also use for root, dirs, files in os.walk(path):
                    for filename in glob.glob(Dataset_Path + dirtype + ppmethod + '/'+ FolderNames[FolderNames.index(i)] + '/*'):
                        print ("Loaded: ", filename)
                        filename_inslices=filename.split('/')
                        del filename_inslices[0] #first slot is a useless blank character
                        filename_inslices_length=len(filename_inslices) #this should be 9.
                        #print('Length: {}'.format(filename_inslices_length))
                        print ('File name in slices: '.format(filename_inslices))
                        print("Filename only (no path): {}".format(filename_inslices[-1])) #last item in list
                        print("Label should be: {}".format(filename_inslices[-2])) #penultimate item in list
                        spamwriter = csv.writer(csvfile, delimiter=',',
                                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                        spamwriter.writerow([filename] + [filename_inslices[-2]])

