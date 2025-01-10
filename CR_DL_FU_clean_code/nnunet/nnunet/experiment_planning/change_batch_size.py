from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np


def change_batch_size(batch_size, plan_file_path):
    '''
    param : 
      batch_size :
          input to change batch size of the task.
      plan_file_path : 
          "nnUNetPlansv2.1_plans_2D.pkl" under "nnunet_preprocessed" folder. 
    '''
    input_file = plan_file_path
    
    a = load_pickle(input_file)

    a['plans_per_stage'][0]['batch_size'] = batch_size
    save_pickle(a, input_file)
    
    print("batch size is now ", a['plans_per_stage'][0]['batch_size'])
    
    
def change_patch_size(patch_size, plan_file_path):
    '''
    param : 
      patch_size :
          input to change patch size of the task.
      plan_file_path : 
          "nnUNetPlansv2.1_plans_2D.pkl" under "nnunet_preprocessed" folder. 
    '''
    input_file = plan_file_path
    
    a = load_pickle(input_file)

    a['plans_per_stage'][0]['patch_size'] = patch_size
    save_pickle(a, input_file)
    
    print("patch size is now ", a['plans_per_stage'][0]['patch_size'])