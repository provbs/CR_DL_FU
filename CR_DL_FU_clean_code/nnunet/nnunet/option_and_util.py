# option_and_util.py
# made by youngjae kim, mi2rl
# 2024-02-05
# any questions or comments : provbs36@gmail.com

from .experiment_planning import change_batch_size as batch_changer

'''
UTILS
'''

def change_batch_size(batch_size, plan_file_path) :
    """
    param : 
        batch_size :
            input to change batch size of the task.
        plan_file_path : 
            the each task's "nnUNetPlansv2.1_plans_2D.pkl" locates under the "nnunet_preprocessed" folder.
    """
    batch_changer.change_batch_size(batch_size, plan_file_path)
    
    return batch_size
    
def change_patch_size(patch_size, plan_file_path) :
    """
    param : 
        patch_size :
            input to change patch size of the task.
        plan_file_path : 
            the each task's "nnUNetPlansv2.1_plans_2D.pkl" locates under the "nnunet_preprocessed" folder.
    """
    batch_changer.change_patch_size(patch_size, plan_file_path)


'''
TRAINING OPTIONS
'''

# model options - 
#   basic : basic nnUNet v1
#   basicMTL_1c : basic nnUNet v1 + MTL ( 1 classifier : diseases + abnormality in one vector )
#   basicMTL_2c : basic nnUNet v1 + MTL ( 2 classifier : diseases / abnormality )
model_options = "basicMTL_2c"

# weight control for MTL
weight_seg = 1.0
weight_lesion_classifier = 0.5
weight_abnormality_classifier = 0.5

# epoch
max_num_epochs = 1000

# learning rate
initial_lr = 1e-2
device_as_class = False # we have included some of X-ray images of patients with medical devices on. To count those labeled devices as a class on MTL, change it to True

# ds
deep_supervision_scales = None
ds_loss_weights = None
pin_memory = True

# augmentation level
# more / insane
augmentation = "more"

# batch size - please put pkl file path under the nnunet_preprocessed folder.
batch_size = 8
plan_file_path = "Task501_240206_balanced_MTL_2c/nnUNetPlansv2.1_plans_2D.pkl"