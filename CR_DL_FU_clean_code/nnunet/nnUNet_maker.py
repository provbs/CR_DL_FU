## nnUnet Task064 sample was used for modifying
## youngjae kim, mi2rl
## lastly modified 24.02.06

import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split


'''
Utility Functions
'''
# 1. 특정 규칙을 가지는 파일 이름을 리스트로 모아주기   
def find_filenames_by_rule_inAfolder(path, rule):
    folder_path = Path(path)
    filename_list = []
    filename_list = [f.stem for f in folder_path.glob(f"{rule}")]

    return filename_list
    
# 2. 리스트 안의 파일 이름들을 원하는 형식으로 가공해주기
def change_filenames_by_rule_inthelist(filename_list, replace_a, replace_b) :
    filename_list = [(file_name.replace(replace_a, replace_b)) for file_name in tqdm(filename_list)]
    return filename_list
    
   
# 3. 리스트 안의 파일들을 다른 폴더로 복사해주기
# extension: 파일 확장자
def copy_files_inthelist_AtoB(filename_list, path_A, path_B, extension) :
    moved_file_num = 0
    print('From ' + path_A + ' to ' + path_B + ' ...')
    num = 0

    try:
        for filename in tqdm(filename_list):
            num+=1
            # if num % 5 == 0 :
            #     print(num)
            shutil.copy(path_A + '/' + filename + extension, 
                        path_B + '/' + filename + extension)
            moved_file_num += 1
    except Exception as e:
        print(e)
    
    print("Succefully copied " , moved_file_num, " file(s).")

    return
    

if __name__ == "__main__":

    # dataset setting
    dataset_base = '/home/yjkim63/workspace/nas203_forGPU_youngjaekim/1_FU_clean-code/nnUNet-nnunetv1/nnunet_raw_data_base/231106_all_balanced'

    print("database : " , dataset_base)

    images_location = dataset_base + "/imagesTr"
    print("images location : " , images_location)
    images_name_list = find_filenames_by_rule_inAfolder(images_location, "*.nii.gz")
    labels_location = dataset_base + "/labelsTr"
    print("labels location : " , labels_location)
    labels_name_list = find_filenames_by_rule_inAfolder(labels_location, "*.nii.gz")

    print("images found .. :" , len(images_name_list))
    print("labels found .. :" ,len(labels_name_list))

    images_name_list = change_filenames_by_rule_inthelist(images_name_list, ".nii", "")
    labels_name_list = change_filenames_by_rule_inthelist(labels_name_list, ".nii", "")

    # file management
    task_id = 501
    task_name = "240206_balanced_MTL_2c"

    foldername = "Task%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw_data, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # print(images_name_list)
    # copy_files_inthelist_AtoB(images_name_list, images_location, imagestr, ".nii.gz")
    # copy_files_inthelist_AtoB(labels_name_list, labels_location, labelstr, ".nii.gz")
    print("TO PROCEED FASTER, PLEASE MOVE THE FILE ON YOUR OWN!")

    # make json file
    json_dict = {}
    json_dict['name'] = ""
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = ""
    json_dict['reference'] = ""
    json_dict['licence'] = ""
    json_dict['release'] = ""
    json_dict['modality'] = {
        "0000": "Xray"
    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Consolidation",
        "2": "Nodule",
        "3": "Pleural_Effusion",
        "4": "Device"
    } 

    json_dict['numTraining'] = len(labels_name_list)
    json_dict['numTest'] = ""
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i.split("/")[-1], "label": "./labelsTr/%s.nii.gz" % i.split("/")[-1]} for i in
                              labels_name_list]
    json_dict['test'] =[]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))
