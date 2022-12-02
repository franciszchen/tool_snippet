import os 
import shutil
import argparse

def pth2txt_folder(folder_path):
    filename_list = os.listdir(folder_path)
    for filename in filename_list:
        if os.path.isfile(os.path.join(folder_path, filename)):
            if '.pth' in filename:
                print(filename)
                file = open(os.path.join(folder_path, filename[:-4]+'.txt'), 'w')
                file.close()
                os.remove(os.path.join(folder_path, filename))

def get_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type = str, default='/home/zchen72/code/FedFFT_final/experiment/reproduce/x', help='path to save the checkpoint')
    parser.add_argument('--keep_tag', type = str, default='kept')
    args = parser.parse_args()
    return args 

    
if __name__ == '__main__':
    
    args = get_args()
    # pth2txt_folder(folder_path=args.dir_path)
    
    dir_path = args.dir_path
    filename_list = os.listdir(dir_path)
    for filename in filename_list:
        if args.keep_tag in filename:
            continue
        if os.path.isdir(os.path.join(dir_path, filename)):
            pth2txt_folder(os.path.join(dir_path, filename))
        
        
