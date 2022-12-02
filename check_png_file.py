import os
import argparse


def check_file(folder):
    scan_list = os.listdir(folder)
    for scan_name in scan_list:
        scan_path = os.path.join(folder, scan_name)
        for img_name in os.listdir(scan_path):
            if '._' in img_name:
                print(os.path.join(
                    scan_path,
                    img_name
                ))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, default="None")
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = get_args()
    check_file(folder=args.source_dir) 