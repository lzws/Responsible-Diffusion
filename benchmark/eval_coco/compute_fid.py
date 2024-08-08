from cleanfid import fid
from config import coco30k_path, coco3k_path


def compute_fid(coco_path, data_path):
    '''
    '''
    fid_value = fid.compute_fid(coco_path, data_path)
    return fid_value

if __name__ == '__main__':

    data_path = ''
    fidc = compute_fid(coco_path=coco30k_path, data_path=data_path)
    print("fid分数：",fidc)