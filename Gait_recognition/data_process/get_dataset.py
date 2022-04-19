import os
import random
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

new_data_path = '..\\data\\CASIABidentification_cropped'
totensor = transforms.ToTensor()
aid_list = [0,18,36,54,72,90,108,126,144,162,180]

def get_img_path(pid,aid,mode,sequence):
# 得到某一张图像路径
    img_path = os.path.join(new_data_path,str(pid))
    img_name = str(pid).zfill(3) + '-' + mode + '-' + str(sequence).zfill(2) + '-' + str(aid).zfill(3) + '.png'
    img_pathname = os.path.join(img_path,img_name)
    if os.path.exists(img_pathname) ==False:
        return str("nofind")
    return img_pathname

# mode:bg,cl,nm
def get_img_row(pid,aid,mode,sequence):
# 获取图像
    img_path = os.path.join(new_data_path,str(pid))
    img_name = str(pid).zfill(3) + '-' + mode + '-' + str(sequence).zfill(2) + '-' + str(aid).zfill(3) + '.png'
    img_pathname = os.path.join(img_path,img_name)
    if os.path.exists(img_pathname) ==False:
        return str("nofind")
    return Image.open(img_pathname)


def get_img_tensor(pid,aid,mode,sequence):
# 获取图像tensor
    img = get_img_row(pid,aid,mode,sequence)
    if img == "nofind":
        return img
    return totensor(img)

def get_img_inform(path):
# 获取对应图像下的标签信息
    path = os.path.basename(path)
    info = path.split(sep = '-')
    pid = int(info[0])
    mode = info[1]
    sequence = int(info[2])
    aid = int(info[3].split(sep = '.')[0])
    return pid,mode,sequence,aid

def imgpathlist_search(pathlist,pid,aid,mode,sequence):
# 查找列表里的元素
    for path in pathlist:
        img_name = str(pid).zfill(3) + '-' + mode + '-' + str(sequence).zfill(2) + '-' + str(aid).zfill(3) + '.png'
        if path.find(img_name) != -1:
            return path
    return -1

def imgpathlist_search(pathlist,pid,aid,mode,sequence):
#在图像列表中查找某一图像
    for path in pathlist:
        img_name = str(pid).zfill(3) + '-' + mode + '-' + str(sequence).zfill(2) + '-' + str(aid).zfill(3) + '.png'
        if path.find(img_name) != -1:
            return path
    return -1

#dataaet_mode:train(pid:1-50),evaluation(pid:51-74),test(pid:75-124)
def get_probe_list(dataset_mode,aid_range):
#获取某一角度下probe图像的列表
    mode = 'nm'
    probe_list = []
    if dataset_mode == "train":
        pid_range = (1,51)
    elif dataset_mode == "evaluation":
        pid_range = (51,75)
    elif dataset_mode == "test":
        pid_range = (75,125)
    elif dataset_mode == "74p":
        pid_range = (1,75)
    for pid in range(pid_range[0],pid_range[1]):
        for sequence in range(5,7):
            for aid in aid_list:
                if aid >=aid_range[0] and aid<=aid_range[1]:
                    img_path = get_img_path(pid,aid,mode,sequence)
                    if img_path != "nofind":
                        probe_list.append(img_path)
    return probe_list

#dataaet_mode:train(pid:1-50),evaluation(pid:51-74),test(pid:75-124)
def get_gallary_list(dataset_mode,aid_range,mode):
#获取某一角度范围下gallary图像的列表
    probe_list = []
    if dataset_mode == "train":
        pid_range = (1,51)
    elif dataset_mode == "evaluation":
        pid_range = (51,75)
    elif dataset_mode == "test":
        pid_range = (75,125)
    elif dataset_mode == "74p":
        pid_range = (1,75)
    for pid in range(pid_range[0],pid_range[1]):
        for sequence in range(1,5):
            for aid in aid_list:
                if aid >=aid_range[0] and aid<=aid_range[1]:
                    img_path = get_img_path(pid,aid,mode,sequence)
                    if img_path != "nofind" :
                        probe_list.append(img_path)
    return probe_list

def get_positive_sample(dataset_mode,probe_aid_range,gallary_aidrange,gmode):
# 获取正样本的列表
    positive_sample_list = []
    probe_list = get_probe_list(dataset_mode, probe_aid_range)
    gallary_list = get_gallary_list(dataset_mode,gallary_aidrange,gmode)
    for probe in probe_list:
        ppid,pmode,psequence,paid = get_img_inform(probe)
        for gaid in aid_list:
            # gmode = nm时，使用不同视角作为gallary
            if gmode == 'nm':
                if gaid >=gallary_aidrange[0] and gaid<=gallary_aidrange[1] and gaid != ppid:
                    for gsequence in range(1,5):
                        gallary = imgpathlist_search(gallary_list,ppid,gaid,gmode,gsequence)
                        if gallary != -1:
                            positive_sample_list.append((probe,gallary))
    return positive_sample_list

def get_negative_sample(dataset_mode,probe_aid,gallary_aidrange,gmode,num):
    negative_sample_list = []
    probe_list = get_probe_list(dataset_mode, probe_aid)
    gallary_list = get_gallary_list(dataset_mode,gallary_aidrange,gmode)
    if dataset_mode == "train":
        pid_range = (1,51)
    elif dataset_mode == "evaluation":
        pid_range = (51,75)
    elif dataset_mode == "test":
        pid_range = (75,125)
    elif dataset_mode == "74p":
        pid_range = (1, 75)
    for probe in probe_list:
        ppid,pmode,psequence,paid = get_img_inform(probe)
        for i in range(int(num/len(probe_list))):
            gallary = random.choice(gallary_list)
            gpid,gmode,gsequence,gaid = get_img_inform(gallary)
            while ppid == gpid or paid == gaid:
                gallary = random.choice(gallary_list)
                gpid,gmode,gsequence,gaid = get_img_inform(gallary)
            negative_sample_list.append((probe,gallary))
    return negative_sample_list

class MyDataset(Dataset):
    def __init__(self,dataset_mode,probe_aid_range,gallary_aidrange,gmode):
        self.dataset_mode = dataset_mode
        self.probe_aid = probe_aid_range
        self.gallary_aidrange = gallary_aidrange
        self.gmode = gmode
        self.positive_sample = get_positive_sample(dataset_mode,probe_aid_range,gallary_aidrange,gmode)
        self.negative_sample = get_negative_sample(dataset_mode,probe_aid_range,gallary_aidrange,gmode,len(self.positive_sample))
    def __getitem__(self,key):
        if key < len(self.positive_sample):
            img_tensor = torch.stack((totensor(Image.open(self.positive_sample[key][0])),totensor(Image.open(self.positive_sample[key][1]))),dim = 1)
            label = torch.tensor([0.,1.])    # 标签0：识别错误；标签1：识别正确
        else:
            key = key - len(self.positive_sample)
            img_tensor = torch.stack((totensor(Image.open(self.negative_sample[key][0])),totensor(Image.open(self.negative_sample[key][1]))),dim = 1)
            label = torch.tensor([1.,0.])
        return torch.reshape(img_tensor,(2,126,86)),label
    def __len__(self):
        return len(self.positive_sample)+len(self.negative_sample)

'''
p = MyDataset("train",(0,18),(0,18),'nm')
print(len(p))

p1 = get_probe_list("train",(0,180))
print(len(p1))
get_img_path(1,0,'nm',5)'''

def get_test_sample(dataset_mode,probe_aid_range,gallary_aidrange,gmode):
    test_sample_list = []
    probe_img = []
    gallary_list = []
    label = []
    probe_list = get_probe_list(dataset_mode, probe_aid_range)
    for probe in probe_list:
        probe_img = probe
        ppid, pmode, psequence, paid = get_img_inform(probe)
        for gaid in aid_list:
            if gaid>=gallary_aidrange[0] and gaid <= gallary_aidrange[1]:
                for gpid in range(75,125):
                    gallary_list.append(get_img_path(gpid,gaid,gmode,random.randint(1,4)))
                    if ppid == gpid:
                        label.append(1)
                    else:
                        label.append(0)
                test_sample_list.append([probe_img,gallary_list,label])
                gallary_list = []
                label = []
        probe_img = []
    return test_sample_list

