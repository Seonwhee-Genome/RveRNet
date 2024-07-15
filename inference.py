import torch, cv2, os, shutil, argparse

import yaml
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import vit_b_16, mobilenet_v2
from transformers import DeiTForImageClassification, ViTForImageClassification
from torch.utils.data import DataLoader
from torchvision.utils import save_image


# 내가 정의한 custom module
from RveRNets.models import RveRNet
from RveRNets.datasets import ResizeAndPad, CustomImageDataset, PairedDataset

from configs.config import Config


def model_load(cfg):
    
    # Create instances of backbones
    backbones = {'mnet': mobilenet_v2(pretrained=True), 'vit' : vit_b_16(pretrained=True), 'deit': ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224',cache_dir=cfg.hf_cache_dir, num_labels=1000), 'deit-dist': DeiTForImageClassification.from_pretrained("facebook/deit-base-distilled-patch16-224",cache_dir=cfg.hf_cache_dir, num_labels=1000)}
    
    # Load the trained model
    model = RveRNet(backbones[cfg.roi_module], backbones[cfg.extra_roi_module], cfg.input_imgsz, cfg.num_classes)
    model_tmp = torch.load(cfg.ckpt_path, map_location=torch.device("cuda"))
    # Remove the 'module.' prefix from keys
    # If you load the state_dict directly, it will have the 'module' prefix 
    # because it was saved using DataParallel during training, which causes an AttributeError. 
    # First, load the state_dict and then remove the 'module' prefix from the keys.
    new_state_dict = {}
    for key, value in model_tmp.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]  # Remove the 'module.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    model.load_state_dict(new_state_dict)
    return model



def data_process(path1, path2, imgsz):
    data_transforms = transforms.Compose([
            ResizeAndPad(imgsz),
            transforms.ToTensor(),
        ])
    
    # Prepare an image for inference
    image1 = cv2.imread(path1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    
    if isinstance(image1, np.ndarray):
        # Convert numpy array to PIL image
        image1 = Image.fromarray(image1)
    image1 = data_transforms(image1).unsqueeze(0)  # Apply the same transformation used during training
    
    image2 = cv2.imread(path2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    if isinstance(image2, np.ndarray):
        # Convert numpy array to PIL image
        image2 = Image.fromarray(image2)
    image2 = data_transforms(image2).unsqueeze(0)  # Apply the same transformation used during training
    return image1, image2

    
def do_inference(cfg):
    model = model_load(cfg)
    model.eval()
    img1, img2 = data_process(cfg.roi_img_path, cfg.extra_roi_img_path, cfg.input_imgsz)
    # Perform inference
    with torch.no_grad():
        outputs = model(img1, img2)  # Assuming your model takes two images as input
    pred = outputs.argmax(dim=1)
            
    del model, outputs
    torch.cuda.empty_cache()
    
    label_dict = convert_cls_int_to_txt(cfg.cls_yaml)
    return label_dict[pred.item()]


def do_batch_inference(cfg):
    '''
    Code that performs inference on a large amount of data with given labels, like a validation set, all at once using the GPU   
    '''
    torch.manual_seed(0)
    np.random.seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_labl = get_cls_int_txt(cfg.cls_yaml) 
            
    # 데이터셋 전처리 및 augmentation
    data_transforms = {
        'train': transforms.Compose([
            ResizeAndPad(target_size=cfg.input_imgsz),
            transforms.RandomHorizontalFlip(), # train에서는 이미지 뒤집기 augmentation 추가
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            ResizeAndPad(target_size=cfg.input_imgsz),
            transforms.ToTensor(),
        ]), 
    }
    
    dataset1 = CustomImageDataset(cfg.roi_batch_path, dataset_labl, transform=data_transforms['val'])
    dataset2 = CustomImageDataset(cfg.extra_roi_batch_path, dataset_labl, transform=data_transforms['val'])
      
    paired_dataset = PairedDataset(dataset1, dataset2)
    paired_loader = DataLoader(paired_dataset, batch_size=cfg.bs, shuffle=True)    

    model = model_load(cfg)
    model.to(device)
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_filenames = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        
        for inputs1, inputs2, labels, filename in paired_loader:  
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)             
            outputs = model(inputs1, inputs2)  # Assuming your model takes two images as input 
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            confidence = torch.exp(outputs).max(dim=1)
            
            # Accumulate labels and predictions for F1 score
            all_filenames.extend(filename)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            
    
    labl_dict = convert_cls_int_to_txt(cfg.cls_yaml)
    gt_all = list(map(lambda num: labl_dict[num], all_labels))
    pred_all = list(map(lambda num: labl_dict[num], all_predictions))
            
    del model, outputs
    torch.cuda.empty_cache()
    acc = correct / total * 100
    print(f"{correct} correctly predicted from total {total} dataset: the accuracy is {acc}")
    return {"file": all_filenames, "gt": gt_all, "pred": pred_all}


def get_cls_int_txt(yaml_name)->dict:
    '''
    A function that reads a YAML file, where category names and index numbers are defined, into a dictionary and returns the dictionary
    '''
    with open(yaml_name, 'r') as stream:
        try:            
            dataset_labl=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    return dataset_labl


def convert_cls_int_to_txt(yaml_name)->dict:
    '''
    A function that swaps the keys and values of a dictionary where category names and index numbers are defined
    '''
    dataset_labl = get_cls_int_txt(yaml_name)    
            
    label_dict = {v: k for k, v in dataset_labl.items()}
    return label_dict


def inference(config_path):
    cfg = Config(config_path)
    
    if cfg.batch_inference:
        out = do_batch_inference(cfg)
    else:
        out = do_inference(cfg)
    print(out)
        

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to test_config file')
    args = parser.parse_args()
    inference(args.config)   
