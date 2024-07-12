import os, csv, argparse
from datetime import datetime

import torch, yaml, wandb
from torchvision import transforms
from torchvision.models import vit_b_16, mobilenet_v2
from transformers import DeiTForImageClassification, ViTForImageClassification
from torch import nn, optim
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_f1_score
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from configs.config import Config
from RveRNets.models import RveRNet
from RveRNets.datasets import ResizeAndPad, CustomImageDataset, PairedDataset
from RveRNets.optimization import WarmupCosineSchedule
from RveRNets.utils import translate_list_to_str, remove_item_by_value
from RveRNets.inference import model_load


def train(config_path):
    cfg = Config(config_path)
    
    # wandb init
    os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    wandb.init(project=cfg.wandb_project, dir=cfg.wandb_logdir)    
    wandb.run.name = cfg.wandb_run
    wandb.run.save()    

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    current_date = datetime.now().strftime("%Y%m%d") # Get the current date in YYYYMMDD format    
      
    ## load the dict which contains the numeric definitions of categories
    with open(cfg.class_dict, 'r') as stream:
        try:
            dataset_labl=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    if cfg.ambiguity_test is False:
        # If ambiguous classes are not trained together with the original dataset, 
        # remove the values corresponding to the class numbers from dataset_labl.
        for value_to_remove in cfg.ambiguity_classes:
            # Remove key-value pairs with the specified value
            remove_item_by_value(dataset_labl, value_to_remove)
            
    
    # Data preprocess and augmentation
    data_transforms = {
        'train': transforms.Compose([
            ResizeAndPad(target_size=cfg.input_imgsz),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            ]),
        'val': transforms.Compose([
            ResizeAndPad(target_size=cfg.input_imgsz),
            transforms.ToTensor(),
        ]),
    }
    
    # Initialize Custom train dataset
    bs = cfg.batch_size 
    dataset1 = CustomImageDataset(cfg.data_for_net1, dataset_labl, transform=data_transforms['train']) 
    dataset2 = CustomImageDataset(cfg.data_for_net2, dataset_labl, transform=data_transforms['train'])
          
    paired_dataset = PairedDataset(dataset1, dataset2)
    paired_loader = DataLoader(paired_dataset, batch_size=bs, shuffle=True, num_workers=2)
    
    # Initialize Custom valid dataset
    val_set1 = CustomImageDataset(cfg.valset_for_net1, dataset_labl, transform=data_transforms['val'])
    val_set2 = CustomImageDataset(cfg.valset_for_net2, dataset_labl, transform=data_transforms['val'])
    paired_valset = PairedDataset(val_set1, val_set2)
    paired_valloader = DataLoader(paired_valset, batch_size=bs, shuffle=True)
    
    
    label2cat = os.listdir(dataset1.img_dir)
    # Assuming output_size is defined (number of classes in your dataset)
    output_size = len(label2cat)  # able to replace with actual number of classes
    
    # Create instances of backbones    
    backbones = {'mnet': mobilenet_v2(pretrained=True), 'vit' : vit_b_16(pretrained=True), 'deit': ViTForImageClassification.from_pretrained(cfg.deit_ver,cache_dir=cfg.deit_cache, num_labels=1000), 'deit-dist': DeiTForImageClassification.from_pretrained(cfg.deit_dist_ver,cache_dir=cfg.deit_cache, num_labels=1000)}
    # Initialize RveR Network    
    if cfg.from_checkpoint:
        model = model_load(cfg.checkpoint_path, cfg.net1, cfg.net2, cfg.input_imgsz, output_size)
    else:
        model = RveRNet(backbones[cfg.net1], backbones[cfg.net2], cfg.input_imgsz, output_size) #.to(device)
    model.unfreeze([model.roi_module, model.extra_roi_module]) # If would like to freeze extra_roi_module only, please designate roi_module and vice versa
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)
    
    # Initialize the Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    
    # Training and validation loop
    num_epochs = cfg.num_epochs  # Define the number of epochs
    num_train_optimization_steps = int(len(paired_dataset) / bs) * num_epochs
    
    scheduler = WarmupCosineSchedule(optimizer,
                                 warmup_steps=num_train_optimization_steps*0.1,
                                 t_total=num_train_optimization_steps)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Directory where you want to save the models
    save_dir = f"{cfg.model_save_path}_{current_date}"
    os.makedirs(save_dir, exist_ok=True)    
    
    # Write the configurations of current run to txt file
    cfg.export_to_txt(os.path.join(save_dir,'train_config.txt'))
    
    # Open the CSV file    
    wfile = open(os.path.join(save_dir,'training_logs.csv'), mode='a', newline='')
    writer = csv.writer(wfile)
    
    if cfg.ambiguity_test:
        writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Training F1 Score', 'Validation Loss', 'Validation Accuracy', 'Validation F1 Score', 'Validation Accuracy (Classes %s)'%(translate_list_to_str(cfg.ambiguity_classes)), 'Validation F1 Score (Classes %s)'%(translate_list_to_str(cfg.ambiguity_classes))])
    else:
        writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Training F1 Score', 'Validation Loss', 'Validation Accuracy', 'Validation F1 Score'])
    
    
    for epoch in range(num_epochs):                
        
        ############## Training phase ###################
        model.train()
        
        train_loss = 0.0
        total_acc_train = 0
        all_labels_train = []
        all_predictions_train = []
        
        train_loader = tqdm(paired_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs1, inputs2, labels, _ in train_loader:
            # inputs1 : input mini-batch for roi module
            # inputs2 : input mini-batch for extra-roi module
            
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            optimizer.zero_grad() # initialization of grad
            outputs = model(inputs1, inputs2) 
            
            acc = (outputs.argmax(dim=1) == labels).sum().item() # Sum up the number of correct predictions in the mini-batch
            total_acc_train += acc # total_acc_train is the accuracy after the summation of all mini-batches
            
            loss = criterion(outputs, labels) # loss function
            loss.backward() # error backpropagation
            optimizer.step() # Step the scheduler
            
            train_loss += loss.item() # accumulation of the loss of every mini-batch
            train_loader.set_postfix(loss=train_loss/len(paired_loader))
            
            # Accumulate labels and predictions for F1 score
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(outputs.argmax(dim=1).cpu().numpy())
        
        
        # Calculate F1 score for training
        train_f1_score = multiclass_f1_score(torch.tensor(all_predictions_train), torch.tensor(all_labels_train), num_classes=len(label2cat))
        
        ############ Validation phase ########################
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        if cfg.ambiguity_test:            
            correct_ambiguous_classes = 0
            total_ambiguous_classes = 0
        
        all_labels_val = []
        all_predictions_val = []
        val_loader = tqdm(paired_valloader, desc=f"Epoch {epoch+1}/{num_epochs} [Validate]")
        
        with torch.no_grad():
            for inputs1, inputs2, labels, _ in val_loader:
                inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
                outputs = model(inputs1, inputs2)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loader.set_postfix(loss=val_loss/len(paired_valloader))                
                
                mask_ambiguous_classes = 'init'
                if cfg.ambiguity_test:
                    for cls_num in cfg.ambiguity_classes:
                        if mask_ambiguous_classes == 'init':
                            mask_ambiguous_classes = (labels == cls_num)
                        else:
                            mask_ambiguous_classes |= (labels == cls_num)
                                    
                    correct_ambiguous_classes += (predicted[mask_ambiguous_classes] == labels[mask_ambiguous_classes]).sum().item()
                    total_ambiguous_classes += mask_ambiguous_classes.sum().item()                
                
                # Accumulate labels and predictions for F1 score
                all_labels_val.extend(labels.cpu().numpy())
                all_predictions_val.extend(outputs.argmax(dim=1).cpu().numpy())
                
        # Calculate F1 score for validation
        val_f1_score = multiclass_f1_score(torch.tensor(all_predictions_val), torch.tensor(all_labels_val), num_classes=len(label2cat), average="macro")
        val_accuracy = 100 * correct / total
        
        if cfg.ambiguity_test:
            # Calculate additional validation accuracy for ambiguous classes
            val_accuracy_ambiguous_classes = 100 * correct_ambiguous_classes / total_ambiguous_classes
            
            # Calculate F1 score for validation for classes ambiguous classes
            val_f1_score_ambiguous_classes = f1_score(all_labels_val, all_predictions_val, labels=cfg.ambiguity_classes, average='macro')
            
            print(f"Epoch {epoch+1}, Training Loss: {train_loss/len(paired_loader):.4f}, | Accuracy: {total_acc_train / len(paired_dataset)*100: .3f}%, Training F1 Score: {train_f1_score:.3f},  Validation Loss: {val_loss/len(paired_valloader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation F1 Score: {val_f1_score:.3f}, Validation Accuracy (Classes {translate_list_to_str(cfg.ambiguity_classes)}): {val_accuracy_ambiguous_classes:.2f}%, Validation F1 Score (Classes {translate_list_to_str(cfg.ambiguity_classes)}): {val_f1_score_ambiguous_classes:.3f}")
            
            wandb.log({"Epoch": epoch+1, "Training Loss": train_loss/len(paired_loader), "Accuracy" : total_acc_train / len(paired_dataset)*100, "Training F1 Score": train_f1_score, "Validation Loss" : val_loss/len(paired_valloader), "Validation Accuracy" : val_accuracy, "Validation F1 Score" : val_f1_score, "Validation Accuracy (Classes %s)"%(translate_list_to_str(cfg.ambiguity_classes)): val_accuracy_ambiguous_classes, "Validation F1 Score (Classes %s)"%(translate_list_to_str(cfg.ambiguity_classes)) :  val_f1_score_ambiguous_classes }) 
                                  
            # Write metrics to CSV file
            writer.writerow([
                epoch + 1,
                train_loss / len(paired_loader),
                total_acc_train / len(paired_dataset) * 100,
                train_f1_score.item(),
                val_loss / len(paired_valloader),
                val_accuracy,
                val_f1_score.item(),
                val_accuracy_ambiguous_classes,
                val_f1_score_ambiguous_classes.item()
            ])
            
        else:
            
            print(f"Epoch {epoch+1}, Training Loss: {train_loss/len(paired_loader):.4f}, | Accuracy: {total_acc_train / len(paired_dataset)*100: .3f}%, Training F1 Score: {train_f1_score:.3f},  Validation Loss: {val_loss/len(paired_valloader):.4f}, Validation Accuracy: {val_accuracy:.2f}%, Validation F1 Score: {val_f1_score:.3f}")
            
            wandb.log({"Epoch": epoch+11, "Training Loss": train_loss/len(paired_loader), "Accuracy" : total_acc_train / len(paired_dataset)*100, "Training F1 Score": train_f1_score, "Validation Loss" : val_loss/len(paired_valloader), "Validation Accuracy" : val_accuracy, "Validation F1 Score" : val_f1_score }) 
                                  
            # Write metrics to CSV file
            writer.writerow([
                epoch + 1,
                train_loss / len(paired_loader),
                total_acc_train / len(paired_dataset) * 100,
                train_f1_score.item(),
                val_loss / len(paired_valloader),
                val_accuracy,
                val_f1_score.item()
            ])
            
        
        scheduler.step()
        
        if (epoch+1) % cfg.save_model_every == 0:
            # Save the model            
            model_path = os.path.join(save_dir, f"{cfg.model_file_prefix}_epoch_{epoch+1}_{current_date}.pth")
            torch.save(model.state_dict(), model_path) # Since the training was done with DataParallel, an additional 'module' key is added to the state_dict. Considering this, you need to load the model accordingly later. If you find this cumbersome, you can save the model using model.module.state_dict().
    wfile.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,help='path to train_config file')
    args = parser.parse_args()
    train(args.config)