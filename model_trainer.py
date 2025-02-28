import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

class FaceDataset(Dataset):
    """Face dataset for training the face recognition model."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FaceRecognitionModel(nn.Module):
    """Face recognition model based on a pre-trained CNN."""
    def __init__(self, num_classes, feature_extract=True, model_name='mobilenet'):
        super(FaceRecognitionModel, self).__init__()
        
        if model_name == 'resnet18':  # Using ResNet18 instead of ResNet50
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features
            
            # Freeze parameters if feature extracting
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final fully connected layer
            self.model.fc = nn.Sequential(
                nn.Linear(num_features, 256),  # Reduced from 512 to 256
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features
            
            # Freeze parameters if feature extracting
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        elif model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
            num_features = self.model.classifier[1].in_features
            
            # Freeze parameters if feature extracting
            if feature_extract:
                for param in self.model.parameters():
                    param.requires_grad = False
            
            # Replace the final classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def forward(self, x):
        return self.model(x)

def evaluate_model(model, dataloader, criterion, device):
    """Evaluate the model on the test set."""
    model.eval()
    
    # Initialize variables
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # Iterate over data
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Save predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, all_preds, all_labels

def plot_training_history(history, save_path=None):
    """Plot the training and validation loss and accuracy."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, scheduler=None):
    """Train the model."""
    # Initialize variables
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Track metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass and optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if scheduler is not None and phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Track history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            # Deep copy the model if it's the best performance on validation set
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict().copy()
        
        print()
    
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def main():
    """Main function to train the face recognition model."""
    parser = argparse.ArgumentParser(description="Train face recognition model")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root directory containing the data folders (train, val, test)")
    parser.add_argument("--model-dir", type=str, default="data/model",
                        help="Directory to save the trained model")
    parser.add_argument("--model-type", type=str, default="mobilenet",
                        choices=["resnet18", "efficientnet", "mobilenet"],
                        help="Type of model architecture to use")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--num-epochs", type=int, default=20,
                        help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--feature-extract", action="store_true",
                        help="Use pre-trained model as feature extractor (freeze base layers)")
    parser.add_argument("--fine-tune", action="store_true",
                        help="Fine-tune the entire model")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force using CPU even if CUDA is available")
    
    args = parser.parse_args()
    
    # Create the model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Create datasets
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")
    
    # Change: Always look for class_mapping.json in the model directory
    mapping_file = os.path.join(args.model_dir, "class_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            class_mapping = json.load(f)
        num_classes = len(class_mapping)
        print(f"Loaded class mapping with {num_classes} classes from {mapping_file}")
    else:
        # Count the number of classes from the training directory
        classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        num_classes = len(classes)
        print(f"No class mapping found at {mapping_file}. Using {num_classes} classes from training directory")
    
    # Create datasets and dataloaders
    image_datasets = {
        'train': FaceDataset(train_dir, transform=data_transforms['train']),
        'val': FaceDataset(val_dir, transform=data_transforms['val']),
        'test': FaceDataset(test_dir, transform=data_transforms['test'])
    }
    
    # Print dataset sizes
    print(f"Training set size: {len(image_datasets['train'])} images")
    print(f"Validation set size: {len(image_datasets['val'])} images")
    print(f"Test set size: {len(image_datasets['test'])} images")
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(image_datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(image_datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=2)
    }
    
    # Initialize the model
    model = FaceRecognitionModel(
        num_classes=num_classes,
        feature_extract=args.feature_extract,
        model_name=args.model_type
    )
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Set different learning rates for different parts of the model
    if args.feature_extract:
        # Only update the newly added layers
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
    else:
        # Update all parameters
        params_to_update = model.parameters()
    
    optimizer = optim.Adam(params_to_update, lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train the model
    print("Starting training...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, device,
        num_epochs=args.num_epochs, scheduler=scheduler
    )
    
    # Plot training history
    plot_training_history(history, save_path=os.path.join(args.model_dir, "training_history.png"))
    
    # Evaluate the model on the test set
    print("Evaluating on test set...")
    test_loss, test_acc, y_pred, y_true = evaluate_model(
        model, dataloaders['test'], criterion, device
    )
    
    # Generate classification report
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            class_mapping = json.load(f)
        target_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    else:
        target_names = [str(i) for i in range(num_classes)]
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("Classification Report:")
    print(report)
    
    # Save the classification report
    with open(os.path.join(args.model_dir, "classification_report.txt"), 'w') as f:
        f.write(report)
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(os.path.join(args.model_dir, "confusion_matrix.png"))
    
    # Save the model
    model_path = os.path.join(args.model_dir, f"face_recognition_{args.model_type}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'num_classes': num_classes,
        'model_type': args.model_type,
        'feature_extract': args.feature_extract
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Change: Always create and save class mapping to the model directory
    if not os.path.exists(mapping_file):
        # Create class mapping
        class_mapping = {str(i): image_datasets['train'].classes[i] 
                         for i in range(len(image_datasets['train'].classes))}
        
        # Save class mapping
        with open(mapping_file, 'w') as f:
            json.dump(class_mapping, f, indent=4)
        print(f"Class mapping saved to {mapping_file}")

if __name__ == "__main__":
    main()