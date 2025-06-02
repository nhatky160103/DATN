import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from backbones.iresnet_plus import (
    iresnet18_plus, 
    iresnet34_plus, 
    iresnet50_plus, 
    iresnet100_plus, 
    iresnet200_plus
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class FaceRecognitionModel(nn.Module):
    def __init__(self, backbone):
        super(FaceRecognitionModel, self).__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(512, 1000)  # Giả sử có 1000 classes
        
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Tính các metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_comparison(results):
    # Chuẩn bị dữ liệu cho biểu đồ
    backbones = [r['name'] for r in results]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Tạo subplot cho từng metric
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        values = [r[metric] for r in results]
        sns.barplot(x=backbones, y=values, ax=axes[idx])
        axes[idx].set_title(f'{metric.capitalize()} Comparison')
        axes[idx].set_ylim(0, 1)
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('backbone_comparison.png')
    plt.close()

def main():
    # Thiết lập device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Danh sách các backbone cần test
    backbones = {
        'iresnet18_plus': iresnet18_plus(),
        'iresnet34_plus': iresnet34_plus(),
        'iresnet50_plus': iresnet50_plus(),
        'iresnet100_plus': iresnet100_plus(),
        'iresnet200_plus': iresnet200_plus()
    }
    
    # Load test dataset
    # Giả sử bạn có một dataset class và dataloader
    # test_dataset = YourFaceDataset(...)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    results = []
    
    # Đánh giá từng backbone
    for name, backbone in backbones.items():
        print(f"\nEvaluating {name}...")
        
        # Tạo model với backbone hiện tại
        model = FaceRecognitionModel(backbone).to(device)
        
        # Load pretrained weights nếu có
        # model.load_state_dict(torch.load(f'weights/{name}.pth'))
        
        # Đánh giá model
        metrics = evaluate_model(model, test_loader, device)
        
        # Thêm kết quả
        results.append({
            'name': name,
            **metrics
        })
        
        # In kết quả
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
    
    # Vẽ biểu đồ so sánh
    plot_comparison(results)
    
    # In bảng so sánh
    print("\nComparison Table:")
    print("-" * 100)
    print(f"{'Backbone':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['name']:<15} {result['accuracy']:<10.4f} {result['precision']:<10.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<10.4f}")

if __name__ == "__main__":
    main() 