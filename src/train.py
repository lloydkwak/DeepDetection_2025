import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
import gc  # 가비지 컬렉션 모듈 추가

# =============================================================================
# 1. 클래스 및 함수 정의
# =============================================================================

class DeepDetectDataset(Dataset):
    def __init__(self, file_paths=None, labels=None, root_dir=None, split='test', transform=None):
        self.transform = transform
        
        if file_paths is not None and labels is not None:
            self.image_paths = file_paths
            self.labels = labels
        else:
            self.image_paths = []
            self.labels = []
            for label, class_name in enumerate(['Real', 'Fake']):
                class_dir = os.path.join(root_dir, split, class_name)
                if not os.path.exists(class_dir): 
                    class_dir = os.path.join(root_dir, split, class_name.lower())
                
                if os.path.exists(class_dir):
                    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    self.image_paths.extend(files)
                    self.labels.extend([label] * len(files))
                else:
                    print(f"Warning: {class_dir} not found!")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Image not found")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Error loading image: {img_path} / {e}")
            return torch.zeros((3, 224, 224)), torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, torch.tensor(self.labels[idx], dtype=torch.float32)

def get_advanced_model():
    model = EfficientNet.from_pretrained('efficientnet-b4')
    num_ftrs = model._fc.in_features 
    model._fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_acc = 0.0
    
    print(f"학습 시작! (총 Epochs: {epochs})")
    
    for epoch in range(epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': running_loss / (progress_bar.n + 1)})
            
        epoch_train_loss = running_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        # --- Validation ---
        model.eval()
        correct = 0
        total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100 * correct / total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f"\n[결과] Epoch {epoch+1} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%")
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> 최고 성능 갱신! 모델 저장됨 ({best_acc:.2f}%)")
            
    return train_losses, val_losses, val_accuracies

# =============================================================================
# 2. 실행 로직
# =============================================================================

if __name__ == '__main__':
    # A. GPU 및 기본 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    # B. Transforms 정의
    train_transform = A.Compose([
        A.Resize(224, 224, interpolation=3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(224, 224, interpolation=3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # C. 데이터 로드
    DATA_DIR = '../data'
    train_paths = []
    train_labels = []
    
    for label, class_name in enumerate(['Real', 'Fake']):
        class_dir = os.path.join(DATA_DIR, 'train', class_name)
        if not os.path.exists(class_dir): class_dir = os.path.join(DATA_DIR, 'train', class_name.lower())
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            train_paths.extend(files)
            train_labels.extend([label] * len(files))

    # Test Data Load for Split
    all_test_paths = []
    all_test_labels = []
    for label, class_name in enumerate(['Real', 'Fake']):
        class_dir = os.path.join(DATA_DIR, 'test', class_name)
        if not os.path.exists(class_dir): class_dir = os.path.join(DATA_DIR, 'test', class_name.lower())
        if os.path.exists(class_dir):
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            all_test_paths.extend(files)
            all_test_labels.extend([label] * len(files))

    if len(all_test_paths) == 0:
        print("Test 폴더 경로를 확인해주세요.")
        sys.exit()

    val_paths, final_test_paths, val_labels, final_test_labels = train_test_split(
        all_test_paths, all_test_labels, test_size=0.5, random_state=42, stratify=all_test_labels
    )
    print(f"✅ 검증(Val): {len(val_paths)}장 / 최종 테스트(Test): {len(final_test_paths)}장")

    # D. 데이터셋 및 로더
    train_dataset = DeepDetectDataset(file_paths=train_paths, labels=train_labels, transform=train_transform)
    val_dataset = DeepDetectDataset(file_paths=val_paths, labels=val_labels, transform=val_transform)
    test_dataset = DeepDetectDataset(file_paths=final_test_paths, labels=final_test_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=32, pin_memory=True)

    # E. 모델 및 설정
    model = get_advanced_model().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # F. 학습 실행
    try:
        train_loss_history, val_loss_history, val_acc_history = train_model(
            model, train_loader, val_loader, criterion, optimizer, device, epochs=3
        )
        
        # G. 결과 저장
        history_df = pd.DataFrame({
            'epoch': range(1, len(train_loss_history) + 1),
            'train_loss': train_loss_history,
            'val_loss': val_loss_history,
            'val_acc': val_acc_history
        })
        history_df.to_csv('train_log.csv', index=False)
        print("학습 기록이 'train_log.csv' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"학습 중 오류 발생: {e}")

    finally:
        # H. 메모리 정리 (학습 종료 후, 혹은 에러 발생 시에도 실행)
        print("\n[System] 프로세스 종료 절차 시작 및 메모리 정리 중...")

        # 1. 모델 및 옵티마이저 삭제
        if 'model' in locals(): 
            del model
        if 'optimizer' in locals(): 
            del optimizer
        if 'criterion' in locals(): 
            del criterion
            
        # 2. 데이터 로더 및 데이터셋 삭제 (Worker 프로세스 종료 유도)
        if 'train_loader' in locals(): del train_loader
        if 'val_loader' in locals(): del val_loader
        if 'test_loader' in locals(): del test_loader
        if 'train_dataset' in locals(): del train_dataset
        if 'val_dataset' in locals(): del val_dataset
        
        # 3. Python 가비지 컬렉션 강제 실행 (순환 참조 제거)
        gc.collect()

        # 4. GPU 캐시 비우기 (torch가 잡고 있는 미사용 메모리 반환)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() # 멀티프로세싱 관련 잔여 메모리 정리

        print("[System] 모든 리소스 해제 및 메모리 정리 완료.")