# 1. Test 데이터셋 로드

test_dir = '../data' 

test_dataset = DeepDetectDataset(root_dir=test_dir, split='test', transform=val_transform) 

test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=32)
print(f"테스트 데이터 개수: {len(test_dataset)}장")

model = get_advanced_model().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

print("최종 테스트(Final Test) 시작...")
print(f"사용하는 데이터: 위에서 분할한 {len(final_test_paths)}장 (Validation에 쓰지 않은 나머지 50%)")

# 2. 평가 진행
# ⚠️ 중요: 위에서 만든 'test_loader'를 그대로 사용합니다. (새로 만들지 않음!)
correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        outputs = model(images)
        predicted = (outputs > 0.5).float()
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 나중에 정밀 분석(Confusion Matrix 등)을 위해 기록
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

acc = 100 * correct / total
print(f"\n==============================")
print(f"🏆 최종 테스트 정확도: {acc:.2f}%")
print(f"==============================")

# (선택 사항) CSV로 저장하고 싶다면
import pandas as pd
result_df = pd.DataFrame({'True_Label': [x[0] for x in y_true], 'Pred_Label': [x[0] for x in y_pred]})
result_df.to_csv('final_test_results.csv', index=False)