import os.path
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# 定义数据集的转换操作
resize_transform = transforms.Resize((390, 490))
transform = transforms.Compose([resize_transform, transforms.ToTensor()])

# 加载训练数据集
train_dataset = datasets.ImageFolder(
    root="D:/pycharmtest/pytorch_test/data/train_data",
    transform=transform
)

# 加载测试数据集
test_dataset = datasets.ImageFolder(
    root="D:/pycharmtest/pytorch_test/data/test_imagesset",
    transform=transform
)

# 创建训练数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 定义模型、损失函数和优化器
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc = nn.Linear(573300, 5)  # 修改全连接层的输入维度为390*490
        self.pth = "D:/pycharmtest/pytorch_test/model.pth"
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入展平为一维向量
        x = self.fc(x)  # 将输入传递给全连接层
        return x
model = TestModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

#检察是否存在训练好的模型，如果有则加载使用
model_path="D:/pycharmtest/pytorch_test/model.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path)
    if 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("使用存在的模型训练")
else:
    start_epoch=0
    print("进行新的训练")

# 训练模型
epochs = 20
for epoch in range(#start_epoch,
        # start_epoch +
        epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

#保存模型
checkpoint={
    'epoch': epoch + 1,
    'model_state_dict':model.state_dict(),
    'optimizer_state_dict':optimizer.state_dict()
}
torch.save(model.state_dict(), "model.pth")

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 打印训练结果和准确率
accuracy = 100 * correct / total
print("Accuracy: {}%".format(accuracy))


