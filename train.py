import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model_test import *


# 定义训练的设备
# device = torch.device("cpu")
device = torch.device("cuda:0")

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, transform=torchvision.transforms.ToTensor()
                                          , download=True)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)


train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集长度为:{}".format(train_data_size))
print("测试数据集长度为:{}".format(test_data_size))

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
model_one = ModelForTest()
# 方法2 使用cuda
model_one.to(device)
# 方法1 使用cuda
# if torch.cuda.is_available():
#     model_one = model_one.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 方法2 使用cuda
loss_fn.to(device)
# 方法1 使用cuda
# if torch.cuda.is_available():
#     loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(model_one.parameters(), lr=learning_rate)


# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮次
epoch = 10

# 添加tensorboard
writer = SummaryWriter("./logs_train")

for i in range(epoch):
    print("------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    model_one.train()
    for data in train_dataloader:
        imgs, targets = data
        # 方法2 使用cuda
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 方法1 使用cuda
        # if torch.cuda.is_available():
        #     imgs = imgs.cuda()
        #     targets = targets.cuda()
        outputs = model_one(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

# torch.no_grad()是一个上下文管理器（context manager）,
# 用于在推断过程中关闭梯度计算以减少内存消耗。
# 在使用torch.no_grad()包裹的代码块中，
# 所有的操作都不会被记录到计算图中，也不会进行梯度计算
    # 测试开始
    model_one.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # 方法2 使用cuda
            imgs = imgs.to(device)
            targets = targets.to(device)

            # 方法1 使用cuda
            # if torch.cuda.is_available():
            #     imgs = imgs.cuda()
            #     targets = targets.cuda()
            outputs = model_one(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            # 测试正确的个数
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    torch.save(model_one, "model_{}.pth".format(i))
    # torch.save(model_one.state_dict(), "model_{}.pth".format(i)) 官方推荐
    print("模型已保存")

writer.close()
