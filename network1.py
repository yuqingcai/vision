from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms

data_path = './data-unversioned/cifar10'
cifar10 = datasets.CIFAR10(data_path, train=True, download=True)
cifar10_test = datasets.CIFAR10(data_path, train=False, download=True)

car_images = [img for img, label in cifar10 if label == 1]

# 设置画布大小
plt.figure(figsize=(10, 10))

# 显示前 25 张汽车图片（如果有这么多）
for i, img in enumerate(car_images[:25]): # 最多显示 25 张
    plt.subplot(5, 5, i + 1)  # 创建 5x5 的子图
    plt.imshow(img)
    plt.axis('off')  # 隐藏坐标轴

plt.show()
