import numpy as np
import matplotlib.pyplot as plt

est = 'saved_trajectories/HoiHa.txt'
data = np.loadtxt(est, delimiter=" ")[::,[1,2]]
x = data[::,[0]]
y = data[::,[1]]
plt.figure()
plt.plot(x, y, marker='o', linestyle='-')  # 使用圆点标记和实线连接数据点
plt.xlabel('X')  # 设置 x 轴标签
plt.ylabel('Y')  # 设置 y 轴标签
plt.savefig('HoiHa.png')