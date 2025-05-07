# 仇河东 324085503103 24机械1班 github：(https://github.com/Key407/CHD/tree/main)
## 作业3.py
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap
# 正确导入 linear_model 子模块
from sklearn.linear_model import LogisticRegressionCV

# 设置随机种子和生成数据
# 设置随机种子，保证每次运行代码时生成的随机数据相同
np.random.seed(0)
# 使用 make_moons 函数生成 200 个样本数据，noise=0.20 表示添加一定的噪声
X, y = make_moons(200, noise=0.20)

# 自定义颜色和形状
# 定义两种颜色，分别用于不同类别的数据点
colors = ['CornflowerBlue', 'Tomato']  # 蓝、橙
# 定义两种形状，分别用于不同类别的数据点
markers = ['o', '*']  # o: 圆形, s: 正方形

# 创建一个新的图形窗口，设置图形的大小为 8x6 英寸
plt.figure(figsize=(8, 6))
# 遍历两个类别（0 和 1）
for i in range(2):
    # 绘制不同类别的数据点
    plt.scatter(X[y == i, 0], X[y == i, 1],
                s=50,  # 数据点的大小
                c=colors[i],  # 数据点的颜色
                marker=markers[i],  # 数据点的形状
                label=f'Class {i}',  # 图例标签
                edgecolors='None',  # 数据点边缘颜色
                alpha=0.8)  # 数据点的透明度

# 设置图形的标题
plt.title("Customized Scatter Plot of make_moons")
# 设置 x 轴的标签
plt.xlabel("X1")
# 设置 y 轴的标签
plt.ylabel("X2")
# 显示图例
plt.legend()
# 显示网格线
plt.grid(True)
# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 显示图形
plt.show()

# 定义一个函数，用于绘制决策边界
def plot_decision_boundary(pred_func):
    # 设置边界范围和网格间隔
    # 计算 x 轴的最小值，并减去 0.5 作为边界
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    # 计算 y 轴的最小值，并减去 0.5 作为边界
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # 定义网格的间隔
    h = 0.01
    # 生成网格点
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 网格点的预测
    # 将网格点的坐标展平并合并成一个二维数组
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    # 将预测结果重新调整为网格的形状
    Z = Z.reshape(xx.shape)

    # ===== 自定义填充颜色 =====
    # 定义背景填充颜色的颜色映射
    cmap_background = ListedColormap(['#a0c4ff', '#ffc9c9'])  # 浅蓝+浅橙
    # 绘制填充等高线图，用于表示决策边界
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.6)

    # ===== 自定义点颜色和形状 =====
    # 定义两种颜色，分别用于不同类别的数据点
    colors = ['CornflowerBlue', 'Tomato']
    # 定义两种形状，分别用于不同类别的数据点
    markers = ['o', '*']
    # 遍历两个类别（0 和 1）
    for i in range(2):
        # 绘制不同类别的数据点
        plt.scatter(X[y == i, 0], X[y == i, 1],
                    s=60,  # 数据点的大小
                    c=colors[i],  # 数据点的颜色
                    marker=markers[i],  # 数据点的形状
                    label=f'Class {i}',  # 图例标签
                    edgecolors='None',  # 数据点边缘颜色
                    alpha=0.9)  # 数据点的透明度

# 使用正确导入的 LogisticRegressionCV
# 创建一个逻辑回归模型对象，使用交叉验证选择最优参数
clf = LogisticRegressionCV()
# 使用训练数据对模型进行训练
clf.fit(X, y)
# 创建一个新的图形窗口，设置图形的大小为 8x6 英寸
plt.figure(figsize=(8, 6))
# Plot the decision boundary
# 调用 plot_decision_boundary 函数，绘制逻辑回归模型的决策边界
plot_decision_boundary(lambda x: clf.predict(x))
# 设置图形的标题
plt.title("Logistic Regression")
# 设置 x 轴的标签
plt.xlabel("X1")
# 设置 y 轴的标签
plt.ylabel("X2")
# 显示网格线
plt.grid(True)
# 将图形保存为 ai_net_img_02.png 文件，分辨率为 300 dpi
plt.savefig("ai_net_img_02.png", dpi=300)

# 训练集大小
# 计算训练数据的样本数量
num_examples = len(X)

# 输入层维度（二维坐标输入）
# 输入数据的特征维度，这里是二维坐标
nn_input_dim = 2

# 输出层维度（2 个类别，使用 one - hot 编码）
# 输出层的节点数量，对应两个类别
nn_output_dim = 2

# 梯度下降参数（手动选择的超参数）
# 学习率，控制每次参数更新的步长
epsilon = 0.01
# 正则化强度（L2 正则项系数），用于防止过拟合
reg_lambda = 0.01

# sigmoid 函数
# 定义 sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# sigmoid 函数的导数
# 定义 sigmoid 函数的导数
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# 计算整个数据集上的总损失（用于评估模型效果）
# 定义一个函数，用于计算模型的损失
def calculate_loss(model):
    # 从模型中提取参数
    # 提取输入层到隐藏层的权重矩阵
    W1, b1 = model['W1'], model['b1']
    # 提取隐藏层到输出层的权重矩阵
    W2, b2 = model['W2'], model['b2']

    # 前向传播，计算预测概率
    # 计算输入层到隐藏层的线性组合
    z1 = X.dot(W1) + b1
    # 对线性组合应用 sigmoid 激活函数
    a1 = sigmoid(z1)
    # 计算隐藏层到输出层的线性组合
    z2 = a1.dot(W2) + b2
    # 对线性组合应用指数函数
    exp_scores = np.exp(z2)
    # 计算 softmax 概率分布
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算交叉熵损失（对数损失）
    # 计算每个样本的负对数概率
    correct_logprobs = -np.log(probs[range(num_examples), y])
    # 计算所有样本的损失之和
    data_loss = np.sum(correct_logprobs)

    # 加入 L2 正则化项（防止过拟合）
    # 计算 L2 正则化项
    data_loss += (reg_lambda / 2) * (
            np.sum(np.square(W1)) + np.sum(np.square(W2))
    )

    # 返回平均损失
    return data_loss / num_examples

# 预测函数：根据输入样本 x，输出类别（0 或 1）
# 定义一个函数，用于对输入样本进行预测
def predict(model, x):
    # 解包模型参数
    # 提取输入层到隐藏层的权重矩阵
    W1, b1 = model['W1'], model['b1']
    # 提取隐藏层到输出层的权重矩阵
    W2, b2 = model['W2'], model['b2']

    # 前向传播，计算每个类别的概率
    # 计算输入层到隐藏层的线性组合
    z1 = x.dot(W1) + b1
    # 对线性组合应用 sigmoid 激活函数
    a1 = sigmoid(z1)
    # 计算隐藏层到输出层的线性组合
    z2 = a1.dot(W2) + b2
    # 对线性组合应用指数函数
    exp_scores = np.exp(z2)
    # 计算 softmax 概率分布
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 返回每个样本概率最大的类别索引（即预测结果）
    return np.argmax(probs, axis=1)

# 训练神经网络，学习模型参数并返回最终模型
# 参数说明：
# - nn_hdim：隐藏层的节点数量
# - num_passes：迭代次数（训练轮数）
# - print_loss：是否每 1000 次打印一次损失
# 定义一个函数，用于构建并训练神经网络模型
def build_model(nn_hdim, num_passes=30000, print_loss=False):
    # 设置随机种子，保证每次运行代码时初始化的权重相同
    np.random.seed(0)

    # 参数初始化（权重随机初始化 + 偏置初始化为 0）
    # 初始化输入层到隐藏层的权重矩阵
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    # 初始化输入层到隐藏层的偏置向量
    b1 = np.zeros((1, nn_hdim))
    # 初始化隐藏层到输出层的权重矩阵
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    # 初始化隐藏层到输出层的偏置向量
    b2 = np.zeros((1, nn_output_dim))

    # 定义一个空字典，用于存储模型的参数
    model = {}

    # 训练过程：使用全量批量梯度下降（Batch GD）
    # 迭代训练 num_passes 次
    for i in range(num_passes):
        # -------- 前向传播 --------
        # 计算输入层到隐藏层的线性组合
        z1 = X.dot(W1) + b1
        # 对线性组合应用 sigmoid 激活函数
        a1 = sigmoid(z1)
        # 计算隐藏层到输出层的线性组合
        z2 = a1.dot(W2) + b2
        # 对线性组合应用指数函数
        exp_scores = np.exp(z2)
        # 计算 softmax 概率分布
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # -------- 反向传播 --------
        # 初始化输出层的误差
        delta3 = probs
        # 计算输出层的误差，减去真实标签
        delta3[range(num_examples), y] -= 1
        # 计算输出层权重的梯度
        dW2 = a1.T.dot(delta3)
        # 计算输出层偏置的梯度
        db2 = np.sum(delta3, axis=0, keepdims=True)

        # 计算隐藏层的误差
        delta2 = delta3.dot(W2.T) * sigmoid_derivative(z1)
        # 计算隐藏层权重的梯度
        dW1 = X.T.dot(delta2)
        # 计算隐藏层偏置的梯度
        db1 = np.sum(delta2, axis=0)

        # -------- 正则化（L2）--------
        # 对输出层权重的梯度加上 L2 正则化项
        dW2 += reg_lambda * W2
        # 对隐藏层权重的梯度加上 L2 正则化项
        dW1 += reg_lambda * W1

        # -------- 参数更新（梯度下降）--------
        # 更新输入层到隐藏层的权重矩阵
        W1 -= epsilon * dW1
        # 更新输入层到隐藏层的偏置向量
        b1 -= epsilon * db1
        # 更新隐藏层到输出层的权重矩阵
        W2 -= epsilon * dW2
        # 更新隐藏层到输出层的偏置向量
        b2 -= epsilon * db2

        # 保存更新后的参数
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 每 1000 次输出一次损失值（可选）
        if print_loss and i % 1000 == 0:
            print(f"迭代 {i} 次后的损失值：{calculate_loss(model):.6f}")

    return model

# 构建一个隐藏层维度为 3 的神经网络模型，并训练
# 调用 build_model 函数，构建并训练一个隐藏层节点数为 3 的神经网络模型
model = build_model(nn_hdim=3, print_loss=True)
# 创建一个新的图形窗口，设置图形的大小为 8x6 英寸
plt.figure(figsize=(8, 6))
# 使用训练好的模型绘制决策边界
# 调用 plot_decision_boundary 函数，绘制神经网络模型的决策边界
plot_decision_boundary(lambda x: predict(model, x))
# 设置图形的标题
plt.title("Decision Boundary for hidden layer size 3")
# 设置 x 轴的标签
plt.xlabel("X1")
# 设置 y 轴的标签
plt.ylabel("X2")
# 显示网格线
plt.grid(True)
# 将图形保存为 ai_net_img_03.png 文件，分辨率为 300 dpi
plt.savefig("ai_net_img_03.png", dpi=300)
# 显示图形
plt.show()

# 可视化不同隐藏层节点数对模型决策边界的影响
# 创建一个新的图形窗口，设置图形的大小为 16x28 英寸
plt.figure(figsize=(16, 28))

# 隐藏层节点数量列表
# 定义不同的隐藏层节点数量
hidden_layer_dimensions = [1, 2, 3, 4, 5, 20, 50]

# 遍历不同隐藏层大小，训练模型并绘图
# 遍历不同的隐藏层节点数量
for i, nn_hdim in enumerate(hidden_layer_dimensions):
    # 创建子图：5 行 2 列，第 i+1 个
    plt.subplot(5, 2, i + 1)
    # 设置子图的标题
    plt.title(f"Hidden Layer size: {nn_hdim}")
    # 训练模型
    model = build_model(nn_hdim)
    # 绘制决策边界
    plot_decision_boundary(lambda x: predict(model, x))
    # 设置 x 轴的标签
    plt.xlabel("X1")
    # 设置 y 轴的标签
    plt.ylabel("X2")

# 显示所有子图
# 自动调整子图参数，使之填充整个图像区域
plt.tight_layout()
# 将图形保存为 ai_net_img_04.png 文件，分辨率为 300 dpi
plt.savefig("ai_net_img_04.png", dpi=300)
# 显示图形
plt.show()
```

### 功能描述
这段代码的主要功能是生成月牙形的样本数据，使用逻辑回归模型和自定义的神经网络模型对数据进行分类，并绘制决策边界。同时，还展示了不同隐藏层节点数量对神经网络模型决策边界的影响。
### 使用方法
1、保存作业3.py文件

2、在终端或命令行中执行启动程序

3、生成图片
