# 使用说明
## 环境设置
```
conda install --yes --file requirements.txt
```
可能安装cuda时有坑，注意一下对应pytorch的2.4.0版本
## 修改配置文件
GRPO的训练参数文件放在verl-main/verl/trainer/config/grpo_trainer.yaml中，打开它
### 修改训练集路径
由于我路径传入都是我这边绝对路径，所以重新部署时得改一下，数据集已经包含在文件夹中了

根据路径verl-main/examples/datasets/svg/train.parquet找到train.parquet文件，复制绝对路径，并在yaml文件中修改以下两个参数，val_files是没用的，但因为不能空所以直接填的训练集路径
```
train_files: xxx/verl-main/examples/datasets/svg/train.parquet
val_files: xxx/verl-main/examples/datasets/svg/train.parquet
```
### 修改actor模型路径
找到actor_rollout_ref里的path，改成你那边的路径
```
model:
 path: xxx/qwen2.5-coder-7b
```
### 调整超参数
下面解释一些超参数，reward那边的超参数最好不动
```
train_batch_size   # 批次大小
enable_gradient_checkpointing   # 梯度检查点
save_freq   # 保存步长
total_epochs   # 训练轮次
n # 每个问题rollout个数
max_response_length   # 最大回答长度
```
## 修改reward模型路径
因为存在两个reward model所以我们并没有把reward集成在verl框架中

## 启动训练
先把verl-main这个路径手动添加到系统搜索路径中

嫌麻烦的话，直接打开verl-main/verl/trainer/main_ppo.py，把我注释掉的解注一下：
```
# import sys
# new_path = 'xxx/xxx/verl-main'(verl-main的绝对路径复制进去)
# sys.path.append(new_path)
```
如果显卡支持P2P，进去根目录下的entrance.sh把P2P环境变量设置删了

然后cd到verl-main，在终端输入下面命令就可以开始训练了：
```
.entrance.sh
```
## 启动Tensorboard
启动训练后，系统会在verl-main/outputs/tensorboard文件夹下新建一个log文件夹，假如这个文件夹叫yyy

接着新建一个终端，输入以下命令，即可启动Tensorboard：
```
tensorboard --logdir xxx/verl-main/outputs/tensorboard/yyy
```
