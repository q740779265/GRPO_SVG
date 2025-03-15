# 使用说明
## 环境设置
```
 conda install --yes --file requirements.txt
```
可能安装cuda时有坑，注意一下对应pytorch的2.4.0版本
## 参数传入
GRPO的训练参数文件放在verl-main/verl/trainer/config/grpo_trainer.yaml中
### 修改训练集路径
由于我路径传入都是我这边绝对路径，所以重新部署时得改一下，数据集已经包含在文件夹中了

根据路径verl-main/examples/datasets/svg/train.parquet找到train.parquet文件，复制绝对路径，并在yaml文件中修改以下两个参数，val_files是没用的，但因为不能空所以直接填的训练集路径
```
# xxx改成你那边的绝对路径
train_files: xxx/verl-main/examples/datasets/svg/train.parquet
val_files: xxx/verl-main/examples/datasets/svg/train.parquet
```


## 启动训练
先把verl-main这个路径手动添加到系统搜索路径中

嫌麻烦的话，直接打开verl-main/verl/trainer/main_ppo.py，把我注释掉的解注一下：
```
# import sys
# new_path = 'xxx/xxx/verl-main'(verl-main的绝对路径复制进去)
# sys.path.append(new_path)
```
然后cd到verl-main，在终端输入下面命令就可以开始训练了：
```
.entrance.sh
```
