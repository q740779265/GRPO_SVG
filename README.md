# 使用说明
## 环境设置
```
 conda install --yes --file requirements.txt
```
可能安装cuda时有坑，注意一下对应pytorch的2.4.0版本
## 启动训练
先把verl-main这个路径手动添加到系统搜索路径中
嫌麻烦的话，直接打开verl-main/verl/trainer/main_ppo.py，把我注释掉的解注一下：
```
# import sys
# new_path = 'xxx/xxx/verl-main'(verl-main的绝对路径复杂进去)
# sys.path.append(new_path)
```
然后cd到verl-main，在终端输入下面命令就可以开始训练了：
```
.entrance.sh
```
