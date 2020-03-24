# DGA_Detection
使用深度学习的方式进行DGA域名检测


### 数据
黑样本：  
&emsp;&emsp;训练集和验证集黑样本来自于2018年以前的netlab DGA域名  
&emsp;&emsp;测试集黑暗样本来自2018-2020之间netlab发现的 DGA域名  
白样本：  
&emsp;&emsp;Alexa域名top n与实验室正常域名库  

### 项目结构
```
|-- data_mining_toolbox  自己写的数据挖掘工具库，需要的可以关注记得关注的我的仓库，持续更新...
|-- images  训练过程Loss、Acc变化图
    |-- {model}-model-epoch-{epoch}-loss.png loss在训练和验证集上的变化图
    |-- {model}-model-epoch-{epoch}-acc.png acc在训练和验证集上的变化图
|-- model  模型参数
|-- report  模型效果比较
    |-- {model}_compare_diff_epoch_report.csv 模型不同轮数的表现
    |-- model_compare_report.csv  不同模型表现


