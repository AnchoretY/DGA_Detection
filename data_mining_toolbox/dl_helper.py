import os
import time
import torch
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import classification_report,\
    confusion_matrix,precision_score,recall_score,roc_auc_score

from Data_Mining_Toolbox.plot_helper import plot_train_curve



def _train_epoch(model, optimizer, epoch, x, y, batch_size):
    """
        进行1轮模型训练
        Parameters:
        ----------------
            model: 模型
            optimier: 优化器
            epoch:当前轮次
            x:输入数据，需要为Tensor
            y:标签,必须为Tensor
            batch_size:批处理大小
    """
    model.train()
    start_time = time.time()
    loss = 0
    correct = 0
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        label = y[index : index + batch_size]
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, label)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()
        
    print("Train epoch: {} \t Time elapse: {}s".format(epoch, int(time.time() - start_time)))
    print("Train Loss: {}  \t 训练准确率: {}".format(round(loss.item(), 4), round(correct.item() / len(x), 4)))
    
    loss_epoch = round(loss.item(), 4)
    acc_epoch  = round(correct.item()/len(x),4)

    return loss_epoch,acc_epoch

def _test_epoch(is_val, model, epoch, x, y, batch_size):
    """ 
        进行1轮模型测试
        Parameters:
        -------------------
        model: 模型
        optimier: 优化器
        epoch:当前轮次
        x:输入数据，需要为Tensor
        y:标签,必须为Tensor
        batch_size:批处理大小
    """
    model.eval()
    loss = 0
    correct = 0
    test_result = []
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        label = y[index : index + batch_size]
        output = model(data)
        loss += F.cross_entropy(output, label, size_average=False).item()
        pred = output.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum()
        test_result += [i for i in pred.cpu().numpy()]

    if(is_val):
        print("Val loss: {} \t 验证准确率: {}".format(round(loss / len(x), 6), round(correct.item() / len(x), 6)))
    else:
        print("test loss: {} \t 测试准确率: {}".format(round(loss / len(x), 6), round(correct.item() / len(x), 6)))
        print(classification_report(y.cpu(), test_result, digits=6))
        
    print()
    loss_epoch = round(loss / len(x), 4)
    acc_epoch  = round(correct.item() / len(x), 4)

    return loss_epoch,acc_epoch
        
def train(model,train_x,train_y,val_x,val_y, epochs, batch_size,save_prefix,optimizer=None):
    """
        模型训练并将模型参数，并将训练过程中每次在验证集上效果有提升时的参数进行保存
        Prameters:
        --------------
            model: 待训练模型
            optimizer: 优化器
            train_x: 训练数据，Tensor类型
            train_y: 训练数据标签，Tensor类型
            val_x: 验证集数据，Tensor类型
            val_y: 验证集标签，Tensor类型
            epochs: 要训练的轮数
            batch_size: 批处理大小
            save_prefix: 存储使得前缀名，一般为模型名，最终存储格式为{}-model-epoch-{}'.format(save_prefix,epoch)
            optimizer: 优化器

    """
    max_val_score = 0
    
    train_loss_list,train_acc_list = [],[]
    val_loss_list,val_acc_list = [],[]
    
    if optimizer is None:
        optimizer = optim.Adam(params=model.parameters(), lr=0.0001, weight_decay=0.00001)
        
    for epoch in range(1, epochs + 1):
        
        train_loss_epoch,train_acc_epoch = _train_epoch(model, optimizer, epoch, train_x, train_y, batch_size)
        if val_x is not None:
            val_loss_epoch,val_acc_epoch = _test_epoch(True, model, epoch, val_x, val_y, batch_size)
        
        train_loss_list.append(train_loss_epoch)
        train_acc_list.append(train_acc_epoch)
        if val_x is not None:
            val_loss_list.append(val_loss_epoch)
            val_acc_list.append(val_acc_epoch)
            
        if epoch%5==0:
            state = model.state_dict()
            torch.save(state, './model/{}-model-epoch-{}.state'.format(save_prefix,epoch))
                
    images_prefix_name = "{}-model-epoch-{}".format(save_prefix,epochs)
    
    plot_train_curve(epochs,train_loss_list,train_acc_list,val_loss_list,val_acc_list,images_prefix_name)
    
    
        
def test(model, test_x, test_y, batch_size):
    """
        在测试集上进行效果测试
        Parmeters:
        ----------------
            model: 待测试模型
            test_x: 测试集数据，Tensor
            test_y: 测试集标签，Tensor
            batch_size: 批处理大小
        
    """
    _test_epoch(False, model, 1, test_x, test_y, batch_size)
    
def predict(model, x, batch_size,proba=False):
    """
        获取预测类别
        Parameters:
        ----------------
            model: 进行预测的模型
            x: 要进行预测的数据向量，Tensor
            batch_size: 批处理大小,Int
            proba: 输出类型，True表示输出为1的概率，False表示输出类别，默认为False
        
        Return:
        -----------------
            result: 预测的label值，list
    """
    
    model.eval()
    result = []
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        output = model(data)
        pred = output.data.max(1)[1]

        result += [i for i in pred.cpu().numpy()]
        
    return result

def predict_proba(model, x, batch_size,proba=False):
    """
        预测为目标类别的概率
        Parameters:
        ----------------
            model: 进行预测的模型
            x: 要进行预测的数据向量，Tensor
            batch_size: 批处理大小,Int
            proba: 输出类型，True表示输出为1的概率，False表示输出类别，默认为False
        
        Return:
        -----------------
            result: 预测的label值，list
    """
    
    model.eval()
    result = []
    for index in range(0, len(x), batch_size):
        data = x[index : index + batch_size]
        output = model(data)
        pred = output.data.max(1)[0]
        result += [i for i in pred.cpu().numpy()]
        
    return result

def compare_model(model_list,test_data,label,save_name="model_compare_report"):
    """
        比较多各模型的效果，评价指标包括TP、FP、TN、FN、Precision、Recall、AUC等
        Parmeters:
        -----------------
            model_list: 要进行比较的模型列表
            test_data: Tensor要进行测试的数据的向量
            label: Tensor,测试数据的标签值
            
        Return:
        -----------------
            df: Dataframe,模型效果对比表
        
    """
    batch_size = 128
    columns = ['model','tp','fp','tn','fn','precision','recall','auc']
    result = []
    label = label.cpu().numpy()
    for model in model_list:
        proba = predict(model,test_data,batch_size,proba=True)
        pred = list(map(lambda x:1 if x>0.5 else 0,proba))
        model_name = model.__class__.__name__
        matrix = confusion_matrix(label,pred)
        tp = matrix[1][1]
        fp = matrix[1][0]
        tn = matrix[0][0]
        fn = matrix[0][1]
        precision = precision_score(label,pred,1)
        recall = recall_score(label,pred,1)
        auc = roc_auc_score(label,proba)      
        
        result.append([model_name,tp,fp,tn,fn,precision,recall,auc])
    df = pd.DataFrame(result,columns=columns)

    if not os.path.exists("./report/"):
        os.makedirs("./report/")
    df.to_csv("./report/{}.csv".format(save_name),index=False)

    return df


def compare_diff_epoch(model,test_data,label,epochs,name_prefix=""):
    """
        比较单个模型在不同训练轮数在测试集上的表现，最终输出总轮数的五个分为点时模型在测试集上的表现
        Parmeters:
        -----------------
            model_list: 要进行比较的模型列表
            test_data: Tensor要进行测试的数据的向量
            label: Tensor,测试数据的标签值
            epochs: 训练的总轮数
            name_prefix: 加载模型的前缀名，默认为为模型名，最终输出格式为./model/{name_prefix}-model-epoch-{epoch}.state
            
        Return:
        -----------------
            df: Dataframe,模型五个分为点时模型在测试集上的效果对比表
        
    """
    if name_prefix=="":
        name_prefix = model.__class__.__name__
    
    
    batch_size = 128
    epoch_list = [int(epochs/5*i) for i in range(1,6)]
    columns = ['model','epoch','tp','fp','tn','fn','precision','recall','auc']
    result = []
    label = label.cpu().numpy()
    
    
    for epoch in epoch_list:
        model.load_state_dict(torch.load("./model/{}-model-epoch-{}.state".format(name_prefix,epoch)))
        proba = predict(model,test_data,batch_size,proba=True)
        pred = list(map(lambda x:1 if x>0.5 else 0,proba))
        model_name = model.__class__.__name__
        matrix = confusion_matrix(label,pred)
        tp = matrix[1][1]
        fp = matrix[1][0]
        tn = matrix[0][0]
        fn = matrix[0][1]
        precision = precision_score(label,pred,1)
        recall = recall_score(label,pred,1)
        auc = roc_auc_score(label,proba)      
        
        result.append([model_name,epoch,tp,fp,tn,fn,precision,recall,auc])
    df = pd.DataFrame(result,columns=columns)

    if not os.path.exists("./report/"):
        os.makedirs("./report/")
    df.to_csv("./report/{}_compare_diff_epoch_report.csv".format(name_prefix),index=False)

    return df