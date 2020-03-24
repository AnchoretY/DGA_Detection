import numpy as np
import matplotlib.pyplot as plt


def plot_curve(x,y,label,title,xlabel,ylabel,figsize=(8,6),ylim=None,\
    grid=True,title_size=20,xylabel_size=15,legend_size=12,save_name=""):
    """
        画折线图
        Parameters:
        ----------
            x: 横坐标值,两种可选形式,:
                optional 画单条折线横坐标值,1-dim list,例如: [100,200,300]
                optional 画多条折线横坐标值,2-dim list,例如:[[100,200,300],[110,150,234]]
            y: 纵坐标值
                optional 画单条折线纵坐标值,1-dim list,例如: [100,200,300]
                optional 画多条折线纵坐标值,2-dim list,例如:[[100,200,300],[110,150,234]]
            title: 标题，字符串
            xlabel: 横轴表示变量名称
            ylabel: 纵轴表示变量名称
            label: 标签值
                optional 单条折线的图例值,String,例如:"title1"
                optional 多条折线的图例值,list,例如:["title1","title2"]
            figsize: 折线图大小，默认为（8，6）
            ylim: 定义纵坐标最大最小值,(ymin,ymax)
            grid: 是否使用网格,默认为True
            title_size: 标题字号
            xylabel_size: 横纵坐标变量名称字号
            legend_size: 图例字号
            save_name: 图片存储名称，默认不进行存储
            
        Example:
        --------
            #画单条曲线
            >>> x = [100,200,300]
            >>> y = [0.6,0.7,0.9]
            >>> title = "Precision Curve"
            >>> label = "label1"
            >>> x_label = "epoch"
            >>> y_label = "precision"
            >>> plt_learn_curve(x,y,title,x_label,y_label,label)
            
            # 画多条曲线
            >>> x = [[100,200,300],[100,200,300]]
            >>> y = [[0.6,0.7,0.9],[0.3,0.6,0.7]]
            >>> title = "Precision Curve"
            >>> label = ["label1","label2"]
            >>> x_label = "epoch"
            >>> y_label = "precision"
            >>> plt_learn_curve(x,y,title,x_label,y_label,label)
        
    """
    plt.figure(figsize=figsize)
    
    # 标题
    font_title = {
        'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : title_size,
    }
    plt.title(title,font_title)
    
    # 坐标轴变量名称
    font_xylabel = {
        'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : xylabel_size,
    }
    plt.xlabel(xlabel,font_xylabel)
    plt.ylabel(ylabel,font_xylabel)
    
    # 纵坐标标度
    if ylim is not None:
        plt.ylim(ylim)
    
    # 是否使用网格
    if grid==True:
        plt.grid()
    
    # 画折线
    line_style = ["ro-","go-","ro--","go--","ro-.","go-.","ro:","go:"]
    dim = np.array(y).ndim
    if dim==1:
        plt.plot(x,y, 'o-', color="r",
                         label=label)
    if dim==2:
        for i,data in enumerate(y):
            plt.plot(x,data, line_style[i], 
                         label=label[i])
            
            
    # 图例设置
    font_legend = {
        'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : legend_size,
    }
    plt.legend(loc="best",prop=font_legend)
    import os


    if save_name!="":
        if not os.path.exists("./images/"):
            os.makedirs("./images/")
        plt.savefig("./images/{}".format(save_name))
        
        
def plot_train_curve(epochs,train_loss_list,train_acc_list,val_loss_list=None,val_acc_list=None,save_prefix='model'):
    """
        画训练过程中损失、准确率变化图
        Parameter:
        -------------
            epochs: int,训练轮数
            train_loss_list: list,训练集各轮数损失函数值
            train_acc_list: list,训练集各轮数准确率值
            val_loss_list: list,测试集各轮数损失函数值
            val_acc_list: list,测试集各轮数准确率值
            save_prefix: string,图片存储名称前缀，默认不进行存储
    """
    
    if val_loss_list is not None:
        plot_curve(range(1,epochs + 1),[train_loss_list,val_loss_list],["train","val"],\
                   "Loss Curve","epoch","Loss",save_name="{}-loss".format(save_prefix))
        plot_curve(range(1,epochs + 1),[train_acc_list,val_acc_list],["train","val"],\
                   "Acc Curve","epoch","Acc",save_name="{}-loss".format(save_prefix))
    else:
        plot_curve(range(1,epochs + 1),train_loss_list,"train","Loss Curve","epoch","Loss",\
                   save_name="{}-loss".format(save_prefix))
        plot_curve(range(1,epochs + 1),train_acc_list,"train","Acc Curve","epoch","Acc",\
                   save_name="{}-loss".format(save_prefix))




def plot_distribution(input_data,data_name):
    """
        画数据分布图
        Parameters:
        --------------
            data: 输入数据，series、list
            data_name: 输入数据的名称,string
    """
    sns.set_style("white")
    sns.set_color_codes(palette='deep')
    f, ax = plt.subplots(figsize=(8, 7))
    #Check the new distribution 
    sns.distplot(input_data, color="b");
    ax.xaxis.grid(False)
    ax.set(ylabel="Frequency")
    ax.set(xlabel=data_name)
    ax.set(title="{} distribution".format(data_name))
    sns.despine(trim=True, left=True)
    plt.show()