#========================================
#          使用pickle进行读写
#========================================

def writebunchobj(path, bunchobj):
    """
        将对象进行持久化，常用于将比较耗时的操作存储执行结果，下次直接进行调用即可，例如：向量化特征提取
    """
    with open(path, "wb") as file_obj:
        pickle.dump(bunchobj, file_obj)

def readbunchobj(path):
    """
        读取持久化的pickle文件，和writebunchobj对应
    """
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch

#========================================
#              异常检测
#========================================

def nsigma_threehold(input_data,n=3):
    """
        获取3Sigma法进行异常检测的阈值
        Parameters:
        -----------------
            input_data: 输入数据，series、list
            n: n sigma中的n,默认为3，int
        Return:
        -----------------
            lower_threehold: 正常数据下阈值
            upper_threehold: 正常数据上阈值
    """
    mean = input_data.mean()
    std = input_data.std()
    lower_threehold = mean-n*std
    upper_threehold = mean+n*std
    print("normal range:{} ~ {}".format(lower_threehold,upper_threehold))
    return lower_threehold,upper_threehold
    
def box_threehold(input_data):
    """
        获取箱型图法进行异常检测的阈值
        Parameters:
        -----------------
            input_data: 输入数据，series
        Return:
        -----------------
            lower_threehold: 正常数据下阈值
            upper_threehold: 正常数据上阈值
    """
    statistics = input_data.describe() #保存基本统计量
    IQR = statistics.loc['75%']-statistics.loc['25%']   #四分位数间距
    QL = statistics.loc['25%']  #下四分位数
    QU = statistics.loc['75%']  #上四分位数
    lower_threehold = QL - 1.5 * IQR #下阈值
    upper_threehold = QU + 1.5 * IQR #上阈值
    print("normal range:{} ~ {}".format(lower_threehold,upper_threehold))
    return lower_threehold,upper_threehold


#========================================
#              字符串映射
#========================================

def string_to_index(str_list,max_len):
    """
        将字符串映射为自然数列表，
            - 只对能够打印的字符串进行映射
            - 从1开始映射，0保留给补位
        Parmaeters:
        -------------------
            str_list: 2-dim array/list,要进行映射的字符串列表
            max_len: 映射字符串截取长度
            
        Return:
        -------------------
            output: 2-dim list,自然数列表
        
        Example:
        ------------------
            >> string_to_ascii(df_train['domain'].values,100)
    """
    import string
    
    chars = string.printable  # 获取能够打印的字符串
    
    char_to_idx = {ch : idx + 1 for idx, ch in enumerate(chars)}
    idx_to_char = {idx + 1 : ch for idx, ch in enumerate(chars)}
    
    output = []
    
    for idx, row in enumerate(str_list):
        #当domain长度超过max_len时，从一级、二级域名开始计算只保留max_Len位
        if(len(row) > max_len):
            row = row[len(row) - max_len:]
        
        tmp = [char_to_idx[ch] for ch in row]
        
        if(len(tmp) < max_len):
            # 当域名长度不够max_len时，映射为自然数后在后面补零
            tmp = tmp + [0 for i in range(max_len - len(tmp))]
        
        output.append(tmp)
    return output

