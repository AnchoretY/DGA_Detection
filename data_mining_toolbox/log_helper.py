#coding=utf-8
import os
import logging

def get_logger(log_filename):
    """
        快速获得logger,INFO以上输出到控制台，Debug以上输出到日志文件
        Parameters: 
        --------------
            log_file: log_file存储的文件名，后缀名为.log
        Return 
        --------------
            logger: logger对象，使用这个对象可以直接进行上面指定的日志管理
    """
    if log_filename.split(".")[-1]!="log":
        raise ValueError("log_filename must end with .log")
    
    
    logger = logging.getLogger()
    logger.setLevel('DEBUG')
    
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    
    chlr = logging.StreamHandler() # 输出到控制台的handler
    chlr.setFormatter(formatter)
    chlr.setLevel('INFO')  # 也可以不设置，不设置就默认用logger的level
 
    if not os.path.exists("./log/"):
        os.makedirs("./log/")
    log_file = "./log/"+log_filename
    
    fhlr = logging.FileHandler(log_file,encoding='utf-8') # 输出到文件的handler
    fhlr.setFormatter(formatter)
    
    logger.addHandler(chlr)
    logger.addHandler(fhlr)
    
    return logging.getLogger()