# -*- coding: utf-8 -*-
"""
2020.07.19 19:50
在降水订正项目中需要用到的所有函数
@author: fzl
"""
#%%
'''
函数介绍:
Part1: 基本的气象数据的读取
    read_micaps_data() 读取一般的micaps数据，每行为一个子list 
    get_station_data() 获取常规站点观测数据，包括地面填图(plot) 和 'r6-p' 
    get_jiami_obs() 读取逐小时的观测资料
    get_EC_thin_data()  获取EC_thin的数据(不包括 EC_thin/physic底下的物理量)，默认EC_thin的数据是等经纬网格的;
    get_EC_thin_physic_data() 获取EC_thin/physic路径下的物理量
    
Part2: 站点和规则格点数据的可视化
    都是基于 basemap 画的，Linux下需要更改为cartopy
    scatter_station_on_map() 数据站点经纬度，根据站点要素值大小，scatter到地图底图上
    contourf_on_map()  传入数据和对应的经纬度范围限制，将数据叠加在地图底图上
    contourf_data_on_map() 传入数据和对应的经纬范围，将数据叠加在地图底图上
    
Part3: 插值函数 站点 <---> 规则格点 
    interp2d_station_to_grid()  将站点数据插值到等经纬度格点
    get_nearest_point_index()  获取与给定经纬度值的点最近的等经纬度格点的经纬度index
    grid_interp_to_station()   将等经纬度网格值 插值到 离散站点。使用griddata进行插值
    
Part4: 本地时 <--> EC和SMS预报时刻的对应, 即获取与站点观测时刻一致的 EC 和 SMS 的预报资料的 时刻戳
    surface_time2_EC_BJ_time()    
    surface_time2_SMS_time()
    
Part5: 获取不同数据源的对应时刻的气象特征 
    get_all_surface_station_Dataset() 构建r6-p(6小时累计降水)surface站点数据集
    get_T0_jiami_surface_station_Dataset()  构建加密观测的数据集(1小时分辨率)
    get_T3_jiami_surface_station_Dataset()  构建加密观测的3小时累计降水变量
    get_all_ECthin_Station_dataset_ori()    根据surface_file的站点数据，获取对应的时刻的 EC细网格物理量资料，并将网格资料插值到站点
    get_all_ECthin_Station_dataset_dst()   将get_all_ECthin_Station_dataset_ori函数的输出做进一步特征组合。
    get_T0_SMS_Station_dataset()      获取与surface_file同时刻的 SMS(华东区域中心的)资料并将其插值到站点上   
    get_T3_SMS_Station_dataset()      获取与surface_file同时刻的 SMS(华东区域中心的)资料 + 累计3/2/1小时降水 并将其插值到站点上
    

Part6: 气象上常见评分函数 
    prep_clf()     计算二分类结果-混淆矩阵的四个元素
    precision()     计算精确度precision: TP / (TP + FP)
    recall()       计算召回率recall: TP / (TP + FN)
    ACC()          计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    FSC()        计算f1 score = 2 * ((precision * recall) / (precision + recall))
    TS()         计算TS评分: TS = hits/(hits + falsealarms + misses)  
    multil_scores()  输入某个阈值下，输出该阈值下预报相对观测的评分（以直接打印的方式） 
    plot_multi_scores() 即画出不同阈值情况下的 模型预测对比实况的 Acc/Recall/Precision/F1/TS/MAR/FAR 的评分情况


Part7: 数据分析 
    drop_outlier() 处理离群值


'''
#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from mpl_toolkits.basemap import Basemap
import numpy.ma as ma
import datetime
import time
import netCDF4 as nc
import h5py
import matplotlib
import scipy 
from scipy.interpolate import griddata
# import cartopy

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


#%%
#########################Part1: 基本数据的读取 #################################

def read_micaps_data(filename):
    '''
    func: 读micaps类型的数据，将其变为list 
    input:
        filename: 文件名称。eg: 'surface/r6-p/18080414.000'
    return:
        data：由多个List组成，其中每行为一个子list。 
    '''
    f=open(filename,mode='r')
    
    #此时每行的都为 字符格式
    str_data = f.readlines()  
    
    data = []
    
    for i in range(len(str_data)):
    
    #将每行的字符类型的数字转为float型,如果某一行存在非数字字符，则跳过该行
        try:
            #依次读取每一行数据
            line_data = str_data[i]
            
            #如果该行为空(只有换行符)，则跳过；否则，将其转换为数值型
            if len(line_data)>1:
                #去除换行符并将每个字符数据分开
                line_data = line_data.strip().split()
                
                #将数据类型由str转换为float型
                line_data=[float(line_data[i]) for i in range(len(line_data))]
#                line_data = np.array(line_data).reshape(1,-1)
                
                data.append(line_data)
        except Exception as e:  
            print('第{:}行含有非数字字符 '.format(i))
            print(e)
            print()
            pass
    
    return data


###############################################################################
def get_station_data(filename,file_type = 'r6-p',loc_range = [18,54,73,135]):
    
    '''
    func: 获取常规站点观测数据，
    inputs:
        filename: 文件名
        file_type : 文件类型。这里支持两种：1. 'plot'。即站点地面填图；2.'r6-p' 。站点的累计6小时降水量
            1. plot
            surface/plot类型的数据存储方式为：
            第0行：数据的时间信息
            之后，每一个站点的信息占据两行：包括 站台号 经纬度 观测要素 等信息;
            因此后续操作需要把 每个站台的信息 只用一行表示
            [0,1,2,6,7,16,19]列分别表示为[站台号,经度,纬度,风向,风速,露点,温度]
            2. r6-p
            surface/r6-p （6小时累计降水量）类型的数据存储方式为：
            前12行都为不重要的信息
            从13行开始，每一行包括 [站台号, 经度，纬度，海拔高度，降水量]

    return:
        station_data 每一行都是一个站点的信息,为 array数组。
         eg:station_data数组shape = 100*20, 即共计100个站点观测，每个站点观测有20个要素(包括经纬度信息等)
        
    ''' 
    
    f=open(filename,mode='r')
    
    #此时每行的都为 字符格式
    str_data = f.readlines()  
    
    data = []
    
    for i in range(len(str_data)):
    
    #将每行的字符类型的数字转为float型,如果某一行存在非数字字符，则跳过该行
        try:
            #依次读取每一行数据
            line_data = str_data[i]
            
            #如果该行为空(只有换行符)，则跳过；否则，将其转换为数值型
            if len(line_data)>1:
                #去除换行符并将每个字符数据分开
                line_data = line_data.strip().split()
                
                #将数据类型由str转换为float型
                line_data=[float(line_data[i]) for i in range(len(line_data))]
    #            line_data = np.array(line_data).reshape(1,-1)
                
                data.append(line_data)
        except Exception as e:  
            print('第{:}行含有非数字字符 '.format(i))
            print(e)
            print()
            pass
    
    if file_type == 'plot':
        
        '''
        surface/plot类型的数据存储方式为：
        第0行：数据的时间信息
        之后，每一个站点的信息占据两行：报考 站台号 经纬度 观测要素 等信息;
        因此后续操作需要把 每个站台的信息 只用一行表示
        '''
        
        #第一行表示时间信息
#        T = data[0]

        station_data = []
        
        for i in range(len(data[1::2])):
            
            station_data.append(data[i*2+1]+data[i*2+2])
        
        station_data = [np.array(station_data[i]).reshape(1,-1) for i in range(len(station_data))] 
        station_data = np.concatenate(station_data,axis=0)
           
        #[0,1,2,6,7,16,19]列分别表示为[站台号,经度,纬度,风向,风速,露点,温度]
        return station_data
        
    elif file_type == 'r6-p':
        '''
        surface/r6-p （6小时累计降水量）类型的数据存储方式为：
        前12行都为不重要的信息
        从13行开始，每一行包括 [站台号, 经度，纬度，海拔高度，降水量]
        '''
        station_data = data[13:]
        station_data = [np.array(station_data[i]).reshape(1,-1) for i in range(len(station_data))]  
        station_data = np.concatenate(station_data,axis=0)
        
    else: 
        print('error!')
        print("Please check you file_type, it must be 'plot' or 'r6-p'!")
        
        station_data =  None

    return station_data


###############################################################################
def get_jiami_obs(abs_file, filetype = 'pd', sort = True):
    '''
    func: 读取逐小时的观测资料
    inputs: 
        abs_file: 加密观测文件的绝对路径；
                eg: 'D:/ori_data/aws_jiami/2018080420.txt' 
        
        filetype: 数据读取成功后返回的数据类型，
                'pd': default, 即pandas类型
                'array': 数组类型
        sort: 是否依据站点号大小对数据进行排序，默认True
                
    return:
        返回气象要素，其中每行为一个站点观测，列为不同要素. 每个要素为 np.float类型 
        依次[0,1,2,3,4,5,
           6,7,8,9,10]依次表示如下要素
        ['站号', '气温', '最高气温', '最低气温', '露点温度', '相对湿度', 
         '小时降水量', 'C2分钟风向', 'C2分钟平均风速', '最大风速的风向', '最大风速']
        
    '''
    #由于abs_file里含有中文，不同平台的默认编码方式不同，可能会出错
    try:
        f = open(abs_file,'r')
    except Exception as e:
        f = open(abs_file,'r',encoding = 'GBK')
        
    #读取所有行
    str_data = f.readlines()  
    f.close()
    
    #去掉换行符，并将字符串分开
    data = [line_data.strip().split(',') for line_data in str_data]
    
    
    #第一行为要素说明
    columns = data[0]
    
    #保留以5开头的站点号对应的观测
    # station_data = [line_data for line_data in data if line_data[0][0] == '5']
    
    #获取所有站点
    station_data = data[1:]  
    
    #将[2:]列的所有要素转为 np.float类型，并将空测用np.nan表示
    valid_station_data = np.zeros(shape = (len(station_data),len(columns[2:])))
    all_stations = []
    all_times = []
    for index,line_data in enumerate(station_data[0:]):
        all_stations.append(str(line_data[0]))
        all_times.append(line_data[1])
        np_line_data = [np.float(var) if len(var) > 0 else np.nan for var in line_data[2:]]
        valid_station_data[index,:] = np_line_data[0:]
      
    #去掉时间列
    pd_data = pd.DataFrame(columns = columns).drop(columns = '时间')
    pd_data['站号'] = all_stations
    # print(valid_station_data.shape)
    # print(len(columns[2:]))
    # print(len(all_stations))
    for i, column in enumerate(columns[2:]):
        pd_data[column] = valid_station_data[:,i]
    
    if sort:
        pd_data = pd_data.sort_values('站号', ascending=bool)
        pd_data.index = range(len(pd_data))
    
    return pd_data if filetype == 'pd' else pd_data.values


###############################################################################
def get_EC_thin_data(filename,plot = True,label_gap = 2):
    '''
    func:获取EC_thin的数据(不包括 EC_thin/physic底下的物理量)，默认EC_thin的数据是 等经纬网格的;
    doc: 空间分辨率为 0.125*0.125 或者 0.25*0.25 ; 时间分辨率为3小时
    input:
        filename: 文件名
        plot: 默认True，绘制数据场
        label_gap: Plot中，x和y label的坐标经纬度间隔
    
    return:
        lon_grid : 场对应的经度信息
        lat_grid : 场对应的纬度信息
        rain_grid：场数值 
        以list形式返回
        [lon_grid,lat_grid,rain_grid]
        
    '''
    
    f=open(filename,mode='r')
    
    #此时每行的都为 字符格式
    str_data = f.readlines()  
    
    data = []
    
    for i in range(len(str_data)):
        
        #将每行的字符类型的数字转为float型,如果某一行存在非数字字符，则跳过该行
        try:
            #依次读取每一行数据
            line_data = str_data[i]
            
            #如果该行为空(只有换行符)，则跳过；否则，将其转换为数值型
            if len(line_data)>1:
                #去除换行符并将每个字符数据分开
                line_data = line_data.strip().split()
                
                #将数据类型由str转换为float型
                line_data=[float(line_data[i]) for i in range(len(line_data))]
                line_data = np.array(line_data).reshape(1,-1)
                
                data.append(line_data)
        except Exception as e:  
            print('第{:}行含有非数字字符 '.format(i))
            print(e)
            print()
            pass
        
    #第0行为时间信息
    #第1行为 经纬度网格信息
#    T = data[0]
    loc_info = data[1]
    
    det_lat = abs(loc_info[0,0])
    det_lon = abs(loc_info[0,1])
    lon_max = loc_info[0,3]
    lon_min = loc_info[0,2]
    lat_max = loc_info[0,4]
    lat_min = loc_info[0,5]
    
    lat_range = np.arange(lat_min,lat_max+det_lat,det_lat) 
    lon_range = np.arange(lon_min,lon_max+det_lon,det_lon)
    
    lon_grid,lat_grid = np.meshgrid(lon_range,lat_range)
    
    lat_grid = lat_grid[-1::-1,:]
    
    #获取等经纬度降水数据，并可视化；
    #各点降水数据从第二行开始
    tp = np.concatenate(data[2:],axis = 0)

    #将降水场可视化出来
    if plot: 
        
        plt.figure(figsize = (16,10))
        
        cs = plt.contourf(tp[-1::-1], 
#                          levels= np.arange(0,32+1,4), 
                           extend='both',
                           cmap = plt.cm.rainbow) #extend参数使得colorbar两端变尖
        #cs.cmap.set_over('red')
        #cs.cmap.set_under('blue')
        #cs.changed()
        cb = plt.colorbar(cs)
        
        cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        font = {'family' : 'serif',
                'color'  : 'darkred',
                'weight' : 'normal',
                'size'   : 20,
                }
        
        cb.set_label('',fontdict=font) #设置colorbar的标签字体及其大小
        
        #确定横纵坐标轴的gap
        label_gap = label_gap
        
        plt.xticks(np.arange(0,len(lon_range),label_gap/det_lon), np.arange(lon_min,lon_max+1,label_gap,dtype = int),fontsize = 16)
        plt.yticks(np.arange(0,len(lat_range),label_gap/det_lat), np.arange(lat_min,lat_max+1,label_gap,dtype = int),fontsize = 16)
        
        plt.xlabel("longitude°E",fontsize = 20)
        plt.ylabel("latitude°N",fontsize = 20)
        plt.title(filename,fontsize = 20)

    rain_info = [lon_grid,lat_grid,tp]
    
    return rain_info


###############################################################################
def get_EC_thin_physic_data(filename,plot = True,label_gap = 2):
    '''
    func:获取EC_thin/physic路径下的物理量，默认EC_thin的数据是 等经纬网格的;
    doc: 空间分辨率为 0.125*0.125 或者 0.25*0.25 ; 时间分辨率为3小时
    
    physic类型的数据存储形式为：
    第0行为时间信息
    第1行为 经纬度网格信息,包括场数据的shape,
    从第2行开始，现实中的每一行数据（纬向方向）由 9行组成；
    #eg: shape为 81*81，每一行数据有81个数据；而在存储时，每10个数据占据一行，
    # 剩下1个数据独占一行，因此 9行对应1行；

    input:
        filename: 文件名
        plot: 默认True，绘制物理量场
        label_gap: Plot中，x和y label的坐标经纬度间隔
    
    return:
        lon_grid : 场对应的经度信息
        lat_grid : 场对应的纬度信息
        rain_grid：场数值 
        以list形式返回
        [lon_grid,lat_grid,rain_grid]
        
    '''
    
    f=open(filename,mode='r')
    
    #此时每行的都为 字符格式
    str_data = f.readlines()  
    
    data = []
    
    for i in range(len(str_data)):
        
        #将每行的字符类型的数字转为float型,如果某一行存在非数字字符，则跳过该行
        try:
            #依次读取每一行数据
            line_data = str_data[i]
            
            #如果该行为空(只有换行符)，则跳过；否则，将其转换为数值型
            if len(line_data)>1:
                #去除换行符并将每个字符数据分开
                line_data = line_data.strip().split()
                
                #将数据类型由str转换为float型
                line_data=[float(line_data[i]) for i in range(len(line_data))]
                line_data = np.array(line_data).reshape(1,-1)
                
                data.append(line_data)
        except Exception as e:  
            print('第{:}行含有非数字字符 '.format(i))
            print(e)
            print()
            pass
        
    #第0行为时间信息
    #第1行为 经纬度网格信息
#    T = data[0]
    loc_info = data[1]
    
    
    #确定loc_info第一个元素是否为 det_lat
    #因为发现EC_thin/physic/pw的一个元素不是det_lat,是第二个元素开始的
    
    #先默认det_lat为第1个元素，一般det_lat<1; 如果第一个元素>1，则index = index+1
    index = 0
    if abs(loc_info[0,index]) > 1:
        index = index+1
        
    det_lat = abs(loc_info[0,index+0])
    det_lon = abs(loc_info[0,index+1])
    lon_max = loc_info[0,index+3]
    lon_min = loc_info[0,index+2]
    lat_max = loc_info[0,index+4]
    lat_min = loc_info[0,index+5]
    
    lat_range = np.arange(lat_min,lat_max+det_lat,det_lat) 
    lon_range = np.arange(lon_min,lon_max+det_lon,det_lon)
    
    lon_grid,lat_grid = np.meshgrid(lon_range,lat_range)
    
    lat_grid = lat_grid[-1::-1,:]
    
    #获取等经纬度场数据，并可视化；
    #从第index+2行开始
    
    length = data[index+2].shape[1]
    
    #判断每一行存储的数据个数（一般是 10）,
    #而一般的场的shape是 81*81，因此需要 int(np.ceil(81/10)) = 9 行
    #来表示真实场的一行数据
    k = int(np.ceil(lat_grid.shape[1]/length))
    
    tp = []
    
    for i in range(lat_grid.shape[0]):
        tp_line = []
        
        for j in range(k):
            tp_line = tp_line+list(data[i*k+2+index+j].ravel())
            
        tp_line = np.array(tp_line).reshape(1,-1)
        tp.append(tp_line)
    
    tp = np.concatenate(tp,axis = 0)
        
    
    #将降水场可视化出来
    if plot: 
        
        plt.figure(figsize = (16,10))
        
        cs = plt.contourf(tp[-1::-1], 
#                          levels= np.arange(0,32+1,4), 
                           extend='both',
                           cmap = plt.cm.rainbow) #extend参数使得colorbar两端变尖
        #cs.cmap.set_over('red')
        #cs.cmap.set_under('blue')
        #cs.changed()
        cb = plt.colorbar(cs)
        
        cb.ax.tick_params(labelsize=16)  #设置色标刻度字体大小。
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        font = {'family' : 'serif',
                'color'  : 'darkred',
                'weight' : 'normal',
                'size'   : 20,
                }
        
        cb.set_label('',fontdict=font) #设置colorbar的标签字体及其大小
        
        #确定横纵坐标轴的gap
        label_gap = label_gap
        
        plt.xticks(np.arange(0,len(lon_range),label_gap/det_lon), np.arange(lon_min,lon_max+1,label_gap,dtype = int),fontsize = 16)
        plt.yticks(np.arange(0,len(lat_range),label_gap/det_lat), np.arange(lat_min,lat_max+1,label_gap,dtype = int),fontsize = 16)
        
        plt.xlabel("longitude°E",fontsize = 20)
        plt.ylabel("latitude°N",fontsize = 20)
        plt.title(filename,fontsize = 20)

    physic_info = [lon_grid,lat_grid,tp]
    
    return physic_info    

#%%
######################### Part2: 数据的可视化：scatter 和 contourf #################  
    
def scatter_station_on_map(station_lon,station_lat,station_value, fill_value=9999,
                           loc_range = [18,54,73,135], det = 5, title = None,
                           save_path = None,
                           size_value_change = False,
                           if_norm = False,
                           norm_range = [0,50]
                           ):
    '''
    func: 将站点位置，根据站点要素值大小，scatter到地图底图上
    inputs：
        station_lon: 站点的经度
        station_lat: 站点的纬度
        station_value: 对应站点的要素值 
        上述三个数据类型 可以是 数组，也可以是 列表
        fill_value : 缺测填充值.默认为 9999; 
        loc_range: scatter区域范围。默认范围为中国大陆区域
                    [18,54,73,135] = [lat_min,lat_max,lon_min,lon_max]
        det: 横轴ticklabel的间隔
        title: 图题，默认None
        save_path: 图片的保存路径
        size_value_change: 散点大小是否随之数值改变
        if_norm: bool
            是否限制数值范围，default：False
        norm_range: int of list
            当if_norm为True时才有效，表示数值限制范围,默认[0,50]
    '''
    #step1: 先将station数据转换一下格式
    station_lon = np.array(station_lon).reshape(-1,1).ravel()
    station_lat = np.array(station_lat).reshape(-1,1).ravel()
    
    
    #使用mask数组；将value=fill_value的值跳过
    station_value = np.array(station_value).reshape(-1,1).ravel()
    mask = station_value == fill_value
    station_value = ma.array(station_value,mask=mask)
    
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]
    
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    
    m=Basemap(projection='cyl',llcrnrlat=lat_min,llcrnrlon=lon_min,
              urcrnrlat=lat_max,urcrnrlon=lon_max,resolution='l',ax=ax1)
    
    m.readshapefile('D:/zhongqi/geo_data/gadm36_CHN_shp/gadm36_CHN_1','states',
                    drawbounds=True)
    
    norm = matplotlib.colors.Normalize(norm_range[0],norm_range[1]) if if_norm else None
    
    if size_value_change:
        h = m.scatter(station_lon,station_lat,
                          # s = 15,
                       s = station_value, #marker大小与数值相关
                      c = station_value, #颜色深浅与数值大小相关
                      cmap = plt.cm.rainbow,
                      norm = norm,
                      )
    
    else:    
        h = m.scatter(station_lon,station_lat,
                  s = 15,
              # s = station_value, #marker大小与数值相关
              c = station_value, #颜色深浅与数值大小相关
              cmap = plt.cm.rainbow,
              norm = norm
              )
    
    cb = m.colorbar(h,size = '4%')
    cb.set_label('',fontsize = 20)
    cb.ax.tick_params(labelsize=16)
#        
    x_grid = np.arange(lon_min,lon_max+1,det)
    y_grid = np.arange(lat_min,lat_max+1,det)
    m.drawparallels(x_grid)
    m.drawmeridians(y_grid)
    plt.grid()
    plt.xticks(x_grid,x_grid,fontsize = 16)
    plt.yticks(y_grid,y_grid,fontsize = 16)
    plt.xlabel('longitude: °E',fontsize = 16)
    plt.ylabel('latitude: °N',fontsize = 16)
    
    if title:
        plt.title(title,fontsize = 20)
    
    #画出海岸线和国境线
    m.drawcoastlines()
    m.drawcounties()
    m.drawcountries()
    
    if save_path != None:
        plt.savefig(save_path, dpi = 200, bbox_inches = 'tight')
    
    return None      

###############################################################################
def contourf_on_map(all_data,loc_range = [30,50,105,125],
                    det_grid = 0.125, method = 'cubic',
                    gap = 3,
                    levels = list(range(0,50,10)),
                    is_norm = False, 
                    vmin = 0, vmax = 100
                    ):
    '''
    func: 传入数据和对应的经纬度范围限制，将数据叠加在地图底图上
    inputs:
        all_data : [lon_grid,lat_grid,data],对应为[经度网格，纬度网格，对应网格数据]
                    三个的shape一样
        loc_range: loc_range 为列表,[lat_min,lat_max,lon_min,lon_max]
                   case1:如果设置的loc_range > all_data的经纬度范围，则默认不使用该参数；
                   case2:当loc_range 在all_data涵盖范围内，则只画出 loc_range范围内的数据
                       
        det_grid : 当loc_range为 case2时，需要对新的经纬度范围进行插值，空间分辨率；默认为 1
        method: 当loc_range为case2时，griddata插值方法。默认使用cubic
        gap:画图时，横纵坐标的经纬度间隔
        levels: int or list
            设置等值区域的个数和数值大小；如果为int,则默认画出levels个等值区域；
            如果为list，则根据List指定的数值大小划定等值区域
        is_norm: bool
            默认False, 表示是否限制数值范围（在colorbar上可以体现出来）
        vmin,vmax: int
            当is_norm = True时才起作用，确定显示的数值的范围
            
    '''
    #获取网格位置及其数据
    ori_lon_grid = all_data[0]
    ori_lat_grid = all_data[1]
    ori_data = all_data[2]
    
    #确定 loc_range范围
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]
    
#    #以原本数据的范围和loc_range的范围 取 交集, 只画出交集范围
#    lat_min = max(np.min(ori_lat_grid),loc_range[0])
#    lat_max = min(np.max(ori_lat_grid),loc_range[1])
#    lon_min = max(np.min(ori_lon_grid),loc_range[2])
#    lon_max = min(np.max(ori_lon_grid),loc_range[3])
    
    
    #如果目标区域在输入数据范围内，则进行插值再画图
    if (lat_max< np.max(ori_lat_grid) and lat_min> np.min(ori_lat_grid)) :
        
        if (lon_max< np.max(ori_lon_grid) and lon_min> np.min(ori_lon_grid)):
            

            x, y = np.meshgrid(np.arange(lon_min,lon_max+det_grid,det_grid),
                               np.arange(lat_min,lat_max+det_grid,det_grid))

            new_lat_grid = y
            new_lon_grid = x
            
            points = [ori_lon_grid.reshape(-1,1),ori_lat_grid.reshape(-1,1)]
            points = np.concatenate(points,axis = 1)
            
            values = ori_data.reshape(-1,1)
            
            new_data = griddata(points,values,(new_lon_grid,new_lat_grid),
                                method=method)
            new_data = new_data[:,:,0]
            

            contourf_data_on_map(new_data,new_lon_grid,new_lat_grid,gap = gap,
                                 levels = levels,
                                 is_norm = is_norm, vmin = vmin, vmax = vmax)
                    
    else:
        contourf_data_on_map(ori_data,ori_lon_grid,ori_lat_grid,gap = gap,
                             levels = levels,
                             is_norm = is_norm, vmin = vmin, vmax = vmax)
 
    return None
        
###############################################################################               
def contourf_data_on_map(data,lon_grid,lat_grid,gap = 5, 
                         levels = 10,
                         is_norm = False, vmin = 0, vmax = 100):
    '''
    func: 传入数据和对应的经纬范围，将数据叠加在地图底图上
    inputs:
        data: 网格数据
        lon_grid : 网格经度
        lat_grid ：网格纬度
        gap : 地图上横纵坐标显示的经纬度数值间隔。即tick_gap
    return 
        输出一张图
    levels: int or list
        设置等值区域的个数和数值大小；如果为int,则默认画出levels个等值区域；
        如果为list，则根据List指定的数值大小划定等值区域
    is_norm: bool
        默认False, 表示是否限制数值范围（在colorbar上可以体现出来）
    vmin,vmax: int
        当is_norm = True时才起作用，确定显示的数值的范围
    ----------
    return None
    '''
    
    lat_min = np.min(lat_grid)
    lat_max = np.max(lat_grid)
    lon_min = np.min(lon_grid)
    lon_max = np.max(lon_grid)
    
#    det_grid = lon_grid[0,1]-lon_grid[0,2]
    
    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])
    
    norm = matplotlib.colors.Normalize(vmin = vmin, vmax = vmax) if is_norm else None
    
    m=Basemap(projection='cyl',llcrnrlat=lat_min,llcrnrlon=lon_min,
              urcrnrlat=lat_max,urcrnrlon=lon_max,resolution='l',ax=ax1)
    
    m.readshapefile('D:/zhongqi/geo_data/gadm36_CHN_shp/gadm36_CHN_1','states',
                            drawbounds=True)
    
    #若lat_grid高纬度值在最上面，则将数据做对应行数反转;
    #因为使用m.contourf时，默认低纬在上
    if lat_grid[0,0] == lat_max:
        lat_grid = lat_grid[-1::-1]
        data = data[-1::-1]
    
    
    h = m.contourf(lon_grid,lat_grid,data,
                  levels= levels,
                   extend='both',
                   norm = norm,
                   cmap = plt.cm.rainbow)
    
    
    cb = m.colorbar(h,size = '4%')
    cb.set_label('',fontsize = 20)
    cb.ax.tick_params(labelsize=16)
    
    x_grid = np.arange(lon_min,lon_max+gap,gap,dtype = int)
    y_grid = np.arange(lat_min,lat_max+gap,gap,dtype = int)
    m.drawparallels(x_grid)
    m.drawmeridians(y_grid)
    plt.grid()
    plt.xticks(x_grid,x_grid,fontsize = 18)
    plt.yticks(y_grid,y_grid,fontsize = 18)
    plt.xlabel('longitude: °E',fontsize = 20)
    plt.ylabel('latitude: °N',fontsize = 20)
#    plt.title(filename,fontsize = 20)
    #        plt.title(u'中国站点分布',fontsize = 20)
    
    #画出海岸线和国境线
    m.drawcoastlines()
    m.drawcounties()
    m.drawcountries()
    
    plt.show()
    
    return None


#%%
######################################### Part3: 插值函数：站点 <----> 格点 ######################

def interp2d_station_to_grid(lon,lat,data,loc_range = [18,54,73,135],
                             det_grid = 1 ,method = 'cubic'):
    '''
    func : 将站点数据插值到等经纬度格点
    inputs:
        lon: 站点的经度
        lat: 站点的纬度
        data: 对应经纬度站点的 气象要素值
        loc_range: [lat_min,lat_max,lon_min,lon_max]。站点数据插值到loc_range这个范围
        det_grid: 插值形成的网格空间分辨率,默认 0.125
        method: 所选插值方法，默认'cubic'
    return:
        
        [lon_grid,lat_grid,data_grid]
    '''
    #step1: 先将 lon,lat,data转换成 n*1 的array数组
    lon = np.array(lon).reshape(-1,1)
    lat = np.array(lat).reshape(-1,1)
    data = np.array(data).reshape(-1,1)
    
    #shape = [n,2]
    points = np.concatenate([lon,lat],axis = 1)
    
    #step2:确定插值区域的经纬度网格
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]
    
    lon_grid, lat_grid = np.meshgrid(np.arange(lon_min,lon_max+det_grid,det_grid), 
                       np.arange(lat_min,lat_max+det_grid,det_grid))
    
#    lat_grid = lat_grid[-1::-1] #保证 在纬度网格中，从上到下，纬度减小
    
    #step3:进行网格插值
    grid_data = griddata(points,data,(lon_grid,lat_grid),method = method)
    grid_data = grid_data[:,:,0]
    
    #保证纬度从上到下是递减的
    if lat_grid[0,0]<lat_grid[1,0]:
        lat_grid = lat_grid[-1::-1]
        grid_data = grid_data[-1::-1]
    
    return [lon_grid,lat_grid,grid_data]

###############################################################################
def get_nearest_point_index(point_lon_lat,lon_grid,lat_grid):
    '''
    func:获取与给定经纬度值的点最近的等经纬度格点的经纬度index
    inputs:
        point_lon_lat: 给定点的经纬度，eg:[110.137,42.353]
        lon_grid: 经度网格
        lat_grid: 纬度网格
    return:
        index: [index_lat,index_lon]
    '''
    #step1: 获取网格空间分辨率;默认纬度和经度分辨率一致
    det = lon_grid[0,1]-lon_grid[0,0]
    
    #step2: 
    point_lon = point_lon_lat[0]
    point_lat = point_lon_lat[1]
    
    lon_min = np.min(lon_grid)
    lat_min = np.min(lat_grid)
#    lat_max = np.max(lat_grid)
    
    index_lat = round((point_lat-lat_min)/det)
    index_lon = round((point_lon-lon_min)/det)
    
    #由于默认的 lat_max值对应的index为0，因此需要反序
    index_lat = lat_grid.shape[0] -index_lat-1
    
    return [int(index_lat),int(index_lon)]

###############################################################################
def grid_interp_to_station(all_data, station_lon,station_lat ,method = 'linear'):
    '''
    func: 将等经纬度网格值 插值到 离散站点。使用griddata进行插值
    inputs: 
        all_data,形式为：[grid_lon,grid_lat,data] 即[经度网格，纬度网格，数值网格]
        station_lon: 站点经度
        station_lat: 站点纬度。可以是 单个点，列表或者一维数组
        method: 插值方法,默认使用 linear 。可选 cubic 和 nearest
    return: station_valus,返回该站点值 
    '''
    station_lon = np.array(station_lon).reshape(-1,1)
    station_lat = np.array(station_lat).reshape(-1,1)
    
    lon = all_data[0].reshape(-1,1)
    lat = all_data[1].reshape(-1,1)
    data = all_data[2].reshape(-1,1)
    
    points = np.concatenate([lon,lat],axis = 1)
    
    station_value = griddata(points,data,(station_lon,station_lat),method=method)
    
    station_value = station_value[:,:,0]
    
    return station_value




#%%
################################Part4: 本地时 <--> EC和SMS预报时刻的对应 #####################

def surface_time2_EC_BJ_time(src_f):
    '''
    func: 一般surface类型资料的文件名为 ： 18080408.000是北京时
          EC_thin资料的文件名为 : 18080408.003, 08表示起报时间，03表示预报时间;
          一天两次预报(08时和20时)，预报时长为114小时.为BJT,北京时
          当知道surface观测时间时(即文件名)，需要找到对应时刻的EC_thin资料。
          即因此需要在时间上匹配 surface文件 和 EC_thin文件。
          1.由于EC_thin的资料的起报时刻的数据为冷启动，不具有参考价值。
               因此这里默认选择起报时间3小时后的数据作为参考。
          2. 一天起报两次，因此任意一个时刻，都有多个不同起报时刻的预报结果相对应
              这里以时间距离最近的那个起报时刻的预报结果为准
    inputs: surface文件的文件名,eg: 18080408.000. 转换为EC_thin格式为 18080320.12. 
            [11,14,17,20] ---> [08.003, 08.006, 08.009, 08.012]        
            [23,02,05,08] ---> [20.003, 20.006, 20.009, 20.012]    
            
    return: 对应的EC_thin的文件名为：18080320.12
        
    '''
    
    #获取surface文件的 时间，具体到 年/月/日/小时
    src_time = src_f.split('.')[0]  
    
    #如果src_f = 2018080408.00,则只保留 18080408
    if len(src_time) > 8: 
        src_time = src_time[-8:]
    
    src_year = int(src_time[0:2])
    src_month = int(src_time[2:4])
    src_day = int(src_time[4:6])
    src_hour = int(src_time[6:])
    
    #如果 surface的观测时间不是在 规定的时刻内，则报错
    if src_hour not in [2,5,8,11,14,17,20,23]:
        print('error! surface file name is not correct!')
    
    else:
        
        #通过datetime.datetime模块输入时间，并进行时间的加减操作
        if src_hour in [11,14,17,20]:
            dst_year = src_year
            dst_month = src_month
            dst_day = src_day
            dst_hour = src_hour - 8
            
            
            dst_hour = str(dst_hour) if dst_hour > 10 else '0' + str(dst_hour)
            dst_hour = '08.0' + dst_hour
            

        elif src_hour == 23:
            
            dst_year = src_year
            dst_month = src_month
            dst_day = src_day
            dst_hour = src_hour - 20
            
            dst_hour = '20.00' + str(dst_hour)
 
            
        elif src_hour in [2,5,8]:
            
            src = datetime.datetime(src_year,src_month,src_day,src_hour)
            delta_day = datetime.timedelta(days = -1) 
            
            dst_time = src + delta_day  
            dst_year = dst_time.year
            dst_month = dst_time.month
            dst_day = dst_time.day
            dst_hour = dst_time.hour + 4
            
            dst_hour = str(dst_hour) if dst_hour > 10 else '0' + str(dst_hour)
            dst_hour = '20.0' + dst_hour
            
        
        dst_year = str(dst_year)
        dst_year = dst_year if len(dst_year)>1 else '0'+dst_year
        
        dst_month = str(dst_month) if dst_month>9 else '0'+str(dst_month)
        dst_day = str(dst_day) if dst_day>9 else '0'+str(dst_day)   
        
        dst_file = dst_year + dst_month + dst_day + dst_hour
        
                
        return dst_file   


###############################################################################
def surface_time2_SMS_time(src_f):
    '''
    func: 一般surface类型资料的文件名为 ： 18080408.000是北京时
         而SMS的格式时间格式为eg: 2018080406.003，06表示起报时间，003表示的预报时间为009时，世界时
         且一天起预报4次，每次预报12小时，时间分辨率为1小时。起报时间为 00 06 12 18；
         当知道surface观测时间时(即文件名)，需要找到对应时刻的SMS资料。
          即因此需要在时间上匹配 surface文件 和 SMS文件。
         1.由于SMS的资料的起报时刻的数据为冷启动，不具有参考价值。
           因此这里默认选择起报时间1小时后的数据作为参考。
         2. 由于一天起报4次，因此任意一个时刻，都有多个不同起报时刻的预报结果相对应
             这里以时间距离最近的那个起报时刻的预报结果为准
    inputs: surface文件的文件名,eg: 18080408.000.
            step1: 北京时转换为UTC时，对应的为 18080400.000
            step2: 转换为SMS格式为 2018080318.006.nc 
    return:
        转换后的SMS文件时次, eg: 2018080318.006.nc
    
    '''
    src_time = src_f.split('.')[0]
    
    #如果src_f = 2018080408.00,则只保留 18080408
    if len(src_time) > 8: 
        src_time = src_time[-8:]
    
    src_year = int(src_time[0:2])
    src_month = int(src_time[2:4])
    src_day = int(src_time[4:6])
    src_hour = int(src_time[6:])
    
    #如果 surface的观测时间不是在 规定的时刻内，则报错
    if src_hour not in [2,5,8,11,14,17,20,23]:
        print('error! surface file name is not correct!')
    
    else:
        
        #通过datetime.datetime模块输入时间，并进行时间的加减操作
        #减去9个小时。因为北京时和世界时相差8个小时，而两个的时间分辨率都为3小时；
        #为了方便起见，设置差9个小时
        src = datetime.datetime(src_year,src_month,src_day,src_hour)
        delta_h = datetime.timedelta(hours=-8) 
        dst_time = src + delta_h
        
        #获取转换为UTC时间的 年月日时 数值
        dst_year = dst_time.year
        dst_month = dst_time.month
        dst_day = dst_time.day
        dst_hour = dst_time.hour
        
        if dst_hour in [3,6]:
            det_h = dst_hour - 0 
            file2 = str(det_h)
            file2 = file2 if len(file2)>1 else '0'+file2
            file2 = '00.0'+file2
        
        elif dst_hour in [9,12]:
            det_h = dst_hour - 6 
            file2 = str(det_h)
            file2 = file2 if len(file2)>1 else '0'+file2
            file2 = '06.0'+file2
        
        elif dst_hour in [15,18]:
            det_h = dst_hour - 12 
            file2 = str(det_h)
            file2 = file2 if len(file2)>1 else '0'+file2
            file2 = '12.0'+file2
        
        elif dst_hour in [21]:
            det_h = dst_hour - 18 
            file2 = str(det_h)
            file2 = file2 if len(file2)>1 else '0'+file2
            file2 = '18.0'+file2
        
        elif dst_hour in [0]:
            det_h = dst_hour+24 - 18 
            
            det_day = datetime.timedelta(days=-1)
            dst_time = dst_time+det_day
            dst_year = dst_time.year
            dst_month = dst_time.month
            dst_day = dst_time.day
            
            file2 = str(det_h)
            file2 = file2 if len(file2)>1 else '0'+file2
            file2 = '18.0'+file2
        
        dst_year = str(dst_year)
                
        dst_year = dst_year if len(dst_year)>1 else '0'+dst_year
        
        dst_month = str(dst_month) if dst_month>9 else '0'+str(dst_month)
        
        dst_day = str(dst_day) if dst_day>9 else '0'+str(dst_day)
        
        file1 = dst_year+dst_month+dst_day
        
        dst_file = '20'+file1+file2+'.nc'
                    
        return dst_file    

#%%
############################### Part5: 构建地面观测(OBS)、EC、SMS的某个时刻的特征数据集 ####################### 
    
def get_all_surface_station_Dataset(r_filepath,
                                    loc_range = [30,50,105,125],
                                    filetype = 'array'):
    '''
    func: 构建surface站点数据集。
    inputs:   
        r_filepath: 地面降水观测文件名：eg: D:/ori_data/20180807/micaps/surface/r6-p/18080514.000
        loc_range:[lat_min,lat_max,lon_min,lon_max].只获取该经纬度范围内的站点数据 
                默认为：[30,50,105,125]
        filetype: 'array',默认输出为np.array类型。
                否则，默认输出为 pd.DataFrame类型
    return: all_vars_station_data。其中每列为一个变量，每行为一个站点数据  
    '''
    
    #读取r6-p数据 和 plot数据
    r_data = get_station_data(r_filepath,file_type = 'r6-p')
    
    #plot观测文件路径名。eg: surface/plot/18080508.000
    plot_filepath = r_filepath.replace('r6-p','plot')
    plot_data = get_station_data(plot_filepath,file_type = 'plot')
        
    #获取plot数据 + r6数据 站台号的list列表
    plot_station_series = list(plot_data[:,0])
    r_station_series = list(r_data[:,0])

    ##获取r_data的站台号序列在 plot_data站台序列号中的index
    ##一般情况下，r_data的stations_series是plot_stations 的子集
    index = [plot_station_series.index(station) for station in r_station_series]
    
    #获取与r_data相同站台号所在行的数据
    plot_data = plot_data[index,:]

    #是否站台序列一致
    print('plot data station is equal with r6 data station? {} '.format(np.all(plot_data[:,0] == r_data[:,0])))


    #设置对应的经纬度范围
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]
    
    ##r_data的[0,1,2,3,4]列分别为[站台号，经度，纬度，高度(m),降水量]
    ##plot_data的[0,1,2,6,7,16,19]列分别表示为[站台号,经度,纬度,风向,风速,露点,温度]
    
    #只选择在loc_range范围内的站点数据
    valid_r_data = r_data[r_data[:,2]>=lat_min]
    valid_r_data = valid_r_data[valid_r_data[:,2] <= lat_max]
    valid_r_data = valid_r_data[valid_r_data[:,1] >= lon_min]
    valid_r_data = valid_r_data[valid_r_data[:,1] <= lon_max]
    
    valid_plot_data = plot_data[plot_data[:,2]>=lat_min]
    valid_plot_data = valid_plot_data[valid_plot_data[:,2] <= lat_max]
    valid_plot_data = valid_plot_data[valid_plot_data[:,1] >= lon_min]
    valid_plot_data = valid_plot_data[valid_plot_data[:,1] <= lon_max]
    
    #是否站台序列一致
    print(np.all(valid_plot_data[:,0] == valid_r_data[:,0]))
    
    columns = ['0_T-0_surface_r6-p',  #逐6小时降水
               'station_num','lon','lat','height', 
               '5_T-0_surface_plot-T', #温度
               '1_T-0_surface_plot-Td', #露点温度
               '2_T-0_surface_plot-wind', #风速
               '2_T-0_surface_plot-wind-dir', #风向
               '2_T-0_surface_cos(plot-wind-dir)',#风向,cos
               '2_T-0_surface_sin(plot-wind-dir)' #风向，sin
               ] 
    
    
    all_vars_data = []
    
    all_vars_data.append(valid_r_data[:,4].reshape(-1,1)) #逐6小时降水
    all_vars_data.append(valid_r_data[:,0].reshape(-1,1)) #站台号
    all_vars_data.append(valid_r_data[:,1].reshape(-1,1)) #经度
    all_vars_data.append(valid_r_data[:,2].reshape(-1,1)) #纬度
    all_vars_data.append(valid_r_data[:,3].reshape(-1,1)) #高度
    all_vars_data.append(valid_plot_data[:,19].reshape(-1,1)) #温度
    all_vars_data.append(valid_plot_data[:,16].reshape(-1,1)) #露点温度
    all_vars_data.append(valid_plot_data[:,7].reshape(-1,1)) #风速
    all_vars_data.append(valid_plot_data[:,6].reshape(-1,1)) #风向
    all_vars_data.append(np.cos(valid_plot_data[:,6].reshape(-1,1)*np.pi/180))  #风向cos()
    all_vars_data.append(np.sin(valid_plot_data[:,6].reshape(-1,1)*np.pi/180))  #风向sin()
    
    all_vars_data = np.concatenate(all_vars_data,axis = 1)
    
    ########################################
     #上面获取有观测的站点的观测数据。下面构建所有站点的观测样本，其中有些站点在T时刻没有观测，则将观测值用nan填充
    ########################################
    
    #按站台号排序
    all_vars_data = pd.DataFrame(all_vars_data,columns = columns).sort_values('station_num', ascending=bool)
    
    #获取当前文件中的站点序列在 all_station列表中的位置
    index = [all_station.index(station) for station in list(all_vars_data['station_num'])]
    all_vars_data = all_vars_data.values
    
    #构建文件：样本数为总站点数len(all_station)，index位置上填上对应的观测数据，其他的以np.nan填充
    all_vars_data_pad = np.zeros(shape = (len(all_station),all_vars_data.shape[1]))
    all_vars_data_pad = pd.DataFrame(all_vars_data_pad,columns = columns).replace(0, np.nan)
    
    all_vars_data_pad.values[index] = all_vars_data
    
    all_vars_data_pad['station_num'] = all_station #填上所有站点
    all_vars_data_pad['lon'] = all_lon #填上所有站点的经度
    all_vars_data_pad['lat'] = all_lat #填上所有站点的纬度
    
    if filetype == 'array':
        
        all_vars_data_pad = all_vars_data_pad.values


    return all_vars_data_pad

#%%
def get_T0_jiami_surface_station_Dataset(jiami_filepath,
                                   loc_range = [30,50,105,125],
                                   filetype = 'pd'):
    
    '''
    func: 构建T0时刻的surface站点数据集。使用的是地面逐小时的加密观测文件
    inputs:   
        jiami_filepath: 地面加密观测文件名;
                eg: 'D:/ori_data/aws_jiami/2018080420.txt' 
        loc_range:[lat_min,lat_max,lon_min,lon_max].只获取该经纬度范围内的站点数据 
                默认为：[30,50,105,125]
        filetype: 'array',默认输出为np.array类型。
                否则，默认输出为 pd.DataFrame类型
    return: all_vars_station_data。其中每列为一个变量，每行为一个站点数据  
    '''
    
    all_station_file = 'D:/zhongqi/ori_data/all_jiami_station_lon_lat_alt.csv'
    station_lon_lat_pd = pd.read_csv(all_station_file)
    all_station = list(station_lon_lat_pd['station_num'])
    all_lon = list(station_lon_lat_pd['lon'])
    all_lat = list(station_lon_lat_pd['lat'])
    all_height = list(station_lon_lat_pd['height'])
    
    
    #读取加密观测的数据,数据类型为pd
    jiami_data = get_jiami_obs(jiami_filepath, filetype = 'pd')
    
   #  ['站号', '气温', '最高气温', '最低气温', '露点温度', '相对湿度', '小时降水量', 'C2分钟风向',
   # 'C2分钟平均风速', '最大风速的风向', '最大风速']
    
    #剔除jiami_data中某些 不在all_station中的站点观测，并为其他站点加上[经度、纬度、高度]信息
    obs_station = list(jiami_data['站号'])
    # print('obs_station length:',len(obs_station))
    error_station_index = [obs_station.index(station) for station in obs_station if station not in all_station]
    if len(error_station_index) > 0:  
        jiami_data = jiami_data.drop(index = error_station_index) 
        jiami_data.index = range(len(jiami_data))  #index重新排序
    
    ########################################
    #上面获取有观测的站点的观测数据。下面构建所有站点的观测样本，其中有些站点在T时刻没有观测，则将观测值用nan填充
    ########################################
    
    #获取剩下的obs_station在所有all_station中的index
    new_obs_station = list(jiami_data['站号'])
    # print('obs_station length:',len(new_obs_station))
    index = [all_station.index(station) for station in new_obs_station]
    
    
    #  ['站号', '气温', '最高气温', '最低气温', '露点温度', '相对湿度', '小时降水量', 'C2分钟风向',
   # 'C2分钟平均风速', '最大风速的风向', '最大风速']
    columns_ch = list(jiami_data.columns)
    
    columns_en = ['0_T-0_surface_r1-p',   #小时降水量 index = 6
                  'station_num','lon','lat','height', 
                  '4_T-0_surface_plot-T',   #温度 index = 1
                  '1_T-0_surface_plot-Td', #露点温度 index = 4
                  '1_T-0_surface_plot-RH', #相对湿度 index = 5
                  '2_T-0_surface_plot-wind-max', #最大风速 index = 10
                  '2_T-0_surface_plot-wind-max-dir', #最大风速风向 index = 9
                  '2_T-0_surface_plot-cos(wind-max-dir)',#风向,cos index = 9
                  '2_T-0_surface_plot-sin(wind-max-dir)', #风向,sin index = 9
                  '2_T-0_surface_plot-wind-mean', #c2分钟平均风速 index = 8
                  '2_T-0_surface_plot-wind-mean-dir', #c2分钟平均风向  index = 7
                  '2_T-0_surface_plot-cos(wind-mean-dir)', #c2分钟平均风向,cos  index = 7
                  '2_T-0_surface_plot-sin(wind-mean-dir)', #c2分钟平均风向,sin   index = 7
                  ]
    

    #构建文件：样本数为总站点数len(all_station)，index位置上填上对应的观测数据，其他的以np.nan填充
    all_vars_data_pad = np.zeros(shape = (len(all_station),len(columns_en)))
    all_vars_data_pad = pd.DataFrame(all_vars_data_pad,columns = columns_en).replace(0, np.nan)
    
    all_vars_data_pad['0_T-0_surface_r1-p'].values[index] = jiami_data['小时降水量']
    all_vars_data_pad['station_num'] = all_station
    all_vars_data_pad['lon'] = all_lon
    all_vars_data_pad['lat'] = all_lat
    all_vars_data_pad['height'] = all_height
    all_vars_data_pad['4_T-0_surface_plot-T'].values[index] = jiami_data['气温']
    all_vars_data_pad['1_T-0_surface_plot-Td'].values[index] = jiami_data['露点温度']
    all_vars_data_pad['1_T-0_surface_plot-RH'].values[index] = jiami_data['相对湿度']
    all_vars_data_pad['2_T-0_surface_plot-wind-max'].values[index] = jiami_data['最大风速']
    all_vars_data_pad['2_T-0_surface_plot-wind-max-dir'].values[index] = jiami_data['最大风速的风向']
    all_vars_data_pad['2_T-0_surface_plot-cos(wind-max-dir)'].values[index] = np.cos(jiami_data['最大风速的风向']*np.pi/180)
    all_vars_data_pad['2_T-0_surface_plot-sin(wind-max-dir)'].values[index] = np.sin(jiami_data['最大风速的风向']*np.pi/180)
    
    all_vars_data_pad['2_T-0_surface_plot-wind-mean'].values[index] = jiami_data['C2分钟平均风速']
    all_vars_data_pad['2_T-0_surface_plot-wind-mean-dir'].values[index] = jiami_data['C2分钟风向']
    all_vars_data_pad['2_T-0_surface_plot-cos(wind-mean-dir)'].values[index] = np.cos(jiami_data['C2分钟风向']*np.pi/180)
    all_vars_data_pad['2_T-0_surface_plot-sin(wind-mean-dir)'].values[index] = np.sin(jiami_data['C2分钟风向']*np.pi/180)
            
    if filetype == 'array':
        all_vars_data_pad = all_vars_data_pad.values

    return all_vars_data_pad  
    

def get_T3_jiami_surface_station_Dataset(jiami_filepath,
                                   loc_range = [30,50,105,125],
                                   filetype = 'pd'):
    
    '''
    func: 构建加密观测的3小时累计降水变量
    inputs:   
        jiami_filepath: 地面加密观测文件名;
                eg: 'D:/ori_data/aws_jiami/2018080420.txt' 
        loc_range:[lat_min,lat_max,lon_min,lon_max].只获取该经纬度范围内的站点数据 
                默认为：[30,50,105,125]
        filetype: 'array',默认输出为np.array类型。
                否则，默认输出为 pd.DataFrame类型
    return: all_vars_station_data。其中每列为一个变量，每行为一个站点数据  
    '''
    file_time0 = jiami_filepath.split('/')[-1].split('.')[0] #获取file对应的观测时间,eg: 2018080420
    
    #获取T时刻的jiami观测的时间
    file_year = int(file_time0[0:4])
    file_month = int(file_time0[4:6])
    file_day = int(file_time0[6:8])
    file_hour = int(file_time0[8:])
    
    #获取T-1，T-2时刻的jiami观测文件名,分别滞后1、2小时
    T0 = datetime.datetime(file_year,file_month,file_day,file_hour)
    delta_h1 = datetime.timedelta(hours = -1)
    delta_h2 = datetime.timedelta(hours = -2)
    
    T1 = T0 + delta_h1
    T2 = T0 + delta_h2
    
    file_time1 = T1.strftime('%Y%m%d%H') #文件名为：eg: 2018080419
    file_time2 = T2.strftime('%Y%m%d%H')
    
    jiami_filepath1 = jiami_filepath.replace(file_time0, file_time1)
    jiami_filepath2 = jiami_filepath.replace(file_time0, file_time2)
    
    #判断jiami_filepath2是否存在，如果存在，则jiami_filepath1必然也存在
    if not os.path.exists(jiami_filepath2):
        print('Error!',jiami_filepath2 ,'not exits!')
    
    data0 = get_T0_jiami_surface_station_Dataset(jiami_filepath, loc_range = [30,50,105,125],filetype = 'pd')
    data1 = get_T0_jiami_surface_station_Dataset(jiami_filepath1,loc_range = [30,50,105,125],filetype = 'pd')
    data2 = get_T0_jiami_surface_station_Dataset(jiami_filepath2,loc_range = [30,50,105,125],filetype = 'pd')
    
    #获取累计3小时降水,和累计2小时降水
    r3_p = data0['0_T-0_surface_r1-p'] + data1['0_T-0_surface_r1-p'] + data2['0_T-0_surface_r1-p']
    r2_p = data0['0_T-0_surface_r1-p'] + data1['0_T-0_surface_r1-p']
    
    data0.insert(0, '0_T-0_surface_r3-p',r3_p)
    data0.insert(1, '0_T-0_surface_r2-p',r2_p)
        
    return np.array(data0.values) if filetype == 'array' else data0

#%%
def get_all_ECthin_Station_dataset_ori(EC_path, surface_file,loc_range = [30,50,105,125]):
    '''
    func: 根据surface_file的站点数据，获取对应的时刻的 EC细网格物理量资料，并将网格资料插值到站点
    inputs:
        EC_path: ecmwf_thin所在路径, eg: 'D:/zhongqi/ori_data/20190804/micaps'
        surface_file: 地面降水观测文件路径+ 文件名：
                eg: 'D:/ori_data/aws_jiami/2018080420.txt' 
        loc_range: [lat_min,lat_max,lon_min,lon_max]。只获取该经纬度范围内的站点插值数据 
    return:
        返回一个DataFrame。columns 为EC变量名称及其路径，数值为对应插值到站点上的值 
        
    '''
    
    all_station_file = 'D:/zhongqi/ori_data/all_jiami_station_lon_lat_alt.csv'
    station_lon_lat_pd = pd.read_csv(all_station_file)
    all_station = list(station_lon_lat_pd['station_num'])
    all_lon = list(station_lon_lat_pd['lon'])
    all_lat = list(station_lon_lat_pd['lat'])
    all_height = list(station_lon_lat_pd['height'])
    
    
    #获取该surface观测的时间，eg: 18080514.000 / 2018080420
    surface_file_time = surface_file.split('/')[-1].split('.')[0] 
    
    #获取与 surface_file_time时间比较接近的 EC资料对应的时间，eg: 18080420.009
    EC_file_time = surface_time2_EC_BJ_time(surface_file_time)
    
    #定位到EC_path路径
    os.chdir(EC_path)
    
    #获取EC_thin的filelist。该文档记录了需要的EC_thin物理量的路径：eg: EC_thin/TP/r3 EC_thin/Q/850
    EC_filename_list_path = 'D:/zhongqi/Features_Lists/EC_filename_list.xlsx'  
    EC_filename_list = pd.read_excel(EC_filename_list_path)
    all_EC_filepath = EC_filename_list['filepath'].dropna()
    
    all_EC_file_stations_values = [] 
    
    ##由于可能存在不与surface_file时刻对应的EC资料，因此需要进行检查
    EC_file0 = all_EC_filepath[0].replace('EC_thin','ecmwf_thin')+'/'+EC_file_time
    
    #如果不存在，则报错，如果存在;
    if not os.path.exists(EC_file0):
        print('Error!',EC_file0,'not exists! please check the file')
        
    else: 
        t1 = time.time()
        
        for i in range(len(all_EC_filepath)):
            
            #EC数据存储时，文件名为：ecmwf_thin,因此先replace一下。之后获取完整文件名: 
            #eg: ecmwf_thin/TP/r3/18080420.009
            EC_file = os.path.join(all_EC_filepath[i].replace('EC_thin','ecmwf_thin'),EC_file_time)
            
            #获取EC网格资料，并插值到特定站点上
            EC_data = get_EC_thin_physic_data(EC_file,plot = False)
            valid_EC_station_values = grid_interp_to_station(EC_data,
                                                             station_lon = all_lon,
                                                             station_lat = all_lat,
                                                             method = 'linear')
            
            all_EC_file_stations_values.append(valid_EC_station_values)
        
        all_EC_file_stations_values = np.concatenate(all_EC_file_stations_values,axis = 1)
        print('total time cost:',time.time()-t1)
    
#        将数组转换为 DataFrame
        all_EC_file_stations_values = pd.DataFrame(all_EC_file_stations_values,
                                                   columns=all_EC_filepath)
        
        
    return all_EC_file_stations_values


def get_all_ECthin_Station_dataset_dst(ori_data,filetype = 'pd'):
    '''
    func: 将 get_all_ECthin_Station_dataset_ori 函数的输入进行进一步特征组合。
    
    inputs: 
        ori_data：为pd.DataFrame格式，其中每列为一个ECthin的物理量名称，共计45个物理量
        filetype: 'array',默认输出为np.array类型。
                否则，默认输出为 pd.DataFrame类型
    return: 
        返回 将一些物理量进行特征变化 的结果。其中每列为一个组合特征，共计45个特征
    '''
    
    all_station_file = 'D:/zhongqi/ori_data/all_jiami_station_lon_lat_alt.csv'
    station_lon_lat_pd = pd.read_csv(all_station_file)
    all_station = list(station_lon_lat_pd['station_num'])
    all_lon = list(station_lon_lat_pd['lon'])
    all_lat = list(station_lon_lat_pd['lat'])
    all_height = list(station_lon_lat_pd['height'])
    
    
    #拷贝.可以保证后续不会修改ori_data的值
    ori_data = ori_data.values.copy()
    
    rows,cols = ori_data.shape
    dst_data = np.zeros((rows,cols + 3))
    
    print('ori_data.shape:',ori_data.shape)
    print('dst_data.shape',dst_data.shape)

    #前11个变量不变
    dst_data[:,0:11] = ori_data[:,0:11] 
    
    #mean(8,9,10),850/700/500hPa三层垂直速度平均
    dst_data[:,11] = np.mean(ori_data[:,[8,9,10]],axis = 1) 
    
    dst_data[:,12:18] = ori_data[:,11:17]
    
    #1000/950/925/900/850/700hPa平均U
    dst_data[:,18] = np.mean(ori_data[:,[17,18,19,20]],axis = 1) 
    
    #1000/950/925/900/850/700hPa平均V
    dst_data[:,19] = np.mean(ori_data[:,[25,26,27,28]],axis = 1)
    
    #1000/950/925/900/850/700/600/500/400/300hPa平均U和Umax
    dst_data[:,20] = np.mean(ori_data[:,[17,18,19,20,21,22,23,24]],axis = 1)
    dst_data[:,21] = np.max(ori_data[:,[17,18,19,20,21,22,23,24]],axis = 1)
    
    #1000/950/925/900/850/700/600/500/400/300hPa平均V和Vmax
    dst_data[:,22] = np.mean(ori_data[:,[25,26,27,28,29,30,31,32]],axis = 1)
    dst_data[:,23] = np.max(ori_data[:,[25,26,27,28,29,30,31,32]],axis = 1)
    
    dst_data[:,24] = ori_data[:,22]
    dst_data[:,25] = ori_data[:,30]
    
    #500hPa风向，用cos/sin表示，v大于0，则cos大于0;u > 0,sin > 0
    dst_data[:,26] = ori_data[:,30]/np.sqrt(np.square(ori_data[:,22]) + np.square(ori_data[:,30])) #cos
    dst_data[:,27] = ori_data[:,22]/np.sqrt(np.square(ori_data[:,22]) + np.square(ori_data[:,30])) #sin
    
    dst_data[:,28] = ori_data[:,19]
    dst_data[:,29] = ori_data[:,27]
    
    #850hPa风向，用cos/sin表示，v大于0，则cos大于0; u > 0,sin > 0
    dst_data[:,30] = ori_data[:,27]/np.sqrt(np.square(ori_data[:,19]) + np.square(ori_data[:,27])) #cos
    dst_data[:,31] = ori_data[:,19]/np.sqrt(np.square(ori_data[:,19]) + np.square(ori_data[:,27])) #sin
    
 
    dst_data[:,32] = ori_data[:,18]
    dst_data[:,33] = ori_data[:,26]
    
    #925hPa风向，用cos/sin表示，v大于0，则cos大于0,u > 0,sin > 0
    dst_data[:,34] = ori_data[:,26]/np.sqrt(np.square(ori_data[:,18]) + np.square(ori_data[:,26]))
    dst_data[:,35] = ori_data[:,18]/np.sqrt(np.square(ori_data[:,18]) + np.square(ori_data[:,26]))
    
    
    dst_data[:,36:42] = ori_data[:,33:39]
    
    #500hPa-850hPa的假相当位温差
    dst_data[:,42] = ori_data[:,38]-ori_data[:,36]
    dst_data[:,43:] = ori_data[:,40:]
    
    #如果 filetype 为 'array',则输出np.array数组
    #否则输出 pd.DataFrame，columns为组合后的变量名称
    if filetype != 'array':
        
        EC_filename_list_path = 'D:/zhongqi/Features_Lists/EC_filename_list.xlsx'  
        #读取变量说明特征说明 
        EC_filename_list = pd.read_excel(EC_filename_list_path)
        all_EC_Com_Features_Name = EC_filename_list['EC_Com_Features_Name']
        
        #将数组转换为 DataFrame
        dst_data = pd.DataFrame(dst_data, columns = all_EC_Com_Features_Name)
        dst_data['station_num'] = all_station #最后一列为站点序号
        dst_data['lon'] = all_lon #经度
        dst_data['lat'] = all_lat #纬度
        dst_data['height'] = all_height #高度
                                              
    return dst_data


#%%
def get_T0_SMS_Station_dataset(SMS_path, surface_file,loc_range = [30,50,105,125],
                                filetype = 'pd',
                                if_plot = False):
    '''
    func: 获取与surface_file同时刻的 SMS(华东区域中心的)资料并将其插值到站点上   
    inputs: 
        SMS_path: SMS的.nc文件所在的路径,eg: eg: 'D:/ori_data/20180807/micaps/warr/nc'
        surface_file: 地面降水观测文件路径+ 文件名
              eg: 'D:/ori_data/aws_jiami/2018080420.txt' 
        loc_range: [lat_min,lat_max,lon_min,lon_max]。只获取该经纬度范围内的站点插值数据
        filetype: 'array',默认输出为np.array类型。
                否则，默认输出为 pd.DataFrame类型
        if_plot: 确认是否画出插值前后的降水分布图，默认False
    returns: 
        all_vars_station_data。其中每列为一个变量，每行为一个站点数据  
        返回一个DataFrame。columns 为EC变量名称及其路径，数值为对应插值到站点上的值 
    '''
    
    
    all_station_file = 'D:/zhongqi/ori_data/all_jiami_station_lon_lat_alt.csv'
    station_lon_lat_pd = pd.read_csv(all_station_file)
    all_station = list(station_lon_lat_pd['station_num'])
    all_lon = list(station_lon_lat_pd['lon'])
    all_lat = list(station_lon_lat_pd['lat'])
    all_height = list(station_lon_lat_pd['height'])
    
    
    #获取该surface观测的时间，eg: 2018080514
    surface_file_time = surface_file.split('/')[-1].split('.')[0]
    
    #定位到SMS_path路径下， eg: D:/ori_data/20180807/micaps/warr/nc 这路径下
    os.chdir(SMS_path)
        
    # #设置对应的经纬度范围
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]
    
    #获取与 surface_file_time时间比较接近的 SMS资料对应的时间，eg: 18080506.003.nc
    SMS_file_time = surface_time2_SMS_time(surface_file_time)

    valid_vars = ['APCP_P8_L1_GLC0_acc',
                  'DPT_P0_L103_GLC0', 
                  'TMP_P0_L103_GLC0',
                  'RH_P0_L103_GLC0',
                  'UGRD_P0_L103_GLC0',
                  'VGRD_P0_L103_GLC0',
                  'PRES_P0_L101_GLC0',
                  'CAPE_P0_L1_GLC0',
                  'CIN_P0_L1_GLC0',
                  'REFC_P0_L10_GLC0']
                  
    
    t1 = time.time()
    
    all_vars_grid_data = []
            
    #读取SMS_file_time0文件中valid_vars变量的数据
    f = nc.Dataset(SMS_file_time0)
    for var in valid_vars[0:]:
        data = f[var][:]
        all_vars_grid_data.append(data)
    
    #获取经纬度数据
    grid_lon = f['ELON_P0_L1_GLC0'][:]
    grid_lat = f['NLAT_P0_L1_GLC0'][:] 
    
    #获取在在loc_range内的index
    index1 = np.where(grid_lon <= lon_max)
    index2 = np.where(grid_lat[index1] >= lat_min)
    
    #获取在loc_range内的经纬度值
    loc_grid_lon = grid_lon[index1][index2]
    loc_grid_lat = grid_lat[index1][index2]
    
    f.close()
    
    all_vars_station_data = []
    
    #将格点插值到站点
    for grid_data in all_vars_grid_data[0:]:
        t2 = time.time()
        
        #只考虑loc_range内的数据，加快插值速度
        grid_data = grid_data[index1][index2]
        valid_vars_station_data = grid_interp_to_station([loc_grid_lon,loc_grid_lat,grid_data],
                                                         station_lon = all_lon,
                                                         station_lat = all_lat,
                                                         method = 'linear')
        
        print('cost:',time.time() - t2)
        all_vars_station_data.append(valid_vars_station_data)
    
    if if_plot:
        
        fill_value = 9999
        #画图比较,mask掉一些nan
        i = 0
        conf_data = pd.DataFrame(all_vars_grid_data[i]).replace(np.nan, fill_value).values
        mask1 = conf_data == fill_value
        mask2 = conf_data > 200
        mask = mask1 + mask2
        ma_conf_data = ma.array(conf_data, mask = mask)
        
        print(ma_conf_data.max())
        print(ma_conf_data.min())
        print()
        
        
        contourf_data_on_map(ma_conf_data,grid_lon,grid_lat)
        scatter_station_on_map(valid_surface_data[:,1],valid_surface_data[:,2],
                               all_vars_station_data[i], fill_value = fill_value)    
    
    all_vars_station_data = np.concatenate(all_vars_station_data,axis = 1)
    
    #如果 filetype 为 'array',则输出np.array数组
    #否则输出 pd.DataFrame，columns为组合后的变量名称
    if filetype != 'array':
        columns = ['0_T-0_SMS_ACC-r1',   #1小时累计降水
                    '1_T-0_SMS_DPT-P0-L103-GLC0',
                    '4_T-0_SMS_TMP-PO-L103-GLC0',
                    '1_T-0_SMS_RH-P0-L103-GLC0',
                    '2_T-0_SMS_UGRD-P0-L103-GLC0',
                    '2_T-0_SMS_VGRD-P0-L103-GLC0',
                    '5_T-0_SMS_PRES-L101-GLC0',
                    '3_T-0_SMS_CAPE-P0-L1-GLC0',
                    '3_T-0_SMS_CIN-P0-L1-GLC0',
                    '3_T-0_SMS_REFC-P0-L10-GLC0']
        
        all_vars_station_data = pd.DataFrame(all_vars_station_data,columns = columns)
        all_vars_station_data['station_num'] = all_station
        all_vars_station_data['lon'] = all_lon
        all_vars_station_data['lat'] = all_lat
        all_vars_station_data['height'] = all_height
        
    print('total time cost:',time.time()-t1)
    
    return all_vars_station_data


def get_T3_SMS_Station_dataset(SMS_path, surface_file,loc_range = [30,50,105,125],
                                filetype = 'array',
                                if_plot = False):
    '''
    func: 获取与surface_file同时刻的 SMS(华东区域中心的)资料 + 累计3/2/1小时降水 并将其插值到站点上   
    inputs: 
        SMS_path: SMS的.nc文件所在的路径,eg: eg: 'D:/ori_data/20180807/micaps/warr/nc'
        surface_file: 地面降水观测文件路径+ 文件名;
            eg: 'D:/ori_data/aws_jiami/2018080420.txt' 
        loc_range: [lat_min,lat_max,lon_min,lon_max]。只获取该经纬度范围内的站点插值数据
        filetype: 'array',默认输出为np.array类型。
                否则，默认输出为 pd.DataFrame类型
        if_plot: 确认是否画出插值前后的降水分布图，默认False
    returns: 
        all_vars_station_data。其中每列为一个变量，每行为一个站点数据  
        返回一个DataFrame。columns 为EC变量名称及其路径，数值为对应插值到站点上的值 
    '''
    
    all_station_file = 'D:/zhongqi/ori_data/all_jiami_station_lon_lat_alt.csv'
    station_lon_lat_pd = pd.read_csv(all_station_file)
    all_station = list(station_lon_lat_pd['station_num'])
    all_lon = list(station_lon_lat_pd['lon'])
    all_lat = list(station_lon_lat_pd['lat'])
    all_height = list(station_lon_lat_pd['height'])
    

    #获取该surface观测的时间，eg: 2018080514
    surface_file_time = surface_file.split('/')[-1].split('.')[0] 
    
     #定位到SMS_path路径下，  eg: D:/ori_data/20180807/micaps/warr/nc 这这路径下
    os.chdir(SMS_path)
        
    # #设置对应的经纬度范围
    lat_min = loc_range[0]
    lat_max = loc_range[1]
    lon_min = loc_range[2]
    lon_max = loc_range[3]
    
    
    #获取与 surface_file_time时间比较接近的 SMS资料对应的时间，eg: 18080506.003.nc
    SMS_file_time = surface_time2_SMS_time(surface_file_time)

    valid_vars = ['DPT_P0_L103_GLC0', 
                  'TMP_P0_L103_GLC0',
                  'RH_P0_L103_GLC0',
                  'UGRD_P0_L103_GLC0',
                  'VGRD_P0_L103_GLC0',
                  'PRES_P0_L101_GLC0',
                  'CAPE_P0_L1_GLC0',
                  'CIN_P0_L1_GLC0',
                  'REFC_P0_L10_GLC0']
                  
    acc_var = 'APCP_P8_L1_GLC0_acc'
    
    #需要三小时累计降水，而SMS资料是逐小时的，因此需要用到SMS_file_time之前两个时刻的数据
    hour = SMS_file_time.split('.')[1] #挑选出时间 eg: 003
    hour_2 = '00'+str(int(hour)-2) 
    hour_1 = '00'+str(int(hour)-1)
    
    SMS_file_time0 = SMS_file_time
    SMS_file_time1 = SMS_file_time.split('.')[0] + '.'+ hour_1 +'.nc'
    SMS_file_time2 = SMS_file_time.split('.')[0] + '.'+ hour_2 +'.nc'
    
    
    #判断文件是否存在，如果SMS_file_time2存在，则SMS_file0/1必然存在
    if not os.path.exists(SMS_file_time2):
        print('Error!',SMS_file_time2 ,'not exits!')
    
    t1 = time.time()
    
    all_vars_grid_data = []
    
    #获得3/2/1个小时累计降水, 三个文件的逐小时降水 累加
    acc_r3 = 0
    acc_r2 = 0
    acc_r1 = 0
    
    i = 0
    for file in [SMS_file_time0,SMS_file_time1,SMS_file_time2]:
        f = nc.Dataset(file)
        r1 = f[acc_var][:]
        if i == 0:
            acc_r1 = r1
        acc_r3 = acc_r3 + r1
        i = i + 1
        f.close()
        
    acc_r2 = acc_r3 - r1

    all_vars_grid_data.append(acc_r3)
    all_vars_grid_data.append(acc_r2)
    all_vars_grid_data.append(acc_r1)
    
    #读取SMS_file_time0文件中valid_vars变量的数据
    f = nc.Dataset(SMS_file_time0)
    for var in valid_vars[0:]:
        data = f[var][:]
        all_vars_grid_data.append(data)
    
    #获取经纬度数据
    grid_lon = f['ELON_P0_L1_GLC0'][:]
    grid_lat = f['NLAT_P0_L1_GLC0'][:] 
    
    #获取在在loc_range内的index
    index1 = np.where(grid_lon <= lon_max)
    index2 = np.where(grid_lat[index1] >= lat_min)
    
    #获取在loc_range内的经纬度值
    loc_grid_lon = grid_lon[index1][index2]
    loc_grid_lat = grid_lat[index1][index2]
    
    f.close()
    
    all_vars_station_data = []
    
    #将格点插值到站点
    for grid_data in all_vars_grid_data[0:]:
        t2 = time.time()
        
        #只考虑loc_range内的数据，加快插值速度
        grid_data = grid_data[index1][index2]
        valid_vars_station_data = grid_interp_to_station([loc_grid_lon,loc_grid_lat,grid_data],
                                                         station_lon = all_lon,
                                                         station_lat = all_lat,
                                                         method = 'linear')
        
        print('cost:',time.time() - t2)
        all_vars_station_data.append(valid_vars_station_data)
    
    if if_plot:
        
        fill_value = 9999
        #画图比较,mask掉一些nan
        i = 0
        conf_data = pd.DataFrame(all_vars_grid_data[i]).replace(np.nan, fill_value).values
        mask1 = conf_data == fill_value
        mask2 = conf_data > 200
        mask = mask1 + mask2
        ma_conf_data = ma.array(conf_data, mask = mask)
        
        print(ma_conf_data.max())
        print(ma_conf_data.min())
        print()
        
        
        contourf_data_on_map(ma_conf_data,grid_lon,grid_lat)
        scatter_station_on_map(valid_surface_data[:,1],valid_surface_data[:,2],
                               all_vars_station_data[i], fill_value = fill_value)    
    
    all_vars_station_data = np.concatenate(all_vars_station_data,axis = 1)
    
    #如果 filetype 为 'array',则输出np.array数组
    #否则输出 pd.DataFrame，columns为组合后的变量名称
    if filetype != 'array':
        columns = ['0_T-0_SMS_ACC-r3',  #3小时累计降水
                   '0_T-0_SMS_ACC-r2',  #2小时累计降水
                   '0_T-0_SMS_ACC-r1',  #1小时累计降水
                    '1_T-0_SMS_DPT-P0-L103-GLC0', #2m露点温度
                    '4_T-0_SMS_TMP-PO-L103-GLC0', #2m温度
                    '1_T-0_SMS_RH-P0-L103-GLC0',  #2m相对湿度 
                    '2_T-0_SMS_UGRD-P0-L103-GLC0',  #东西方向10m风速
                    '2_T-0_SMS_VGRD-P0-L103-GLC0', #南北方向10m风速
                    '5_T-0_SMS_PRES-L101-GLC0',   #海平面气压
                    '3_T-0_SMS_CAPE-P0-L1-GLC0',   #对流有效位能
                    '3_T-0_SMS_CIN-P0-L1-GLC0',   #对流抑制能
                    '3_T-0_SMS_REFC-P0-L10-GLC0'  #综合雷达回波
                    ]
        
        all_vars_station_data = pd.DataFrame(all_vars_station_data,columns = columns)
        all_vars_station_data['station_num'] = all_station
        all_vars_station_data['lon'] = all_lon
        all_vars_station_data['lat'] = all_lat
        all_vars_station_data['height'] = all_height
        
    print('total time cost:',time.time()-t1)
    
    return all_vars_station_data

#%%
########################################Part6: 与降水有关评分函数 ######################## 
def prep_clf(obs,pre, threshold=0.1):
    '''
    func: 计算二分类结果-混淆矩阵的四个元素
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        hits, misses, falsealarms, correctnegatives
        #aliases: TP, FN, FP, TN 
    '''
    #根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)

    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))

    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))

    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))

    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))

    return hits, misses, falsealarms, correctnegatives


def precision(obs, pre, threshold=0.1):
    '''
    func: 计算精确度precision: TP / (TP + FP)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FP)


def recall(obs, pre, threshold=0.1):
    '''
    func: 计算召回率recall: TP / (TP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return TP / (TP + FN)


def ACC(obs, pre, threshold=0.1):
    '''
    func: 计算准确度Accuracy: (TP + TN) / (TP + TN + FP + FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    
    returns:
        dtype: float
    '''

    TP, FN, FP, TN = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return (TP + TN) / (TP + TN + FP + FN)

def FSC(obs, pre, threshold=0.1):
    '''
    func:计算f1 score = 2 * ((precision * recall) / (precision + recall))
    '''
    precision_socre = precision(obs, pre, threshold=threshold)
    recall_score = recall(obs, pre, threshold=threshold)

    return 2 * ((precision_socre * recall_score) / (precision_socre + recall_score))

def TS(obs, pre, threshold=0.1):
    
    '''
    func: 计算TS评分: TS = hits/(hits + falsealarms + misses) 
    	  alias: TP/(TP+FP+FN)
    inputs:
        obs: 观测值，即真实值；
        pre: 预测值；
        threshold: 阈值，判别正负样本的阈值,默认0.1,气象上默认格点 >= 0.1才判定存在降水。
    returns:
        dtype: float
    '''

    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre, threshold=threshold)

    return hits/(hits + falsealarms + misses) 

def MAR(obs, pre, threshold=0.1):
    '''
    func : 计算漏报率 misses / (hits + misses)
    MAR - Missing Alarm Rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: MAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return misses / (hits + misses)

def FAR(obs, pre, threshold=0.1):
    '''
    func: 计算误警率。falsealarms / (hits + falsealarms) 
    FAR - false alarm rate
    Args:
        obs (numpy.ndarray): observations
        pre (numpy.ndarray): prediction
        threshold (float)  : threshold for rainfall values binaryzation
                             (rain/no rain)
    Returns:
        float: FAR value
    '''
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre = pre,
                                                           threshold=threshold)

    return falsealarms / (hits + falsealarms)


def multil_scores(obs, pre, threshold):
    '''
    func: 输入某个阈值下，输出该阈值下预报相对观测的评分（以直接打印的方式） 
    '''
    print('Threshold:', threshold)
    print('Acc:',ACC(obs,pre,threshold = threshold))
    print('Recall score:',recall(obs,pre,threshold = threshold))
    print('Precision score:',precision(obs,pre,threshold = threshold))
    print('F1 score:',FSC(obs,pre,threshold = threshold))
    print('TS评分:',TS(obs,pre,threshold = threshold))
    print('漏报率(MAR)评分:',MAR(obs,pre,threshold = threshold))
    print('误报率(FAR)评分:',FAR(obs,pre,threshold = threshold))
    

def plot_multi_scores(obs, pre, thresholds = [0.1, 5,10,15,20,25,30,35,40], title = None):

    '''
    画出不同阈值情况下的 模型预测对比实况的 Acc/Recall/Precision/F1/TS/MAR/FAR 的评分情况
    Parameters
    ----------
    obs : np.array
        数组
    pre : np.array
        数组.
    thresholds : list
        The default is [0.1, 5,10,15,20,25,30,35,40]. 进行评分判断的阈值
    title : str or None
        The default is None, 图片的 title
    Returns
    -------
    None.

    '''
    
    Acc_scores = []
    Recall_scores = []
    Precision_scores = []
    F1_scores = []
    TS_scores = []
    MAR_scores = []
    FAR_scores = []
    
    # thresholds = [0.1,5,10, 15,20,25,30]
    
    for threshold in thresholds:
        Acc_scores.append(ACC(obs,pre, threshold = threshold))
        Recall_scores.append(recall(obs,pre, threshold = threshold))
        Precision_scores.append(precision(obs,pre, threshold = threshold))
        F1_scores.append(FSC(obs,pre, threshold = threshold))
        TS_scores.append(TS(obs,pre, threshold = threshold))
        MAR_scores.append(MAR(obs,pre, threshold = threshold))
        FAR_scores.append(FAR(obs,pre, threshold = threshold))
        
         
    plt.figure(figsize = (12,8))
    plt.plot(thresholds, Acc_scores, linewidth = 2,label = 'Acc')
    plt.plot(thresholds, Recall_scores,linewidth = 2, label = 'Recall')
    plt.plot(thresholds, Precision_scores,linewidth = 2, label = 'Precision')
    plt.plot(thresholds, F1_scores, linewidth = 2,label = 'F1')
    plt.plot(thresholds, TS_scores,linewidth = 2, label = 'TS')
    plt.plot(thresholds, MAR_scores, linewidth = 2,label = 'MAR')
    plt.plot(thresholds, FAR_scores, linewidth = 2, label = 'FAR')
    plt.plot([20,20],[0,1],linewidth = 2)
    
    plt.ylabel('scores',fontsize = 20)
    plt.yticks(fontsize = 18)
    plt.ylim([0,1])
    
    plt.xlabel('Thresholds',fontsize = 20)
    plt.xticks(thresholds,thresholds, fontsize = 18)
    
    plt.legend(fontsize = 12)
    
    if title:
        plt.title(title, fontsize = 20)

    plt.show()
    
    return None
    

#%%
####################################Part7: 数据分析 #####################################
def drop_outlier(x,max_threshold=50,min_threshold=1):
    '''
    func: 处理离群值，用异常阈值 或者 max_threshold代替
    Parameter
    ---------
    x: np.array
    max_threshold: int
        default 50, 1小时降水量最大正常值
    min_threshold: int
        default 1, 使用data >= min_threshold 的样本去做降水分布分析
    '''
    
    #如果data中所有值在max_threshold范围内，则认为该data无异常
    
    data = x.copy()
    
    if np.all(data <= max_threshold):
        return data
    
    else:
        valid_data = data.ravel()[data.ravel() >= min_threshold]
        
        #获取最大值，平均值和标准差
        max_value = np.max(data)
        mean_valid_data = np.mean(valid_data)
        std = np.std(valid_data)
        
        #设置最小异常值
        outlier_threshold = mean_valid_data + 3*std
        
        print('max_value:',max_value)
        print('mean_valid_value:',mean_valid_data)
        print('std:',std)
        print('outlier_threshold:',outlier_threshold)
        
        #如果异常阈值>max_value，则说明所有值都是正常的
        if outlier_threshold >= max_value:
            pass
        else:
            if outlier_threshold >= max_threshold:  #如果存在异常，如果异常阈值 > max_threshold 
                data[data >= outlier_threshold] = outlier_threshold
                # data[data >= outlier_threshold] = np.nan
            else: 
                #如果存在异常，异常阈值小于max_threshold,则将大于 max_threshold的异常值变为max_threshold 
                data[data >= max_threshold] = max_threshold
                # data[data >= max_threshold] = np.nan
                
        return data



###############################################################################
###############################################################################
###############################################################################
