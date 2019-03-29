"""
15% 0.796082949309
-15% 0.853686635945
10% 0.743087557604
-10% 0.735023041475
5% 0.626728110599
-5% 0.676267281106

15% 0.820276497696
-15% 0.832949308756
10% 0.744239631336
-10% 0.758064516129
5% 0.599078341014
-5% 0.647465437788

# 进行了缺失值添补之后
15% 0.801614763552
-15% 0.865051903114
10% 0.757785467128
-10% 0.772779700115
5% 0.617070357555
-5% 0.668973471742

15% 0.81199538639
-15% 0.855824682814
10% 0.763552479815
-10% 0.77047289504
5% 0.647058823529
-5% 0.644752018454
"""
import pandas as pd
from datetime import timedelta
from datetime import datetime
# 进行缺失值的添补
def giveback_value(each_index, time_Series, mean):
    """
    each_index 指的是当前的时间
    :param each_index:
    :return:
    """
    # 判断是否是工作日，如果是星期天，我认为其是星期五的值，如果找不到，就用平均值，# TODO：这样会不会好一点，缺失值用附近几天的窗口值来平均
    this_day = pd.to_datetime(each_index)
    if this_day.weekday() == 5:  # 寻找星期六
        if (this_day - timedelta(1)) in time_Series.index:  # 在总序列中寻找时间，如果找到，填补，找不到，填补平均值
            return time_Series[(this_day - timedelta(1))]
        else:
            return mean
    if this_day.weekday() == 6:  # 寻找星期日
        if (this_day - timedelta(2)) in time_Series.index:  # 在总序列中寻找时间，如果找到，填补，找不到，填补平均值
            return time_Series[(this_day - timedelta(2))]
        else:
            return mean
    return mean
# 读取数据
trainSet_NAV = pd.read_csv('/Users/sheng/Desktop/基金/fin_data/ODS_MDS.NAV.csv', encoding='gb18030')
# 取得publishDate 和 declaredate的值，判断其缺失情况,肉眼观察，declaredate大量缺失,SYMBOL,PUBLISHDATE是主键，未有大量缺失
train_view = trainSet_NAV[['SYMBOL', 'PUBLISHDATE', 'NAV1']]
## 由于有很多种基金，所以，我们选择某一种
#  取得代码号为1的列
trainSet_NAV_1 = train_view[train_view['SYMBOL'] == 1]
# 构造时间序列，索引为PUBLISHDATE_index
time_Series = trainSet_NAV_1['NAV1']
temporary = list(trainSet_NAV_1['PUBLISHDATE'])
PUBLISHDATE_index = [pd.to_datetime(each) for each in temporary]
time_Series.index = PUBLISHDATE_index
# 对索引进行索引，调整成适合的时间序列
time_Series = time_Series.sort_index()
# todo 对一天有两个时间点我应该怎么处理呢？# 看看时间序列有没有重复的,我们发现是有重复的，也就是一天可能有两个数据
# 选取工作日的前90天
# 净值转化为相对量
# 选取前90天的工作日
print(time_Series)
# todo:工作日是工作日，缺失值是缺失值，对工作日的缺失值，我选择处理，对非工作日的缺失值，取得前几天的净值，我选择忽略，忽略了怎么确定90天呢，90天净值不好确定
# 如果是02-15日，是星期一，是工作日，我就向前搜索两日，以其值为净值，如果没有视为缺失值，在测试symbol1例子中，是没有的
# 如果是2016-02-01是有缺失值的，我以其值为净值
print("对于记录构造向量")
front = timedelta(90)
behind = timedelta(30)
dict_label_store = []  # label的存储字典
dict_store = []  # 前90天的输入值的存储字典
for now in time_Series.index:  # 大概3000左右条记录
    # print(now)
    now = pd.to_datetime(now)
    # 保证now的前面有90天，now后有30天
    if (now - front) < time_Series.index[0] or (now + behind) > time_Series.index[-1]:
        continue
    # 筛选在这120天中的时间序列
    time_Series1 = time_Series[time_Series.index > (now - front)]
    time_Series2 = time_Series1[time_Series1.index < (now + behind)]
    # 筛选出在90天中的时间序列
    time_Series3 = time_Series1[time_Series1.index < now]
    # 计算time_series的均值，作为异常值添补，放在这里是为了减少计算次数，是前90天的异常值进行添补
    # TODO sheng:用90天的数据进行添补，还是用全部的进行缺失值添补，我认为应该用90天的数据进行缺失值添补。
    mean = time_Series3.mean()
    # 构建最后的字典
    this_dict = {}  # 存储此时的数据
    for item in range(90):
        this_dict[str(item)] = 0
    this_dict_label = {}  # 存储超脱点的数据
    this_dict_label['5%'] = 0
    this_dict_label['10%'] = 0
    this_dict_label['15%'] = 0
    this_dict_label['-5%'] = 0
    this_dict_label['-10%'] = 0
    this_dict_label['-15%'] = 0
    # 对于当前的now，判断其前90天的一天是否在索引中，如果在就存入dict中，，如果不是,则进行缺失值添补，因为有限保证在的值能填充上去，所以，不但心其它
    for each in range(90):
        time_front = timedelta(each)
        # 判断是否在索引中
        if (now - time_front) in time_Series.index:
            this_dict[str(each)] = time_Series[now - time_front]
        else:
            this_dict[str(each)] = giveback_value(now - time_front, time_Series, mean) / time_Series[now]
    # 计算后30位中的label
    # 首先判断后30天的某一天是否在索引中，如果在，记录该天的数据，并计算该天是否值得打标点，也就是超过标记点
    for each in range(90, 120):
        time_behind = timedelta(each)
        if now + time_behind in time_Series.index:
            # 判断是否能成为超凡出种的点，实质上这样我可以这样处理，对超越点的所有情况进行统计
            now_value_behind = time_Series[now + time_behind]
            now_value = time_Series[now]
            if (now_value_behind - now_value) / now_value < -0.15:
                this_dict_label['-15%'] = 1
            if (now_value_behind - now_value) / now_value > 0.15:
                this_dict_label['15%'] = 1
            if (now_value_behind - now_value) / now_value < -0.1:
                this_dict_label['-10%'] = 1
            if (now_value_behind - now_value) / now_value > 0.1:
                this_dict_label['10%'] = 1
            if (now_value_behind - now_value) / now_value < -0.05:
                this_dict_label['-5%'] = 1
            if (now_value_behind - now_value) / now_value > 0.05:
                this_dict_label['5%'] = 1
    dict_label_store.append(this_dict_label)
    dict_store.append(this_dict)
print('开始进行pandas_DataFrame的构建')
label_set = pd.DataFrame()
for each_dict_label_index in range(len(dict_label_store)):
    pd_dict = dict_label_store[each_dict_label_index]
    pd_dict.update(dict_store[each_dict_label_index])
    label_set = label_set.append(pd.DataFrame(pd_dict, index=[each_dict_label_index]))
label_set = label_set.reset_index()
label_set.to_csv('guowei_processor.csv')