import pandas as pd
from collections import Counter
# 读取数据
trainSet_NAV = pd.read_csv('/Users/sheng/Desktop/工作/基金/fin_data/ODS_MDS.NAV.csv', encoding='gb18030')

# 取得publishDate 和 declaredate的值，判断其缺失情况,肉眼观察，declaredate大量缺失,SYMBOL,PUBLISHDATE是主键，未有缺失
train_view = trainSet_NAV[['SYMBOL', 'PUBLISHDATE', 'NAV1']]

# 按照SYMBOL进行分组
trainSet_NAV_1 = train_view.groupby('SYMBOL')

# 对每一组，统计其PUBLISHDATE
work_data_Counter = Counter()
for name, value in trainSet_NAV_1:
    temporary = list(value['PUBLISHDATE'])
    print(temporary[-1])
    work_data_Counter.update(temporary)

# 得到counter中最长十个字典的最长长度
print(work_data_Counter.most_common(10))

work_date_list = list(work_data_Counter)
print(work_date_list)


## 用Tushare获取交易日信息


## 最开始的日期：2001-12-21 00:00:00
## 结束的日期 ：2017-03-01 00：00：00

##


