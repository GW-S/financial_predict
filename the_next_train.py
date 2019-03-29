import numpy as np
import pandas as pd
print('读取文件')
processerSet = pd.read_csv('guowei_processor.csv')
print(processerSet.sample(10))
print(processerSet.dtypes)
print(processerSet.describe())
# 令人窒息的操作
print(processerSet[processerSet.isnull().any(axis=1)])
processerSet.dropna(inplace=True) # 很奇妙，去掉了出错的两行
print(processerSet.describe())
# 构造模型
value_columns = [str(each)for each in range(90)]
label_columns = ['15%','-15%','10%','-10%','5%','-5%']
value_dataSet = processerSet[value_columns]
for each in label_columns:
    print(each)
    label_dataSet = processerSet[each]
    print("两种类型的比例")
    print(label_dataSet.value_counts()) # 发现可以到达0/1的差距
    value_dataSet_1 = value_dataSet[label_dataSet.values == 1]# label_dataSet按照索引切割
    label_dataSet_1 = label_dataSet[value_dataSet_1.index]
    value_dataSet_0 = value_dataSet[label_dataSet.values ==0].sample(len(value_dataSet_1))
    label_dataSet_0 = label_dataSet[value_dataSet_0.index]
    value_dataSet_list = [value_dataSet_0,value_dataSet_1]
    label_dataSet_list = [label_dataSet_0,label_dataSet_1]
    value_dataSet_temporary = pd.concat(value_dataSet_list,axis=0)
    label_dataSet_temporary = pd.concat(label_dataSet_list,axis=0)
    """进行采样"""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(value_dataSet_temporary,label_dataSet_temporary, train_size=0.7)

    # from sklearn.linear_model import LogisticRegression
    # LR = LogisticRegression()
    # LR.fit(X_train,y_train)
    # print(each,LR.score(X_test,y_test))        # 没有划分数据集，准确率为0.81522,切分
    from sklearn.ensemble import RandomForestClassifier
    LR = RandomForestClassifier()
    LR.fit(X_train,y_train)
    print(each,LR.score(X_test,y_test))
    # from sklearn.ensemble import GradientBoostingClassifier
    # LR = GradientBoostingClassifier(learning_rate=0.001, max_depth=1, max_features=0.65, min_samples_leaf=4,
    #                            min_samples_split=4, n_estimators=100, subsample=0.7)
    # LR.fit(X_train,y_train)
    # print(each,LR.score(X_test,y_test))

    # from keras.models import Sequential
    # from keras.layers import Dense, Embedding
    # from keras.layers import LSTM
    #
    # print(X_train.shape)
    # 建立序列结构
    # model = Sequential()
    # model.add(LSTM(units=2, input_shape=(90, 1)))  # recurrent——dropout 是线型变换的神经元断开比例
    # 输入：3D （samples，timesteps，input_dim）必须是3D的
    # dense 就是全连接层
    # model.add(Dense(2, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # epochs 训练次数
    # model.fit(X_train, y_train, batch_size=32, epochs=3, validation_data=(X_test,y_test))
    other0 = LR.predict_proba(value_dataSet)
    other1 = LR.predict(value_dataSet)
    other2 = pd.DataFrame(label_dataSet)
    print(other0)
    print(other1)
    output = value_dataSet
    other2["predict"] = list(other1)
    other2["predict_proba"] = list(other0)
    print(other2.columns)
    other2.to_csv(str(each)+"probability.csv")