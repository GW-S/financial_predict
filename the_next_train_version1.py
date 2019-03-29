# 逻辑回归为什么会带来这个现象
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
    label_dataSet = processerSet[each]
    # print(label_dataSet[label_dataSet.isnull()])

    value_dataSet_1 = value_dataSet[label_dataSet.values == 1]  # label_dataSet按照索引切割
    label_dataSet_1 = label_dataSet[value_dataSet_1.index]

    value_dataSet_0 = value_dataSet[label_dataSet.values == 0].sample(len(value_dataSet_1))
    label_dataSet_0 = label_dataSet[value_dataSet_0.index]

    print(len(value_dataSet_1))

    value_dataSet_list = [value_dataSet_0, value_dataSet_1]
    label_dataSet_list = [label_dataSet_0, label_dataSet_1]

    value_dataSet_temporary = pd.concat(value_dataSet_list, axis=0)
    label_dataSet_temporary = pd.concat(label_dataSet_list, axis=0)




    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(value_dataSet_temporary, label_dataSet_temporary,train_size=0.7)

    # from sklearn.linear_model import LogisticRegression
    # LR = LogisticRegression()
    # LR.fit(X_train,y_train)
    # print(each,LR.score(X_test,y_test))        # 没有划分数据集，准确率为0.81522,切分

    from tpot import TPOTClassifier

    tpot_config = {
        'sklearn.ensemble.GradientBoostingClassifier': {
            'n_estimators': [100],
            'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
            'max_depth': range(1, 11),
            'min_samples_split': range(2, 21),
            'min_samples_leaf': range(1, 21),
            'subsample': np.arange(0.05, 1.01, 0.05),
            'max_features': np.arange(0.05, 1.01, 0.05)
        },

        'sklearn.preprocessing.Binarizer': {
            'threshold': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.decomposition.FastICA': {
            'tol': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.cluster.FeatureAgglomeration': {
            'linkage': ['ward', 'complete', 'average'],
            'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine', 'precomputed']
        },

        'sklearn.preprocessing.MaxAbsScaler': {
        },

        'sklearn.preprocessing.MinMaxScaler': {
        },

        'sklearn.preprocessing.Normalizer': {
            'norm': ['l1', 'l2', 'max']
        },

        'sklearn.kernel_approximation.Nystroem': {
            'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2',
                       'sigmoid'],
            'gamma': np.arange(0.0, 1.01, 0.05),
            'n_components': range(1, 11)
        },

        'sklearn.decomposition.PCA': {
            'svd_solver': ['randomized'],
            'iterated_power': range(1, 11)
        },

        'sklearn.preprocessing.PolynomialFeatures': {
            'degree': [2],
            'include_bias': [False],
            'interaction_only': [False]
        },

        'sklearn.kernel_approximation.RBFSampler': {
            'gamma': np.arange(0.0, 1.01, 0.05)
        },

        'sklearn.preprocessing.RobustScaler': {
        },

        'sklearn.preprocessing.StandardScaler': {
        },

        'tpot.builtins.ZeroCount': {
        },

        # Selectors
        'sklearn.feature_selection.SelectFwe': {
            'alpha': np.arange(0, 0.05, 0.001),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },

        'sklearn.feature_selection.SelectPercentile': {
            'percentile': range(1, 100),
            'score_func': {
                'sklearn.feature_selection.f_classif': None
            }
        },

        'sklearn.feature_selection.VarianceThreshold': {
            'threshold': np.arange(0.05, 1.01, 0.05)
        },

        'sklearn.feature_selection.RFE': {
            'step': np.arange(0.05, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        },

        'sklearn.feature_selection.SelectFromModel': {
            'threshold': np.arange(0, 1.01, 0.05),
            'estimator': {
                'sklearn.ensemble.ExtraTreesClassifier': {
                    'n_estimators': [100],
                    'criterion': ['gini', 'entropy'],
                    'max_features': np.arange(0.05, 1.01, 0.05)
                }
            }
        }

    }
    # generations 确定子代的迭代次数
    # population_size=10 是创建个体的初始数量
    # offspring_size 每一代所需创造个体数
    # crossover_rate 用于创造后代的个体所占的百分比
    # mutation_rate 属性值随机更改的概率

    # 基于遗传算法的一个东西


    tpot = TPOTClassifier(generations=1, population_size=10, verbosity=2,
                           config_dict=tpot_config)
    tpot.fit(X_train,y_train)
    tpot.score(X_test,y_test)

    tpot.export('/Users/sheng/PycharmProjects/untitled/guowei/chishi.py')

    #tpot.score()
    # tpot.export(result.py)    导出标准的scikit-learn代码
