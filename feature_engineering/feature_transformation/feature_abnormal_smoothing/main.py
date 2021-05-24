import run as r


def main(fileName):
    return r.run(fileName)
'''
删除低方差的特征
    VarianceThreshold

单变量特征选择
SelectKBest 删除除  最高评分特征

SelectPercentile 删除除用户指定的最高得分百分比以外的所有特征

对每个特征使用通用的单变量统计检验：误报率SelectFpr，误发现率 SelectFdr或家庭错误SelectFwe。

GenericUnivariateSelect允许使用可配置的策略执行单变量特征选择。这允许使用超参数搜索估计器选择最佳的单变量选择策略。

对于回归：f_regression，mutual_info_regression

对于分类：chi2，f_classif，mutual_info_classif



递归特征消除
RFE RFECV

使用SelectFromModel特征选择
SelectFromModel
'''