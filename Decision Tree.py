
import pandas as pd
import numpy as np
import json

# 路径需要对应自己的文件路径，编码方式用notepad++打开文件，右下角能看到
df = pd.DataFrame(pd.read_csv("E:/1/机器学习/西瓜数据集3.0.csv", encoding="ansi"))
df.drop(labels=["编号"], axis=1, inplace=True)  # 删除编号这一列，inplace=True表示直接在原对象修改
df["好瓜"].replace(to_replace=["是", "否"], value=["好瓜", "坏瓜"], inplace=True)  # 替换，这一步不是必须的
featureList = df.columns[:-1]  # 除了好瓜属性之外的其他属性
featureValue = {}
for feature in featureList[:-2]:  # 这部分是离散属性
    featureValue[feature] = set(df[feature])  # 各个属性及其出现过的取值，如色泽: (青绿，乌黑，浅白)
# print(featureValue)

T = {}  # 候选点集合
for feature in featureList[-2:]:  # 连续属性
    T1 = df[feature].sort_values()
    T2 = T1[:-1].reset_index(drop=True)  # 重新设置索引
    # T2和T3的索引都从原来的T1继承得来，T2的第一个元素的索引是0，T3的第一个元素的索引是1，要想让它们相加，必须要索引一致
    T3 = T1[1:].reset_index(drop=True)  
    T[feature] = (T2+T3)/2
# print(T["密度"])


def Ent(D):  # 集合D的好瓜属性的信息熵
    """
    @ return: return the entropy of D
    """
    # D[["好瓜"]]是DataFrame D["好瓜"]是Series
    # DataFrame类没有value_counts()方法，所以用D["好瓜"]
    # value_counts()得到各个元素的出现次数，除以长度得到频率
    frequency = D["好瓜"].value_counts()/len(D["好瓜"])
    # print(frequency)
    return -sum(pk*np.log2(pk) for pk in frequency)  # 按照熵的定义进行计算即可
# print(Ent(df))
    

def split_discrete(D, feature):  # 按照指定feature把D划分为几个子集
    """
    @ return: subset of D
    """
    splitD = []
    # groupby()得到的结果是(feature, group)的tuple
    # 其中Dv[0]是feature Dv[1]group是子集，属于DataFrame类型
    for Dv in D.groupby(by=feature, axis=0):
        splitD.append(Dv)
    return splitD
# splitD = split_discrete(df, "好瓜")
# print(splitD)
# splitD = split_discrete(df, ["色泽", "根蒂"])
# print(splitD)


def split_continues(D, feature, splitValue):
    """
    @ D: type: Dataframe
    @ return: subset of D
    """
    splitD = []
    # 以splitValue为划分点把D分成两部分
    # D[feature] <= splitValue的比较结果是多个True和False
    # D[D[feature] <= splitValue]把feature属性上小于等于splitValue的元素选出来
    splitD.append(D[D[feature] <= splitValue])
    splitD.append(D[D[feature] > splitValue])
    return splitD
# print(split_continues(df, "密度", 0.381))


def Gain_discrete(D, feature):
    # 按照定义进行计算，注意Dv是(feature, group)元组，Dv[1]才是相应的子集
    gain = Ent(D) - sum(len(Dv[1])/len(D)*Ent(Dv[1]) for Dv in split_discrete(D, feature))
    return gain


def Gain_continues(D, feature):
    """
    @ return: _max最大增益 splitValue对应划分点(划分值)
    """
    _max = 0
    splitValue = 0
    # T[feature]的元素本质上是一个键值对 for t in T[feature]取到的是key而不是value
    # 加.values迭代器取的是对应元素，不加.values的话，取到的是索引值，也就是0123456
    for t in T[feature].values:  # 尝试各个划分点，并取可以使增益最大的划分点
        temp = Ent(D) - sum(len(Dv)/len(D)*Ent(Dv) for Dv in split_continues(D, feature, t))
        if _max < temp:
            _max = temp
            splitValue = t
    
    return _max, splitValue


def chooseBestFeature(D, A):
    informationGain = {}
    for feature in A:
        if feature in ["密度", "含糖率"]:  # 密度和含糖率是连续属性
            ig, splitValue = Gain_continues(D, feature)
            informationGain[feature+"<=%.3f"%splitValue] = ig
        else:
            informationGain[feature] = Gain_discrete(D, feature)
    # print(informationGain)
    
    # informationGain的元素是(feature: 对应的信息增益)键值对，下面是按照信息增益排序的写法
    informationGain = sorted(informationGain.items(), key=lambda ig:ig[1], reverse=True)
    # 返回对应信息增益
    return informationGain[0][0]


def countMajority(D):  # mode()求出现次数最多的元素，iloc取得对应的类：好瓜或坏瓜（是或否）
    # print(D["好瓜"].mode())  
    return D["好瓜"].mode().iloc[0]
# print(countMajority(df))


def treeGenerate(D, A):
    # 按照好瓜属性分组，结果只有一个组，也就是 全都是好瓜或全都是坏瓜
    # 表明已经到了叶子节点，返回判断结果
    if len(split_discrete(D, "好瓜")) == 1: 
        return D["好瓜"].iloc[0]
    # 属性集合A为空，或按照A(可能包含不止一个属性)分组只有一个组(所有样本在A上的取值相同)，也是到达叶子节点
    # 返回D里数量最多的类型
    if len(A) == 0 or len(split_discrete(D, A.tolist())) == 1:
        return countMajority(D)
    
    # 选择信息增益最大的属性
    bestFeature = chooseBestFeature(D, A)
    # print("best feature:", bestFeature)
    if "<=" in bestFeature:  # 连续属性
        bestFeature, splitValue = bestFeature.split("<=")
        myTree = {bestFeature+"<="+splitValue:{}}
        [D0, D1] = split_continues(D, bestFeature, float(splitValue))
        # 因为A的类别是Index，所以这里直接用Index()复制一份
        # 连续属性在之后的划分还可以继续使用，所以不需要去掉
        A0 = pd.Index(A)
        A1 = pd.Index(A)
        myTree[bestFeature+"<="+splitValue]["yes"] = treeGenerate(D0, A0)
        myTree[bestFeature+"<="+splitValue]["no"] = treeGenerate(D1, A1)
    else:  # discrete
        myTree = {bestFeature:{}}
        for bestFeatureValue, Dv in split_discrete(D, bestFeature):
            # 在样本集里，bestFeature属性上已经没有bestFeatureValue这一取值
            # 但在实际情况中，是可能还会有bestFeatureValue这一取值的，所以把它们分为D里数量最多的类型
            if len(Dv) == 0:
                return countMajority(D)
            else:
                A2 = pd.Index(A)
                # 离散属性，因为之后的划分不再需要该属性，所以去掉
                A2 = A2.drop([bestFeature])
                Dv = Dv.drop(labels=[bestFeature], axis=1)
                myTree[bestFeature][bestFeatureValue] = treeGenerate(Dv, A2)
    return myTree


if __name__ == "__main__":
    myTree = treeGenerate(df, featureList)
    myTree = json.dumps(myTree, indent=2, ensure_ascii=False, separators=(',', ':'))  # 这里做个格式化，只为了容易看
    print(myTree)
