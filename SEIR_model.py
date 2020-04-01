import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd
import pandas
from math import *
import datetime
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')


class Train_Dynamic_SEIR:
    """
    'eons' (模型的时间点个数，default 1000)
    'Susceptible' (初始时刻易感者人数, default 950)
    'Exposed' (潜伏期的人数)
    'Infected' (初始时刻感染者的人数, default 50)
    'Resistant' (初始时刻恢复者的数量, default 0)
    'rateSI' (接触率，base rate 'beta' from S to E, default 0.05)
    'rateIR' (康复率，base rate 'gamma' from I to R, default 0.01)
    'rateAl' (感染率，base rate of isolation 'altha', from E to I, default 0.1)
    """

    def __init__(self, data: pandas.core.frame.DataFrame,
                 population: int, epoch=1000, rateIR=0.01, rateAl=0.1, c=1, b=-3, alpha=0.1):
        self.epoch = epoch  # 时期，每个时期的权重要改变
        self.steps = len(data)
        # 真实数据
        self.Exposed = list(data['E'])
        self.Infected = list(data['I'])
        self.Resistant = list(data['R'])
        self.Susceptible = list(population - data['E'] - data['I'] - data['R'])
        # 估算数据
        self.S_pre = [];
        self.E_pre = [];
        self.I_pre = [];
        self.R_pre = [];
        self.past_days = data['Days'].min()  # 计算第一个训练点之前的天数

        # 模型中的其它参数
        self.c = c;  # 计算beta公式中的常数，初始参数
        self.b = b;  # 计算beta公式中的常数der b，初始参数
        self.alpha = alpha;  # 计算beta公式中的数，初始参数
        self.rateSI = self._calculate_beta(c=self.c, t=0, b=self.b, alpha=self.alpha)  # 计算感染率beta，初始参数
        self.rateIR = rateIR
        self.rateAl = rateAl
        self.numIndividuals = population  # 全国总人口
        self.results = None
        self.estimation = None
        self.modelRun = False
        self.loss = None
        self.betalist = []

    def _calculate_beta(self, c: float, t: int, alpha: float, b: float):
        """
        根据logistics公式计算beta
        """
        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def _calculate_loss(self):
        """
        计算损失值，loss = sqrt (sum of squared loss)，使用平方损失函数
        """
        return mean_squared_error(self.Infected, self.I_pre)

    def _calculate_MAPE(self):
        """
        平均绝对百分误差
        """
        y = np.array(self.Infected)
        y_pred = np.array(self.I_pre)
        mape = np.abs((y - y_pred)) / np.abs(y)
        return np.mean(mape)

    def _update(self):
        """
        Helper function of train() function.
        尝试在迭代中使用梯度下降来查找（全局参数），计算新的迭代，然后更新参数
        """
        E = 2.71828182846
        alpha_eta = 0.000000000000001;  # learning rate
        b_eta = 0.00000000001;  # learning rate
        c_eta = 0.0000000000001;  # learning rate
        alpha_temp = 0.0;
        c_temp = 0.0;
        b_temp = 0.0;
        for t in range(0, self.steps):  # 数据文本的行数
            formula = E ** (self.alpha * (t + self.b))
            formula2 = E ** (-self.alpha * (t + self.b))

            loss_to_beta = -2 * (self.Infected[t] - self.I_pre[t]) * (self.I_pre[t]) * t * self.Susceptible[
                t] / self.numIndividuals

            # 使用链式法则计算偏导数
            beta_to_alpha = -self.c * formula * (t + self.b) * (formula - 1) * pow((1 + formula), -3)
            beta_to_b = -self.c * formula * self.alpha * (formula - 1) * pow((1 + formula), -3)
            beta_to_c = formula2 * pow((1 + formula2), -2)

            alpha_temp += loss_to_beta * beta_to_alpha  # new gradient
            b_temp += loss_to_beta * beta_to_b  # new gradient
            c_temp += loss_to_beta * beta_to_c  # new gradient

        self.alpha -= alpha_eta * alpha_temp;  # update values
        self.b -= b_eta * b_temp;
        self.c -= c_eta * c_temp;

    def train(self):
        """
        使用真实数据带入SEIR模型进行估算
        通过时间（epoch）迭代不断调整参数

        训练目的：使用梯度下降通过最小损失函数找到最佳beta（接触率）

        梯度下降法:
            为了解决梯度，我们使用新的alpha，c和𝑏值迭代数据点并计算偏导数。
           新的梯度告诉我们成本函数在当前位置（当前参数值）的斜率以及更新参数的方向。
      我们更新的大小由学习率控制。 （请参见上面的_update（）函数）
        """
        for e in range(self.epoch):
            # prediction list
            self.S_pre = [];
            self.E_pre = [];
            self.I_pre = [];
            self.R_pre = [];

            for t in range(0, self.steps):
                if t == 0:
                    self.S_pre.append(self.Susceptible[0])
                    self.E_pre.append(self.Exposed[0])
                    self.I_pre.append(self.Infected[0])
                    self.R_pre.append(self.Resistant[0])
                    self.rateSI = self._calculate_beta(c=self.c, t=t, b=self.b,
                                                       alpha=self.alpha)
                    # print("time {}, beta {}".format(t, self.rateSI))

                    # collect the optimal fitted beta
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSI)

                else:
                    self.rateSI = self._calculate_beta(c=self.c, t=t, b=self.b,
                                                       alpha=self.alpha)
                    # print("time {}, beta {}".format(t, self.rateSI))

                    # collect the optimal fitted beta
                    if e == (self.epoch - 1):
                        self.betalist.append(self.rateSI)

                    # 将真实数据应用于SEIR公式，计算出各种状态的人的数量
                    S_to_E = (self.rateSI * self.Susceptible[t] * self.Infected[t]) / self.numIndividuals
                    E_to_I = (self.rateAl * self.Exposed[t])
                    I_to_R = (self.Infected[t] * self.rateIR)
                    self.S_pre.append(self.Susceptible[t] - S_to_E)
                    self.E_pre.append(self.Exposed[t] + S_to_E - E_to_I)
                    self.I_pre.append(self.Infected[t] + E_to_I - I_to_R)
                    self.R_pre.append(self.Resistant[t] + I_to_R)

            # 记录最后一次迭代时的估计值
            if e == (self.epoch - 1):
                self.estimation = pd.DataFrame.from_dict({'Time': list(range(len(self.Susceptible))),
                                                          'Estimated_Susceptible': self.S_pre,
                                                          'Estimated_Exposed': self.E_pre,
                                                          'Estimated_Infected': self.I_pre,
                                                          'Estimated_Resistant': self.R_pre},
                                                         orient='index').transpose()
                self.loss = self._calculate_loss()
                MAPE = self._calculate_MAPE()
                print("The loss in is {}".format(self.loss))
                print("The MAPE in the whole period is {}".format(MAPE))
                # print("Optimial beta is {}".format(self.rateSI))

            ## calculate loss in each iteration
            self.loss = self._calculate_loss()

            # print("The loss in iteration {} is {}".format(e, self.loss))
            # print("Current beta is {}".format(self.rateSI))

            ## ML optimization.
            self._update()  # Update parameters using Gradient Descent in each step

        return self.estimation  # the lastest estimation

     # 计算基本可再生人数
    def plot_fitted_beta_R0(self, real_obs: pandas.core.frame.DataFrame):
        fig, ax = plt.subplots(figsize=(15, 6))
        plt.plot(self.estimation['Time'], self.betalist, color='green')
        Rlist = [x / self.rateIR for x in self.betalist]  # beta随时间的变化而变化，因此R0也会变化，接触率/康复率
        plt.plot(self.estimation['Time'], Rlist, color='blue')

        # 设置x轴
        datemin = real_obs['date'].min()
        numdays = len(real_obs)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=90, fontsize=15)
        plt.xlabel('2020Date')  # 2020 Date
        plt.ylabel('Rate')  # Rate
        plt.yticks(fontsize=15)
        plt.title('Fitted Dynamic Contact Rate and Transmissibility of COVID-19 over time', fontsize=18)
        plt.legend(['Transmissibility', 'Contact Rate'], prop={'size': 16}, bbox_to_anchor=(0.5, 1.02),
                   ncol=2, fancybox=True, shadow=True)   # 加上图例
        plt.show()

    def plot_fitted_result(self, real_obs: pandas.core.frame.DataFrame):
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.plot(self.estimation['Time'], self.estimation['Estimated_Infected'], color='green')  # 预测值
        plt.plot(self.estimation['Time'], real_obs['I'], color='y')                              # 真实值
        plt.plot(self.estimation['Time'], self.estimation['Estimated_Exposed'], color='blue')
        plt.plot(self.estimation['Time'], real_obs['E'], color='royalblue')

        # set x tricks
        datemin = real_obs['date'].min()
        numdays = len(real_obs)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=90, fontsize=10)
        plt.xlabel('2020 Date')
        plt.ylabel('Population')
        plt.title('Fitted value by Dynamic SEIR model', fontsize=20)
        plt.legend(['Estimated Infected', 'Real Infected', 'Estimated Exposed', 'Real Exposed'], prop={'size': 12}, bbox_to_anchor=(0.5, 1.02),
                   ncol=4, fancybox=True, shadow=True)
        plt.show()


class dynamic_SEIR:
    """
    参数说明见class Train_Dynamic_SEIR
    # 添加 rateIR1,加入了死亡率
    """

    def __init__(self, eons=1000, Susceptible=950, Exposed=100, Infected=50, Resistant=0, rateIR=0.01, rateAl=0.1,
                 alpha=0.3, c=5, b=-10, past_days=30):
        self.eons = eons  # number of prediction days
        self.Susceptible = Susceptible
        self.Exposed = Exposed
        self.Infected = Infected
        self.Resistant = Resistant
        self.rateSI = None
        self.rateIR = rateIR
        ###
        self.rateAl = rateAl
        self.numIndividuals = Susceptible + Infected + Resistant + Exposed  # total population
        self.alpha = alpha
        self.c = c
        self.b = b
        self.past_days = past_days  # make prediction since the last observation
        self.results = None
        self.modelRun = False

    def _calculate_beta(self, c: float, t: int, alpha: float, b: float, past_days: int):
        t = t + past_days
        return c * exp(-alpha * (t + b)) * pow((1 + exp(-alpha * (t + b))), -2)

    def run(self, death_rate):
        Susceptible = [self.Susceptible]
        Exposed = [self.Exposed]
        Infected = [self.Infected]
        Resistant = [self.Resistant]

        for i in range(1, self.eons):  # 预测的天数
            self.rateSI = self._calculate_beta(c=self.c, t=i, b=self.b,
                                               alpha=self.alpha, past_days=self.past_days)

            # print(self.rateSI)
            # 各种状态的人数
            S_to_E = (self.rateSI * Susceptible[-1] * Infected[-1]) / self.numIndividuals
            E_to_I = (self.rateAl * Exposed[-1])
            # print(Exposed[-1])
            I_to_R = (Infected[-1] * self.rateIR)
            Susceptible.append(Susceptible[-1])
            Exposed.append(Exposed[-1] + S_to_E - E_to_I)
            Infected.append(Infected[-1] + E_to_I - I_to_R )
            Resistant.append(Resistant[-1] + I_to_R)

        # 死亡率*感染人数
        Death = list(map(lambda x: (x * death_rate), Infected))
        # 治愈=移除-死亡
        Heal = list(map(lambda x: (x * (1 - death_rate)), Resistant))

        self.results = pd.DataFrame.from_dict({'Time': list(range(len(Susceptible))),
                                               'Susceptible': Susceptible, 'Exposed': Exposed, 'Infected': Infected,
                                               'Resistant': Resistant,
                                               'Death': Death, 'Heal': Heal},
                                              orient='index').transpose()
        self.modelRun = True
        return self.results

    def plot(self, title, ylabel, xlabel, starting_point):
        if self.modelRun == False:
            print('Error: Model has not run. Please call SIR.run()')
            return
        print("Maximum infected case: ",
              format(int(max(self.results['Infected']))))
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(self.results['Time'], self.results['Susceptible'], color='blue')
        plt.plot(self.results['Time'], self.results['Infected'], color='red')
        plt.plot(self.results['Time'], self.results['Exposed'], color='orange')
        plt.plot(self.results['Time'], self.results['Resistant'], color='palegreen')
        plt.plot(self.results['Time'], self.results['Heal'], color='green')
        plt.plot(self.results['Time'], self.results['Death'], color='grey')
        # set x trick
        datemin = starting_point
        numdays = len(self.results)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=90)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(['Susceptible', 'Infected', 'Exposed', 'Removed', 'Heal', 'Death'], prop={'size': 15},
                   bbox_to_anchor=(0.5, 1.02), ncol=6, fancybox=True, shadow=True)
        plt.title(title, fontsize=20)
        plt.show()

    def plot_noSuscep(self, title, ylabel, xlabel, starting_point):
        if self.modelRun == False:
            print('Error: Model has not run. Please call SIR.run()')
            return
        print("Maximum infected case: ",
              format(int(max(self.results['Infected']))))
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(self.results['Time'], self.results['Infected'], color='red')
        plt.plot(self.results['Time'], self.results['Resistant'], color='palegreen')
        plt.plot(self.results['Time'], self.results['Exposed'], color='orange')
        plt.plot(self.results['Time'], self.results['Heal'], color='green')
        plt.plot(self.results['Time'], self.results['Death'], color='grey')
        # set x trick
        datemin = starting_point
        numdays = len(self.results)
        labels = list((datemin + datetime.timedelta(days=x)).strftime('%m-%d') for x in range(numdays))
        plt.xticks(list(range(numdays)), labels, rotation=60)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(['Infected', 'Removed', 'Exposed', 'Heal', 'Death'], prop={'size': 12}, bbox_to_anchor=(0.5, 1.02),
                   ncol=5, fancybox=True, shadow=True)
        plt.title(title, fontsize=20)
        plt.show()




## 灵敏度分析
def plot_test_data_with_MAPE(test, predict_data, title):
    y = test["I"].reset_index(drop=True)
    y_pred = predict_data[:len(test)]['Infected'].reset_index(drop=True)
    mape = np.mean(np.abs((y - y_pred)) / np.abs(y))
    print("The MAMPE is: ".format(mape))
    print(mape)

    fig, ax = plt.subplots(figsize=(15, 6))
    plt.plot(test['date'], y, color='steelblue')
    plt.plot(test['date'], y_pred, color='orangered')

    plt.xlabel('2020 Date')
    plt.ylabel('Infected case')
    plt.title(title, fontsize=20)
    plt.legend(['Observation', 'Prediction'], loc='upper left', prop={'size': 12},
               bbox_to_anchor=(0.5, 1.02), ncol=2, fancybox=True, shadow=True)
    plt.show()