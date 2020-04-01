import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pylab as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']   #设置显示中文
from SEIR_model import *
from model_method import *
import warnings
warnings.filterwarnings('ignore')


## Some assumptions
China_population = 1400000000
Hubei_population = 58500000

df = pd.read_csv("使用数据.csv")
# 数据清理
df['date'] = pd.to_datetime(df['date'])
df = df[df['date'] > datetime.datetime(2019, 12, 7)]  #  2019-12-08开始
df = df[df['date'] != df['date'].max()]    # 不算今天的记录，因为他可能没有上传完成，导致数据不完整
# print(df)

# 数据准备
df['R'] = df['cured'] + df['dead']
SIR_data = df[['date', 'Days', 'countryCode','province', 'city', 'net_confirmed', 'suspected', 'R',
              ]].rename(columns={"net_confirmed": "I", "suspected": "E"})

# 2020-02-14之前的数据来训练模型，之后的数据来验证
China_df = SIR_data[SIR_data['date'] < datetime.datetime(2020, 2, 14)]
China_total = get_China_total(China_df)
China_total.tail(2)

Dynamic_SEIR1 = Train_Dynamic_SEIR(epoch = 10000, data = China_total,
                 population = 1400000000, rateAl = 1/7, rateIR=1/14, c = 1, b = -10, alpha = 0.08)

estimation_df = Dynamic_SEIR1.train()
est_beta = Dynamic_SEIR1.rateSI
est_alpha = Dynamic_SEIR1.alpha
est_b = Dynamic_SEIR1.b
est_c = Dynamic_SEIR1.c
population = Dynamic_SEIR1.numIndividuals

estimation_df.tail(2)
# print(estimation_df.tail(2))

Dynamic_SEIR1.plot_fitted_beta_R0(China_total)
Dynamic_SEIR1.plot_fitted_result(China_total)

## 将最后的观察值用作新SEIR模型的初始点
I0 = list(China_total['I'])[-1]
R0 = list(China_total['R'])[-1]
# 假设潜伏期内的个体总数是当前易感病例的4倍
E0 = list(China_total['E'])[-1] *4
S0 = population - I0 - E0 - R0

seir_new = dynamic_SEIR(eons=50, Susceptible=S0, Exposed = E0,
                    Infected=I0, Resistant=R0, rateIR=1/14,
                    rateAl = 1/7, alpha = est_alpha, c = est_c, b = est_b, past_days = China_total['Days'].max())
result = seir_new.run(death_rate = 0.02) # assume death rate is 2%
seir_new.plot_noSuscep('Dynamic SEIR for China total', 'population', 'Date', starting_point = China_total['date'].max())

"""
Calculate MAPE test score using SEIR model result
"""
test = get_China_total(SIR_data[SIR_data['date'] >= datetime.datetime(2020, 2, 14)])
plot_test_data_with_MAPE(test, result,'Infected cases prediction for China total')

