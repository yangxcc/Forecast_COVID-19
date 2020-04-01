import numpy as np
import scipy.optimize as optimization
import pandas as pd
import pandas

## 数据处理

def get_China_total(df) -> pandas.core.frame.DataFrame:
    return df[(df['countryCode'] == 'CN') & (df['province'].isnull())]
