# -*- coding: utf-8 -*-
"""
# Verilerimizi derin öğrenme yapımıza göre düzenleyeceğiz
"""

import numpy as np
import pandas as pd

File = "./Covid19-Turkey.csv"
DF = pd.read_csv(File).drop(columns=['Date', 'Pneumonia', 'SeriouslyIll', 'Tests',
                                     'Recovered', 'Confirmed', 'Deaths'])
DF = DF[DF['DailyTests'] != 0]
# Data = [pd.to_datetime(DF["Date"]).values.reshape(-1, 1), DF[""].values.reshape(-1, 1),
#         DF["Confirmed"].values.reshape(-1, 1), DF["Deaths"].values.reshape(-1, 1),
#         DF["Pneumonia"].values.reshape(-1, 1), DF["SeriouslyIll"].values.reshape(-1, 1),
#         DF["Recovered"].values.reshape(-1, 1), DF["DailyTests"].values.reshape(-1, 1),
#         DF["DailyConfirmed"].values.reshape(-1, 1), DF["DailyRecovered"].values.reshape(-1, 1),
#         DF["DailyDeaths"].values.reshape(-1, 1)]

X = ['DailyTests', 'DailyConfirmed', 'DailyRecovered']

Train_X, Train_Y = DF[X].iloc[:-30,:], DF.drop(columns=X).iloc[:-30,:]
Test_X, Test_Y = DF[X].iloc[-10:,:], DF.drop(columns=X).iloc[-10:,:]
Val_X, Val_Y = DF[X].iloc[-30:-10,:], DF.drop(columns=X).iloc[-30:-10,:]

"""
# Derin Öğrenme modelimizi yükleyip doğruluk değerini ve tahmin değerlerini görütüleyeceğiz.
"""

from tensorflow import keras
from keras.layers import *

Model = models.load_model("./GreatModel")
print(Model.evaluate(Test_X, Test_Y))
print(Model.predict(Test_X))
print(Test_Y)