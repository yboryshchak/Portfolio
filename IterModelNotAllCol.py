import os
import math
import numpy as np
from numpy import array, asarray, save
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Dropout, concatenate
from tensorflow.python.keras.utils.vis_utils import plot_model

tf.distribute.MirroredStrategy()
tf.config.threading.get_inter_op_parallelism_threads()

Zoom = 20
def MyActivation(x):
    y = K.switch(x>0, 2*x, 0.1*x)
    return y

def Rounded100_func(column_list):
    new_array = [x//100 for x in column_list]
    return new_array

def Rounded1000_func(column_list):
    new_array = [x//1000 for x in column_list]
    return new_array

data_dir = r"D:\HousePriceGitHub\HousePriceFiles"
bch = 200000
Tsize = 0.05
mdelta = 128
learnrate = 0.0004
ep = 4000
pat = 1000

w0 = 0.1
w1 = 1

train_df = pd.read_csv(r"D:\HousePriceGitHub\HousePriceFiles\train.csv",  low_memory=False, nrows=500, index_col='Id')
train_cols = list(train_df.columns)

float_columns = []
num_columns = []
str_columns = []
for col in train_cols:
    if (train_df[col].dtype==np.float64 or train_df[col].dtype==np.int64):
        num_columns.append(col)
        float_columns.append(col)
    else:
        str_columns.append(col)

numlen = len(num_columns)
strlen = len(str_columns)

categ_columns = ['MSZoning', 'BldgType', 'Neighborhood', 'SaleType', 'SaleCondition', 'Condition1', 'Condition2', 'ExterQual', 'ExterCond', 'Foundation', 'KitchenQual', 'GarageType']
num_na = [0 for x in range(numlen)]
str_na = ['Unknown' for x in range(strlen)]
num_types = ['float32' for x in range(numlen)]
str_types = ['str' for x in range(strlen)]

num_columns.extend(str_columns)
num_na.extend(str_na)
num_types.extend(str_types)

dtypes_dict = dict(zip(str_columns, str_types))
na_dict = dict(zip(num_columns, num_na))

alltrain_df = pd.read_csv(r"D:\HousePriceGitHub\HousePriceFiles\train.csv",  low_memory=False, dtype=dtypes_dict, index_col='Id')
alltrain_df.fillna(na_dict, inplace=True)
Sprice_list = alltrain_df['SalePrice'].to_list()
Fprice_list = [x/100 for x in Sprice_list]
Kprice_list =  [x/1000 for x in Sprice_list]
alltrain_df['PriceK'] = Rounded1000_func(Sprice_list)
alltrain_df['PriceFranklins'] = Rounded100_func(Sprice_list)
alltrain_df['SalesYear'] = alltrain_df['YrSold']-2005
NumTrain_df = alltrain_df[float_columns]
NumTrain_df.drop(['SalePrice', 'YrSold']  , axis=1, inplace=True)
NumTrain_df.astype('float32')
CatTrain_df = alltrain_df[categ_columns]

enc = OrdinalEncoder()
CatTrain_df = enc.fit_transform(CatTrain_df)

X0 = NumTrain_df.values
Xnp_0 = NumTrain_df.to_numpy() 
X1 = CatTrain_df
X2 = alltrain_df[['SalesYear']].values
Xnp_2 = alltrain_df['SalesYear']
y0 = alltrain_df[['PriceFranklins']].values
y1 = alltrain_df['PriceK'].values

D0 = len(X0[0])
D1 = len(X1[0])
D2 = len(X2[0])
Dmerge = D0+D1
Dout = Dmerge + 1

X0_train, X0_test, X1_train, X1_test, X2_train, X2_test, y0_train, y0_test, y1_train, y1_test = train_test_split(X0, X1, X2, y0, y1, test_size=Tsize)
earlstop = EarlyStopping(monitor = 'loss', mode='min', min_delta=mdelta, patience=pat)

# Numerical Input
input_0 = tf.keras.Input(shape=(D0, ))
layer0_x = Dense(D0, activation="elu")(input_0)
layer0_x = Dense(D0,  activation="elu")(layer0_x)
layer0_x = Dense(D0,  activation=MyActivation)(layer0_x)
# Categorical Input
input_1 = tf.keras.Input(shape=(D1, ))
layer1_x = Dense(D1, activation="elu")(input_1)
layer1_x = Dense(D1,  activation="elu")(layer1_x)
layer1_x = Dense(D1,  activation=MyActivation)(layer1_x)
# Merging Two Inputs
merge0 = concatenate([layer0_x, layer1_x])
comblayer_m =  Dense(Dmerge,  activation="elu")(merge0)
comblayer_m =  Dense(Dmerge,  activation="elu")(comblayer_m)
# Merging with YearSold
input_2 = tf.keras.Input(shape=(D2, ))
merge1 = concatenate([comblayer_m, input_2])
tcomblayer =  Dense(Dout, activation="elu")(merge1)
#  Output_0 group
outlayer_0 =  Dense(Dout, activation="elu")(tcomblayer)
outlayer_0 =  Dense(Dout, activation="elu")(outlayer_0)
output_0 = Dense(1, activation="elu")(outlayer_0)
#  Output_1 group
outlayer_1 =  Dense(Dout, activation="elu")(tcomblayer)
outlayer_1 =  Dense(Dout, activation="elu")(outlayer_1)
output_1 = Dense(1, activation="elu")(outlayer_1)
# Compile Model
func_model =  Model(inputs=[input_0, input_1, input_2], outputs=[output_0, output_1])   
func_model.compile(optimizer=Adam(learning_rate = learnrate), loss=['mse', 'mse'], loss_weights=[w0, w1])   
history = func_model.fit(x=[X0_train, X1_train, X2_train], y=[y0_train, y1_train], epochs=ep, batch_size=bch ,  callbacks=[earlstop])
test_loss = func_model.evaluate([X0_test, X1_test, X2_test],  [y0_test, y1_test])
func_model.save(r"D:\HousePriceGitHub\SavedModels\NotAllColModel.h5")   

fig = plt.figure(figsize=(12,8))
plt.plot(history.history['loss'],label='training loss')
plt.legend(loc=0)
plt.xlabel('epochs')
plt.ylabel('losses on dataset')
plt.grid(True)
plt.title("Training and validation loss")
plt.show()
plt.close(fig)


learnrate = 0.0001
ep = 8000
Err0_list = []
Err1_list = []
lrate_list = []
for num in range(0, 4):
    mdelta = 4**(3-num)
    X0_train, X0_test, X1_train, X1_test, X2_train, X2_test, y0_train, y0_test, y1_train, y1_test = train_test_split(X0, X1, X2, y0, y1, test_size=Tsize)
    earlstop = EarlyStopping(monitor = 'loss', mode='min', min_delta=mdelta, patience=pat)
    func_model = tf.keras.models.load_model(r"D:\HousePriceGitHub\SavedModels\NotAllColModel.h5", custom_objects={'MyActivation': MyActivation}, compile=False)
    func_model.compile(optimizer=Adam(learning_rate = learnrate), loss=['mse', 'mse'], loss_weights=[w0, w1]) 
    history = func_model.fit(x=[X0_train, X1_train, X2_train], y=[y0_train, y1_train], epochs=ep, batch_size=bch ,  callbacks=[earlstop])
    func_model.save(r"D:\HousePriceGitHub\SavedModels\NotAllColModel.h5")
    hisloss = history.history['loss']
    picname = "NotAllCol_iter" +str(num+1)+".jpg"
    picpath =  os.path.join(data_dir, picname)
    fig = plt.figure(figsize=(12,8))
    plt.plot(history.history['loss'],label='training loss')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('losses on dataset')
    plt.grid(True)
    plt.title("Training and validation loss")
    plt.show()
    fig.savefig(picpath)
    plt.close(fig)
    predictions = func_model.predict([Xnp_0, X1, X2])
    Pred_list0 = [x[0] for x in predictions[0]]
    Pred_list1 = [x[0] for x in predictions[1]]
    Err_list0 = [abs(x-y) for (x,y) in zip(Pred_list0, Fprice_list)]
    Err_list1 = [abs(x-y) for (x,y) in zip(Pred_list1, Kprice_list)]
    ErrAve0 = 100*np.average(Err_list0)
    ErrAve1 = 1000*np.average(Err_list1)
    Err0_list.append(ErrAve0)
    Err1_list.append(ErrAve1)
    lrate_list.append(learnrate)
    learnrate = learnrate/4
    ep = 2*ep

Errors_df = pd.DataFrame({'LearnRate': lrate_list, 'AveError_0': Err0_list, 'AveError_1': Err1_list})
print(Errors_df)
Errors_df.to_csv(r"D:\HousePriceGitHub\NotAllColModelErrors.csv", index=False)












