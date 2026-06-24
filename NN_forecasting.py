'''
Neural Networks models for time series forecasting
'''

from NN_train import train, predict, predict_iteration,load_model, save_model
from src.util import *
from sklearn.preprocessing import MinMaxScaler
import eval
from src.ts_decompose import ts_decompose
import pandas as pd
import numpy as np
from TSFE import TSFE, init_net

# forecasting using neural networks, support multi-step ahead forecasting
def single_model_forecasting_trend(data, lag, h_train, h_test, lr, epoch, batch_size, hidden_num, method, use_cuda, iteration):

    # normalize time series
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(data)

    trainData, testData = divideTrainTest(dataset)

    flag = True  # using RNN format or not
    trainX, trainY = create_multi_ahead_samples(trainData, lag, h_train, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lag, h_test, RNN=flag)
    trainY = np.squeeze(trainY).reshape(-1, h_train)
    testY = np.squeeze(testY).reshape(-1, h_test)
    print("train X shape:", trainX.shape)
    print("train y shape:", trainY.shape)
    print("test X shape:", testX.shape)
    print("test y shape:", testY.shape)

    # 训练模型
    net = train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batch_size,
                lag=lag, method=method, hidden_num=hidden_num, use_cuda=use_cuda)
    save_model(net, f"102_Model/trained_model_trend_{iteration}.pth")

    # # 调用模型
    # net = load_model(path=f"102_Model/trained_model_trend_{iteration}.pth", hidden_num=10, use_cuda=True)

    testPred = predict_iteration(net, testX, h_test, use_cuda=use_cuda, RNN=flag)

    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluation
    R = eval.calcR(testY, testPred)
    print("test R", R)

    RMSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", RMSE)

    return testPred, testY

def single_model_forecasting_season(data, lag, h_train, h_test, lr, epoch, batch_size, hidden_num, method, use_cuda, iteration):

    # normalize time series
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(data)

    trainData, testData = divideTrainTest(dataset)

    flag = True  # using RNN format or not
    trainX, trainY = create_multi_ahead_samples(trainData, lag, h_train, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lag, h_test, RNN=flag)
    trainY = np.squeeze(trainY).reshape(-1, h_train)
    testY = np.squeeze(testY).reshape(-1, h_test)
    print("train X shape:", trainX.shape)
    print("train y shape:", trainY.shape)
    print("test X shape:", testX.shape)
    print("test y shape:", testY.shape)

    # 训练模型
    net = train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batch_size,
                lag=lag, method=method, hidden_num=hidden_num, use_cuda=use_cuda)
    save_model(net, f"102_Model/trained_model_season_{iteration}.pth")

    # # 调用模型
    # net = load_model(path=f"102_Model/trained_model_season_{iteration}.pth", hidden_num=10, use_cuda=True)

    testPred = predict_iteration(net, testX, h_test, use_cuda=use_cuda, RNN=flag)

    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluation
    R = eval.calcR(testY, testPred)
    print("test R", R)

    RMSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", RMSE)

    return testY, testY

def single_model_forecasting_res(data, lag, h_train, h_test, lr, epoch, batch_size, hidden_num, method, use_cuda, iteration):

    # normalize time series
    scaler = MinMaxScaler(feature_range=(-1, 1))
    dataset = scaler.fit_transform(data)

    trainData, testData = divideTrainTest(dataset)

    flag = False  # using RNN format or not
    trainX, trainY = create_multi_ahead_samples(trainData, lag, h_train, RNN=flag)
    testX, testY = create_multi_ahead_samples(testData, lag, h_test, RNN=flag)
    trainY = np.squeeze(trainY).reshape(-1, h_train)
    testY = np.squeeze(testY).reshape(-1, h_test)
    print("train X shape:", trainX.shape)
    print("train y shape:", trainY.shape)
    print("test X shape:", testX.shape)
    print("test y shape:", testY.shape)

    # 训练模型
    net = train(trainX, trainY,  epoch=epoch, lr=lr, batchSize=batch_size,
                lag=lag, method=method, hidden_num=hidden_num, use_cuda=use_cuda)
    save_model(net, f"102_Model/trained_model_res_{iteration}.pth")

    # # 调用模型
    # net = load_model(path=f"102_Model/trained_model_res_{iteration}.pth", hidden_num=10, use_cuda=True)

    testPred = predict_iteration(net, testX, h_test, use_cuda=use_cuda, RNN=flag)

    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluation
    R = eval.calcR(testY, testPred)
    print("test R", R)

    RMSE = eval.calcRMSE(testY, testPred)
    print("test RMSE", RMSE)

    return testPred, testY

# forecasting using neural networks with time series decomposition, support multi-step ahead forecasting
def decomposition_model_forecasting(ts, dataset, lag, h_train, h_test, freq, epoch, lr, batch_size, hidden_num, use_cuda, method):

    # time series decomposition，去掉了首尾两个值
    trend, seasonal, residual = ts_decompose(ts, freq)
    # # 保存预测结果到CSV文件，结果进行了归一化
    # pd.DataFrame(trend, columns=["TPI_trend"]).to_csv('data/trend.csv', index=False)
    # pd.DataFrame(seasonal, columns=["TPI_seasonal"]).to_csv('data/seasonal.csv', index=False)
    # pd.DataFrame(residual, columns=["TPI_residual"]).to_csv('data/residual.csv', index=False)
    # pd.DataFrame(residual + seasonal + residual, columns=["TPI"]).to_csv('data/total.csv', index=False)

    # forecasting sub-series independently
    errors = []  # 创建一个空列表来收集所有的tmperror值
    for iteration in range(1):

        trend_pred, trend_y = single_model_forecasting_trend(trend, lag=lag, h_train=h_train, h_test=h_test, epoch=epoch, lr=lr,
                                                       hidden_num=hidden_num, batch_size=batch_size, method=method, use_cuda=use_cuda, iteration=iteration)

        res_pred, res_y = single_model_forecasting_res(residual, lag=lag, h_train=h_train, h_test=h_test, epoch=epoch, lr=lr,
                                                   hidden_num=hidden_num, batch_size=batch_size, method='TSFE', use_cuda=use_cuda, iteration=iteration)

        season_pred, season_y = single_model_forecasting_season(seasonal, lag=lag, h_train=h_train, h_test=h_test, epoch=epoch, lr=lr,
                                                         hidden_num=hidden_num, batch_size=batch_size, method=method, use_cuda=use_cuda, iteration=iteration)

        trend_pred = trend_pred.reshape(-1, h_test)
        trend_y = trend_y.reshape(-1, h_test)
        res_pred = res_pred.reshape(-1, h_test)
        res_y = res_y.reshape(-1, h_test)
        season_pred = season_pred.reshape(-1, h_test)
        season_y = season_y.reshape(-1, h_test)

        print("trend_pred shape is", trend_pred.shape)
        print("res_pred shape is", res_pred.shape)
        print("season_pred shape is", season_pred.shape)
        print("trend_y shape is", trend_y.shape)
        print("res_y shape is", res_y.shape)
        print("season_y shape is", season_y.shape)

        testPred = trend_pred + res_pred + season_pred
        testY = trend_y + res_y + season_y



        # 将NumPy数组转换为Pandas DataFrame
        pred_df = pd.DataFrame(testPred)
        test_df = pd.DataFrame(testY)

        # 保存预测结果到CSV文件
        pred_df.to_csv(f'103_Result/testPred_{iteration}.csv', index=False)
        test_df.to_csv('103_Result/testY.csv', index=False)

        # Open the file in write mode
        with open("103_Result/output.txt", "a") as file:
            file.write(f"test {iteration}: \n")

            R = eval.calcR(testY, testPred)
            print("test R", R)
            file.write(f"test R: {R}\n")

            R_t = eval.calcR(trend_y, trend_pred)
            print("test R_t", R_t)
            file.write(f"test R_t: {R_t}\n")

            R_s = eval.calcR(season_y, season_pred)
            print("test R_s", R_s)
            file.write(f"test R_s: {R_s}\n")

            R_r = eval.calcR(res_y, res_pred)
            print("test R_r", R_r)
            file.write(f"test R_r: {R_r}\n")

            RMSE = eval.calcRMSE(testY, testPred)
            print("test RMSE", RMSE)
            file.write(f"test RMSE: {RMSE}\n")

        tmperror = np.concatenate([R, RMSE]).reshape(2, h_test)
        errors.append(tmperror)  # 添加每次迭代的tmperror值到列表中
    return errors


# 在全局范围内：
if __name__ == '__main__':

    # parameters
    for siye in range(1,2):
        lag = 60
        h_train = 1
        h_test = 6
        batch_size = 32
        epoch = 100
        METHOD = "GRU"  # fixed
        freq = 6
        lr = 0.001
        hidden_num = 10  # fixed

        print("lag:", lag)
        print("batch size", batch_size)
        print("h train:", h_train)
        print("h test:", h_test)
        print("epoch:", epoch)
        print("METHOD:", METHOD)
        print("freq:", freq)
        print("lr:", lr)

    # datasets
        ts, data = load_data("101_Data/tpi_monthly.csv", columnName="Value")

        # print(avgerror)
        for i in range(1):
            error = decomposition_model_forecasting(ts=ts, dataset=data, lag=lag, h_train=h_train,
                                                              h_test=h_test,
                                                              epoch=epoch, lr=lr, use_cuda=True, batch_size=batch_size,
                                                              freq=freq,
                                                              method=METHOD, hidden_num=hidden_num)

        # error_dir = "error-tmp.txt"
        # file = open(error_dir, 'a')
        # file.write(str(lag))
        # file.write(" average:")
        # file.write('\n')
        # # file.write(str(avgerror).strip().strip('[]')) # 原代码
        # file.write(str(error).strip().strip('[]'))  # 后加
        # file.write('\n')
        # file.close()