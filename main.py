import randomForest as rf
import finance_data as fd
import linearRegression as lr

if __name__ == "__main__":
    chip_companies = ['NVDA']



    linear = lr.LinearRegressionTrading()
    chip_companies = ['NVDA']
    linear.plot_stock_progress(chip_companies)
    linear.plot_stock_with_regression(chip_companies)

    rfc = rf.RFC(chip_companies)

    data = rfc.prepare_data()
    print("Dataset Sample:\n", data)
    
    model = rfc.train_model(data)
    #rfc.plot_feature_importance(model, ['P/E Ratio', 'EPS', 'Market Cap', 'Close', 'Open'])