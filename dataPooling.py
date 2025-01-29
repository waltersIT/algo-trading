import RFC as rfc
import finance_data as fd
import linearRegression as lr
import newsTrading as nt

class DataPooling:
    def get_RFC_predictions():
        chip_companies = ['NVDA']
        randfc = rfc.RFC(chip_companies)
        
        data = randfc.prepare_data()
        model = randfc.train_model(data)
        response = randfc.backtest(model, data)
        return response

    def get_news_predictions():
        response = nt.HeadlineAnalyzer().get_sentiment()
        return response

    def get_linear_regression():
        linear = lr.LinearRegressionTrading()
        chip_companies = ['NVDA']
        #linear.plot_stock_progress(chip_companies)
        #linear.plot_stock_with_regression(chip_companies)
        slope = linear.get_stock_slope(chip_companies)
        return slope


    def return_data():
        linIndicator = ""
        newsIndicator = ""

        info = []
        randFC = DataPooling.get_RFC_predictions()
        newsPrediction = DataPooling.get_news_predictions()
        linearReg = DataPooling.get_linear_regression()
        #again im tired and just making it so it works for one number
        #make it so it can handle arrays so you dont run it multiple times
        #rlly make it so all of them can be called in main and declare the chip companies there
        #ur gonna need to refactor this hoe and make it more efficient bro
        if linearReg > 0:
            linIndicator = "BUY"
        else:
            linIndicator = "SELL"
            
        if newsPrediction > 3:
            newsIndicator = "BUY"
        else:
            newsIndicator = "SELL"


        info = {"rfc" : randFC, 
                "news" : newsIndicator, 
                "linear" : linIndicator}
        return info

"""
if __name__ == "__main__":
    info = DataPooling.return_data()
    print(info)
"""


    
