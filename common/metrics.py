def MAPE(predictions, actuals):
    return ((predictions - actuals).abs() / actuals).mean()


def sMAPE(predictions, actuals):
    return ((predictions - actuals).abs() / (predictions.abs() + actuals.abs())).mean()