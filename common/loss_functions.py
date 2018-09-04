import pandas as pd

def pinball_loss(predictions, actuals, q):
    zeros = pd.Series([0]*len(predictions))
    return (predictions-actuals).combine(zeros, max)*(1-q) + (actuals-predictions).combine(zeros, max)*q