import pandas as pd

# TRAIN_BASE_END = pd.to_datetime('2016-12-01')
# TRAIN_ROUNDS_ENDS = pd.to_datetime(['2016-12-15', '2016-12-31',
#                                     '2017-01-15', '2017-01-31',
#                                     '2017-02-14', '2017-02-28'])
#
# TEST_STARTS_ENDS = [pd.to_datetime(('2017-01-01', '2017-02-01')),
#                     pd.to_datetime(('2017-02-01', '2017-03-01')),
#                     pd.to_datetime(('2017-02-01', '2017-03-01')),
#                     pd.to_datetime(('2017-03-01', '2017-04-01')),
#                     pd.to_datetime(('2017-03-01', '2017-04-01')),
#                     pd.to_datetime(('2017-04-01', '2017-05-01'))]

TRAIN_BASE_END = pd.to_datetime('2016-11-01')
TRAIN_ROUNDS_ENDS = pd.to_datetime(['2016-12-01', '2016-12-01',
                                    '2017-01-01', '2017-01-01',
                                    '2017-02-01', '2017-02-01'])

TEST_STARTS_ENDS = [pd.to_datetime(('2017-01-01', '2017-02-01')),
                    pd.to_datetime(('2017-02-01', '2017-03-01')),
                    pd.to_datetime(('2017-02-01', '2017-03-01')),
                    pd.to_datetime(('2017-03-01', '2017-04-01')),
                    pd.to_datetime(('2017-03-01', '2017-04-01')),
                    pd.to_datetime(('2017-04-01', '2017-05-01'))]