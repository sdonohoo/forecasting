args = commandArgs(trailingOnly=TRUE)
seed_value = args[1]
library('data.table')
library('quantreg')
data_dir = 'energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/baseline/data/features'
train_dir = file.path(data_dir, 'train')
test_dir = file.path(data_dir, 'test')

train_file_prefix = 'train_round_'
test_file_prefix = 'test_round_'

output_file = file.path(paste('energy_load/GEFCom2017_D_Prob_MT_hourly/submissions/baseline/submission_seed_', seed_value, '.csv', sep=""))

quantiles = seq(0.1, 0.9, by = 0.1)

result_all = list()
counter = 1

for (iR in 1:6){
  print(paste('Round', iR))
  train_file = file.path(train_dir, paste(train_file_prefix, iR, '.csv', sep=''))
  test_file = file.path(test_dir, paste(test_file_prefix, iR, '.csv', sep=''))
  
  train_df = fread(train_file)
  test_df = fread(test_file)
  
  # month = unique(test_df$MonthOfYear)
  #
  # train_df = train_df[MonthOfYear == month]
  
  zones = unique(train_df[, Zone])
  hours = unique(train_df[, Hour])
  
  
  for (z in zones) {
    print(paste('Zone', z))
    for (h in hours){
      train_df_sub = train_df[Zone == z & Hour == h]
      test_df_sub = test_df[Zone == z & Hour == h]
      
      result = data.table(Zone=test_df_sub$Zone, Datetime = test_df_sub$Datetime, Round=iR)
      
      for (tau in quantiles){
        
        model =  rq(DEMAND ~ LoadLag + DryBulbLag +
                      annual_sin_1 + annual_cos_1 + annual_sin_2 + annual_cos_2 + annual_sin_3 + annual_cos_3 +
                      weekly_sin_1 + weekly_cos_1 + weekly_sin_2 + weekly_cos_2 + weekly_sin_3 + weekly_cos_3,
                    data=train_df_sub, tau = tau)
        
        result$Prediction = predict(model, test_df_sub)
        result$q = tau
        
        result_all[[counter]] = result
        counter = counter + 1
      }
    }
  }
}

result_final = rbindlist(result_all)

fwrite(result_final, output_file)
