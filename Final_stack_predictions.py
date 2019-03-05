""" This is the last python file to execute. The output csv file from this will give us the file, ML Squad would like to evaluate for LeaderBoard Ranking. """

input('Ensure current working directory is Flipkart_ML_Squad_Source')
PREDICTIONS = "predictions*"
import glob
import pandas as pd

i = 0
for filename in glob.glob(PREDICTIONS):
    if i == 0:
        tot = pd.read_csv(filename)
        i+=1
    else: 
        tot += pd.read_csv(filename)
        i += 1
tot[['x1', 'x2', 'y1', 'y2']] = tot[['x1', 'x2', 'y1', 'y2']]/i
tot[['x1', 'x2', 'y1','y2']] = tot[['x1', 'x2', 'y1','y2']].astype(int)
tot.to_csv('FINAL_PREDICTION.csv', index = False)