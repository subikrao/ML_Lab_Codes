import pandas as pd
import numpy as np
import statistics as st
import matplotlib.pyplot as mp
excel_file = 'Lab_Session1_Data.xlsx'
df = pd.read_excel(excel_file, sheet_name='IRCTC Stock Price')
col_d=df.iloc[0:, 3]
mean =st.mean(col_d)
print("Mean of Column 'Price' = ", mean)
variance = st.variance(col_d)
print("Variance of Column 'Price' = ", variance, "\n")
wednesdays_prices = df[df['Day'] == 'Wed']
mean_wednesdays = st.mean(wednesdays_prices['Price'])

print("Sample Mean of Wednesday Prices:", mean_wednesdays)
print("Population Mean of All Prices:", mean, "\n")

# chg% prob

chg=[]
chg=df['Chg%']
neg_chg=list(filter(lambda x: (x<0), chg))
nc=len(neg_chg)
cc=len(chg)
# probability of loss
probability_of_loss = 1 - nc/cc

print("Probability of loss on the stock is",probability_of_loss)

probability_of_profit_on_all_days = 1 - probability_of_loss
print("Probability of profit on the stock is",probability_of_profit_on_all_days, "\n")


# wednesdays
w = df[df['Day'] == 'Wed'] 
# wednesdays count
w_count = len(w['Chg%'])
# wednesdays profits
wed_profits  =w[w['Chg%'] > 0]
# wednesdays profits count
wp_count = len(wed_profits['Chg%'])
# probability of profits on wednesdays
wedprofit_prob= wp_count/probability_of_profit_on_all_days

print("Probability of making profit on wed=", wedprofit_prob)
# print(wed_profits)


# Calculate the conditional probability of making profit, given that today is Wednesday.

# P(profit | wed) = P(profit and wednesday) / P(wednesday)

# = wed_profits / w
conditional_profit = wp_count / w_count
print("conditional probability of making a profit, given that today is wednesday = ", conditional_profit)


# xax = df["Day","Chg%"]
# yax = df["Chg%"]
# df.plot( x = "Day" , y = "Chg%" )
mp.scatter(df["Day"], df["Chg%"])
mp.xlabel("Day of the week")
mp.ylabel("% Change in stock price of IRCTC")
mp.grid()
mp.show()
