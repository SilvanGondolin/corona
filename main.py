import numpy as np
from scipy import optimize
import pandas
from matplotlib import pyplot as plt

# https://www.srf.ch/news/international/schweiz-und-weltweit-so-entwickeln-sich-die-coronavirus-fallzahlen

def func(t, base, bias, factor):
    t0 = 0
    return bias + factor*base**(t-t0)

# pd.read_csv('file.csv', index_col=0, header=None).T
# file_path = "/home/raphael/my_data/my_code/corona/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv"
file_path = "/home/raphael/my_data/my_code/corona/COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
data = pandas.read_csv(filepath_or_buffer=file_path,
                       index_col=0,
                       skiprows=0).T
data.columns = data.iloc[0]

# sw_idx = data.index[data['Country/Region'] == "Switzerland"].tolist()
# print(data.iloc[sw_idx])

switzerland = data["Switzerland"]
cases = np.array(switzerland[3:])
cases = cases[34:]
days = np.arange(cases.shape[0])

training_idcs = np.arange(cases.shape[0])
training_data = cases[training_idcs]
training_days = days[training_idcs]

fitted_params = optimize.curve_fit(func, xdata=training_days, ydata=training_data)[0]

print("base   = " + str(fitted_params[0]))
print("bias = " + str(fitted_params[1]))
print("factor   = " + str(fitted_params[2]))
# print("day0   = " + str(fitted_params[3]))

# future_days = np.arange(100)
future_days = days

plt.plot(days, cases)
plt.plot(future_days, func(future_days, fitted_params[0], fitted_params[1], fitted_params[2]))
plt.show()