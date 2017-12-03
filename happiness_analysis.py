import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
# import tensorflow as tf
# import sonnet as snt 

dataframe = pd.read_fwf('data/happiness_analysis/2015_clean.csv') #.to_string(index=False)

happiness_score = dataframe.as_matrix(['Happiness.Score'])
economy_gdp = dataframe.as_matrix(['Economy..GDP.per.Capita.'])
country = dataframe.as_matrix(['Country'])
# FIXME: This is probably not the best way to do this :(
country = country.reshape(country.shape[0])

fig, ax = plt.subplots()
ax.scatter(happiness_score, economy_gdp)

#for i, txt in enumerate(country):
#   ax.annotate(str(txt), (happiness_score[i], economy_gdp[i]))

plt.xlabel('Happiness Score')
plt.ylabel('Economy (GDP per capita)')
axes = plt.gca()
axes.set_xlim([0, 20])
axes.set_ylim([0, 100])

plt.grid()
plt.show()

# print ("{}, {}".format(happiness_score[0], economy_gdp[0]))

