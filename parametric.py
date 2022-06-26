import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from pnplot import *

dataset_filename = 'priority_scopp_statisticsnew.xlsx'

dataframe = pd.read_excel(open(dataset_filename, 'rb'),
                               sheet_name='Parametric_Analysis')
fig = plt.figure(figsize=(10,8))
axis = fig.add_subplot(211)

#Mission Time Figure
sns.lineplot(x="Bias", y="Mission Time", hue="Map", data=dataframe[0:12])
plt.xlabel('Bias')
plt.ylabel('Mission Times (s)')
plt.title('SCoPP Mission Times for 10 Robots')
#plt.show()

#Computing Time Figure

axis2 = fig.add_subplot(212)

sns.lineplot(x="Bias", y="Computing Time", hue="Map", data=dataframe[0:12])
plt.xlabel('Bias')
plt.ylabel('Computing Time (s)')
plt.title('SCoPP Computing Times for 10 Robots')
plt.show()

plt.savefig('Standarddev.png')