

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from pnplot import *

dataset_filename = 'priority_scopp_statisticsnew.xlsx'

dataframe = pd.read_excel(open(dataset_filename, 'rb'),
                               sheet_name='dataset')
fig = plt.figure(figsize=(10,8))
axis = fig.add_subplot(211)

#Mission Time Figure
sns.boxplot(x="Priority", y="Mission Time", hue="Map", data=dataframe)
plt.xlabel('# of Priority Points')
plt.ylabel('Mission Times (s)')
plt.title('SCoPP Mission Times for 10 Robots')
#plt.show()

#Computing Time Figure

axis2 = fig.add_subplot(212)

sns.boxplot(x="Priority", y="Computing Time", hue="Map", data=dataframe)
plt.xlabel('# of Priority Points')
plt.ylabel('Computing Time (s)')
plt.title('SCoPP Computing Times for 10 Robots')
plt.show()

plt.savefig('Standarddev.png')