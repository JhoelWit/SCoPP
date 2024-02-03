

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

from pnplot import *

data = 'Priority SCoPP Statistics.xlsx'

header = ['0', '10', '100', '1000']

missiondf = pd.read_excel(data,
sheet_name = 0,
header = 0,
index_col = False,
keep_default_na = True,
usecols= [2, 4]
)
missiondf_wide = pd.read_excel(data,
sheet_name = 0,
header = 0,
index_col = False,
keep_default_na = True,
usecols= [7, 9, 11, 13, 15, 17, 19, 21],
skiprows= [0, 10-46]
)

computingdf = pd.read_excel(data,
sheet_name = 0,
header = 0,
index_col = False,
keep_default_na = True,
usecols= [3, 5]
)
computingdf_wide = pd.read_excel(data,
sheet_name = 0,
header = 0,
index_col = False,
keep_default_na = True,
usecols= [6, 8, 10, 12, 14, 16, 18, 20],
skiprows= [0, 10-46]
)


#mediumMapMissionData = {'0 Priority Points': [df['Medium map mission time (s)'][0:10]], '10 Priority Points': [df['Medium map mission time (s)'][12:21]], \
#'100 Priority Points': [df['Medium map mission time (s)'][24:33]], '1000 Priority Points': [df['Medium map mission time (s)'][36:45]]}

print(missiondf_wide)
fig = plt.figure()
axis = fig.add_subplot(111)

sns.boxplot(x = '# of Priority Points', y = 'Mission Time', data = missiondf_wide)

# sns.boxplot(x = 'Priority Points', y = 'Mission Time (s)', hue = 'Map', data=missiondf_wide)
plt.xlabel('# of Priority Points')
plt.ylabel('Mission Times (s)')
plt.show()