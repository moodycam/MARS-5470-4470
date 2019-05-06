import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from pandas import ExcelWriter
from pandas import ExcelFile

#-------------------------------

file = "C:\\Users\Bridgett\Desktop\Major_Ecoregions.xlsx"
Major_ecoregions = pd.read_excel(file)

#-------------------------------

Major_ecoregions.columns = ['Index','Major Ecoregion', 'Major_ecoreg_area', 'Major_ecoreg_percent_area']

#-------------------------------

Major_ecoregions=Major_ecoregions.drop(Major_ecoregions.columns[[0]], axis=1)

#-------------------------------

Major_ecoregions

#-------------------------------

Major_ecoregions

#-------------------------------

file1 = "C:\\Users\Bridgett\Desktop\Sub_Ecoregions.xlsx"
Sub_ecoregions = pd.read_excel(file1)

#-------------------------------

Sub_ecoregions.columns = ['Index','Veg_ID', 'Sub_ecoreg_area', 'Sub_ecoreg_percent_area']

#-------------------------------

Sub_ecoregions=Sub_ecoregions.drop(Sub_ecoregions.columns[[0]], axis=1)

#-------------------------------

Sub_ecoregions

#-------------------------------

ax = Major_ecoregions.plot(kind='barh', figsize=(8, 10), color='#86bf91', zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

  # Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

  # Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

  # Set x-axis label
ax.set_xlabel("Total Area (Meters)", labelpad=20, weight='bold', size=12)

  # Set y-axis label
ax.set_ylabel("Major Ecoregions", labelpad=20, weight='bold', size=12)

#-------------------------------

sorted_by_ascending = Major_ecoregions.sort_values(['Major_ecoreg_percent_area'], ascending=False)

#-------------------------------

sorted_by_ascending["Major_ecoreg_percent_area"]

#-------------------------------

sorted_by_ascending["Major_ecoreg_percent_area"].plot(kind="barh")
plt.show()

#-------------------------------

Major_ecoregions.plot.(kind="barh",x='Major_ecoreg_percent_area', y='Major Ecoregion', rot=0)

#-------------------------------

df=pd.sorted_by_ascending["Major_ecoreg_percent_area"]

#-------------------------------

ax = Major_ecoregions[['Major_ecoreg_percent_area','Major Ecoregion']].plot(kind='bar', title ="Major Ecoregions by %Area in the RGV", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("Percent", fontsize=12)
plt.show()

#-------------------------------

Area=Major_ecoregions['Major_ecoreg_percent_area']
Label=Major_ecoregions['Major Ecoregion']
y_pos = np.arange(len(Label))

# Create horizontal bars
plt.barh(y_pos, Area)

# Create names on the y-axis
plt.yticks(y_pos, Label)

# Show graphic
plt.show()

#-------------------------------

Major_Ecoregions=sorted_by_ascending

#-------------------------------

Area=Major_Ecoregions['Major_ecoreg_percent_area']
Label=Major_Ecoregions['Major Ecoregion']
y_pos = np.arange(len(Label))

# Create horizontal bars
plt.barh(y_pos, Area)

# Create names on the y-axis
plt.yticks(y_pos, Label)

# Show graphic
plt.show()

#-------------------------------

ax = Major_Ecoregions[['Major Ecoregion', 'Major_ecoreg_percent_area']].plot(kind='bar', title ="Major Ecoregions by %Area in the RGV", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel(" Maj_Eco", fontsize=12)
ax.set_ylabel("Percent", fontsize=12)

plt.show()

#-------------------------------

Major_Ecoregions = Major_ecoregions.sort_values(['Major_ecoreg_percent_area'], ascending=True)

#-------------------------------

plt.barh?

#-------------------------------

fig, ax = plt.subplots(figsize=(15, 10))
Label=Major_Ecoregions['Major Ecoregion']
y = np.arange(len(Label))
Area=Major_Ecoregions['Major_ecoreg_percent_area']
x=[5,10,15,20,25,30]


ax.barh(y, Area, align='center',
        color='kyr', ecolor='black')
ax.set_yticks(y)
ax.set_xticks(x)
ax.set_yticklabels(Label, fontsize=16)
ax.set_xticklabels(x, fontsize=14)
ax.invert_yaxis()
ax.set_xlabel('% Area(Meters)',fontsize=16)
ax.set_title('Major Ecoregions by % Area in the RGV (Meters)',fontsize=24)
plt.show()

#-------------------------------

Major_Ecoregions.tail(5)

#-------------------------------

Sub_ecoregions

#-------------------------------

Major_ecoregions

#-------------------------------

TMDT=[7004,7005]
TSG=[7103,7104,7105,7107]
TCT=[7204,7205,7207]
TF=[7402,7403,7404,7405,7406,7417,7407]
TR=[7602,7604,7605,7606,7607]
MNAMT=[9000,9007,9187,9600,9104,9106,9124,9107,9128,9116,9204]
AHRMT=[9304,9307,9410,9411,9317]
TCDW=[10004,10006,10017]
TSCP=[2206,2207]
TCSBTM=[5600,5605,5617,5606,5616]
RGDTWS=[7802,7804,7805]
TCB=[6100]
TCDCG=[6200,6306,6307]
SCPIW=[6507]
STSBTF=[6600,6610]
TL=[7305,7306,7307]
TPGRF=[7502]
TSL=[7700,7707]
TST=[6806]
CSTCFFW=[6402,6403,6405]
TCG=[6707]

#-------------------------------

df2 = []

for i in Sub_ecoregions['Veg_ID']:
    if i in TMDT:
        df2.append(0)
    elif i in TSG:
        df2.append(1)
    elif i in TCT:
        df2.append(2)
    elif i in TF:
        df2.append(3)
    elif i in TR:
        df2.append(4)
    elif i in MNAMT:
        df2.append(5)
    elif i in AHRMT:
        df2.append(6)
    elif i in TCDW:
        df2.append(7)
    elif i in TSCP:
        df2.append(8)
    elif i in TCSBTM:
        df2.append(9)
    elif i in RGDTWS:
        df2.append(10)
    elif i in TCB:
        df2.append(11)
    elif i in TCDCG:
        df2.append(12)
    elif i in SCPIW:
        df2.append(13)
    elif i in STSBTF:
        df2.append(14)
    elif i in TL:
        df2.append(15)
    elif i in TPGRF:
        df2.append(16)
    elif i in TSL:
        df2.append(17)
    elif i in TST:
        df2.append(18)
    elif i in CSTCFFW:
        df2.append(19)
    elif i in TCG:
        df2.append(20)
    else:
        break

Sub_ecoregions['Maj_ER'] = df2

#-------------------------------

Sub_ecoregions

#-------------------------------

Sub_ecoregions.to_excel('Desktop/Sub_ecoregions1.xlsx')

#-------------------------------

# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# y-axis in bold
rc('font', weight='bold')

# Values of each group
bars1 = [12, 28, 1, 8, 22]
bars2 = [28, 7, 16, 4, 10]
bars3 = [25, 3, 23, 25, 17]

# Heights of bars1 + bars2
bars = np.add(bars1, bars2).tolist()

# The position of the bars on the x-axis
r = [0,1,2,3,4]

# Names of group and bar width
names = ['A','B','C','D','E']
barWidth = 1

# Create brown bars
plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
# Create green bars (top)
plt.bar(r, bars3, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)

# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")

# Show graphic
plt.show()

#-------------------------------

# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import pandas as pd

# y-axis in bold
rc('font', weight='bold')

# Values of each group
bars1 = [12]
bars2 = [28]
bars3 = [25]

# Heights of bars1 + bars2
bars = np.add(bars1, bars2).tolist()

# The position of the bars on the x-axis
r = [0,1,2,3,4]

# Names of group and bar width
names = ['A','B','C','D','E']
barWidth = 1

# Create brown bars
plt.bar(r, bars1, color='#7f6d5f', edgecolor='white', width=barWidth)
# Create green bars (middle), on top of the firs ones
plt.bar(r, bars2, bottom=bars1, color='#557f2d', edgecolor='white', width=barWidth)
# Create green bars (top)
plt.bar(r, bars3, bottom=bars, color='#2d7f5e', edgecolor='white', width=barWidth)

# Custom X axis
plt.xticks(r, names, fontweight='bold')
plt.xlabel("group")

# Show graphic
plt.show()

#-------------------------------

%matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')


data = [[2000, 2000, 2000, 2001, 2001, 2001, 2002, 2002, 2002], #MER
        ['Jan', 'Feb', 'Mar', 'Jan', 'Feb', 'Mar', 'Jan', 'Feb', 'Mar'], #SER
        [1, 2, 3, 4, 5, 6, 7, 8, 9]] #Area

rows = zip(data[0], data[1], data[2])
headers = ['Year', 'Month', 'Value']


df = pd.DataFrame(rows, columns=headers)

df

#-------------------------------

fig, ax = plt.subplots(figsize=(10,7))

months = df['Month'].drop_duplicates()
margin_bottom = np.zeros(len(df['Year'].drop_duplicates()))
colors = ["#006D2C", "#31A354","#74C476"]

for num, month in enumerate(months):
    values = list(df[df['Month'] == month].loc[:, 'Value'])

    df[df['Month'] == month].plot.bar(x='Year',y='Value', ax=ax, stacked=True,
                                    bottom = margin_bottom, color=colors[num], label=month)
    margin_bottom += values

plt.show()

#-------------------------------

Sub_ecoregions

#-------------------------------

MER=Sub_ecoregions['Maj_ER'].unique()
MER

#-------------------------------

SER=Sub_ecoregions['Veg_ID'].unique()
SER

#-------------------------------







fig, ax = plt.subplots(figsize=(10,7))

months = df['Month'].drop_duplicates()
margin_bottom = np.zeros(len(df['Year'].drop_duplicates()))
colors = ["#006D2C", "#31A354","#74C476"]

for num, month in enumerate(months):
    values = list(df[df['Month'] == month].loc[:, 'Value'])

    df[df['Month'] == month].plot.bar(x='MER',y='SER', ax=ax, stacked=True,
                                    bottom = margin_bottom, color=colors[num], label=month)
    margin_bottom += values

plt.show()

#-------------------------------

fig, ax = plt.subplots(figsize=(10,7))

Sub_ERs = Sub_ecoregions['Veg_ID'].drop_duplicates()
margin_bottom = np.zeros(len(Sub_ecoregions['Maj_ER'].drop_duplicates()))
colors = ["#006D2C", "#31A354","#74C476"]

for num, vegid in enumerate(Sub_ERs):
    area = list(Sub_ecoregions[Sub_ecoregions['Veg_ID'] == vegid].loc[:, 'Sub_ecoreg_percent_area'])

    Sub_ecoregions[Sub_ecoregions['Veg_ID'] == vegid].plot.bar(x='Maj_ER',y='Veg_ID', ax=ax, stacked=True,
                                    bottom = margin_bottom, color=colors[num], label=vegid)
    margin_bottom += values

plt.show()

#-------------------------------


#-------------------------------

N=6

y1=[3,9,11,2,6,4]

y2=[6,4,7,8,3,4]

xvalues = np.arange(N)

plt.bar(xvalues,y1,color='b', label ='Team1')
plt.bar(xvalues,y2, color='r', bottom =y1, label = 'Team2')
plt.xticks(xvalues, ('V1', 'V2', 'V3', 'V4', 'V5'))

plt.xlabel('Teams')
plt.ylabel('Scores')
plt.title('Stacked Bar Graphs')
plt.legend()

#-------------------------------

N=1 #I need 5 columns

y1=[3] #7004 SER

y2=[6] #7005 SER

xvalues = np.arange(N)

plt.bar(xvalues,y1,color='b', label ='Team1')
plt.bar(xvalues,y2, color='r', bottom =y1, label = 'Team2')
plt.xticks(xvalues, ('V1'))

plt.xlabel('Teams')
plt.ylabel('Scores')
plt.title('Stacked Bar Graphs')
plt.legend()

#-------------------------------

N=1 #I need 5 columns

y1=[3.471420] #Area of 7004 SER

y2=[1.856571] #Area of 7005 SER

xvalues = np.arange(N)

plt.bar(xvalues,y1,color='b', label ='Team1')
plt.bar(xvalues,y2, color='r', bottom =y1, label = 'Team2')
plt.xticks(xvalues, ('Y'))

plt.xlabel('Teams')
plt.ylabel('Scores')
plt.title('Stacked Bar Graphs')
plt.legend()

#-------------------------------

N=1 #I need 5 columns

#Assigning Area of Subecoregion to a Variable
a1=[3.471420] #Area of 7004 SER
a2=[1.856571] #Area of 7005 SER

#Second Bar
b1=
b2=

#Plotting the area of the SubEcoregion in a bar
plt.bar(xvalues,a1,color='b', label ='Team1')
plt.bar(xvalues,a2, color='r', bottom =y1, label = 'Team2')

#Second Bar


#plot
xvalues = np.arange(N)
plt.xticks(xvalues, ('A'))
plt.xlabel('Teams')
plt.ylabel('Scores')
plt.title('Stacked Bar Graphs')
plt.legend()

#-------------------------------

Sub_ecoregions

#-------------------------------

Veg_ID=Sub_ecoregions['Veg_ID']
Maj_ER=Sub_ecoregions['Maj_ER']

#-------------------------------

Sub_ecoregions.Veg_ID[Sub_ecoregions.Maj_ER == 12]

#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------


#-------------------------------

Top_Maj_ER=Major_Ecoregions.tail(5)

#-------------------------------

Top_Maj_ER

#-------------------------------


#-------------------------------


#-------------------------------


