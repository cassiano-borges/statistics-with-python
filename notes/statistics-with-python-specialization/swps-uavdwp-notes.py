True + True # == 2

# Python standard library
### https://docs.python.org/3/library/index.html#library-index

import numpy as np
import pandas as pd

a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])

np.dot(a,b)
    # array([[37, 40],
    #        [85, 92]])

################################################################################
#                                                                              #
########################   importing data with pandas   ########################
#                                                                              #
################################################################################

# ================================= Data Frame =================================

# Store the url string that hosts our .csv file
url = "Cartwheeldata.csv"

# Read the .csv file and store it as a pandas Data Frame
df = pd.read_csv(url)

# Output object type
type(df)

# We can view our Data Frame by calling the head() function
df.head()
#    ID  Age Gender  GenderGroup  ... CWDistance  Complete  CompleteGroup  Score
# 0   1   56      F            1  ...         79         Y              1      7
# 1   2   26      F            1  ...         70         Y              1      8
# 2   3   33      F            1  ...         85         Y              1      7
# 3   4   39      F            1  ...         87         Y              1     10
# 4   5   27      M            2  ...         72         N              0      4
#
# [5 rows x 12 columns]


df.columns
# Index(['ID', 'Age', 'Gender', 'GenderGroup', 'Glasses', 'GlassesGroup',
#        'Height', 'Wingspan', 'CWDistance', 'Complete', 'CompleteGroup',
#        'Score'],
#       dtype='object')

df.dtypes                   # see the datatypes for each column
df.loc(rows, columns)       # [row, [columns]]
df.iloc(rows, columns)      # integer based splicing
df.ix()
df.column_name.unique()     # unique values of that column
df.groupby(['columnA', 'columnB'])
pd.read_csv("my_file.csv")

# Splicing the data
# Return all observations of CWDistance
df.loc[:,"CWDistance"] == df.CWDistance
# Select all rows for multiple columns, ["CWDistance", "Height", "Wingspan"]
df.loc[:,["CWDistance", "Height", "Wingspan"]]
# Select 9 rows for multiple columns, ["CWDistance", "Height", "Wingspan"]
df.loc[:9, ["CWDistance", "Height", "Wingspan"]]
# Select range of rows for all columns
df.loc[10:15]

# integer based splicing
df.iloc[:4]
df.iloc[1:5, 2:4]

# List unique values in the df['Gender'] column
df.Gender.unique()
    # array(['F', 'M'], dtype=object)

# Use .loc() to specify a list of mulitple column names
df.loc[:,["Gender", "GenderGroup"]]
df.groupby(['Gender','GenderGroup']).size()
# Gender  GenderGroup
# F       1              12
# M       2              13
# dtype: int64


url = "nhanes_2015_2016.csv"
da = pd.read_csv(url)
print(da.shape)
    # (5735, 28)
da.columns
# Index(['SEQN', 'ALQ101', 'ALQ110', 'ALQ130', 'SMQ020', 'RIAGENDR', 'RIDAGEYR',
#        'RIDRETH1', 'DMDCITZN', 'DMDEDUC2', 'DMDMARTL', 'DMDHHSIZ', 'WTINT2YR',
#        'SDMVPSU', 'SDMVSTRA', 'INDFMPIR', 'BPXSY1', 'BPXDI1', 'BPXSY2',
#        'BPXDI2', 'BMXWT', 'BMXHT', 'BMXBMI', 'BMXLEG', 'BMXARML', 'BMXARMC',
#        'BMXWAIST', 'HIQ210'],
#       dtype='object')

# ways to slicing the data
w = da["DMDEDUC2"]
x = da.loc[:, "DMDEDUC2"]
y = da.DMDEDUC2
z = da.iloc[:, 9]  # DMDEDUC2 is in column 9

# finding a maximum within a variable
print(da["DMDEDUC2"].max())
print(da.loc[:, "DMDEDUC2"].max())
print(da.DMDEDUC2.max())
print(da.iloc[:, 9].max())

# counting the number of missing values
print(pd.isnull(da.DMDEDUC2).sum())
print(pd.notnull(da.DMDEDUC2).sum())
    # 261
    # 5474

################################################################################
#                                                                              #
#############################   categorical data   #############################
#                                                                              #
################################################################################

# function describe() to get nummerical summaries
## min
## Q1
## MEDIAN
## Q3
## MAX
## MEAN
## SD - standard deviation
## n - sample size

# =================================== Numpy ====================================
# =================================   array   ==================================

arr.shape                   # 1D - (B,)     /    2D - (A, B)
arr[A, B]
arr.dtype
np.array([a, b], dtype=)    # array([a, b])
np.arrange(a, b, c)         # a to b-1 by c
np.ones((A, B))             # AxB of 1
np.zeros((A, B))            # AxB of 0
np.full((A, B), C)          # AxB of C
np.random.randn((A, B))     # AxB of random
# operations
## between arrays
np.add(arr1, arr2)          # arr1 + arr2
np.subtract(arr1, arr2)     # arr1 - arr2
np.multiply(arr1, arr2)     # arr1 * arr2
np.divide(arr1, arr2)       # arr1 / arr2
## same array
np.sqrt(arr)                # arr**1/2
np.sum(arr, axis=)
np.mean(x, axis=)

# ==============================================================================

import numpy as np

# 3 x 1 array
a = np.array([1,2,3])
a.shape
# (3,)
print(a[0], a[1], a[2])
# 1 2 3

# 2x2 array
b = np.array([[1,2], [3,4]])
b.shape
# (2, 2)
print(b[0,0], b[0,1], b[1,1])
# 1 2 4

# 3x2 array
c = np.array([[1,2], [3,4], [5,6]])
c.shape
# (3, 2)
print(c[0,1], c[1,0], c[2,0], c[2,1])
# 2 3 5 6

# 2x3 zeros array
d = np.zeros((2,3))
# [[0. 0. 0.]
#  [0. 0. 0.]]
# 4x2 arrays of ones
e = np.ones((4,2))
# [[1. 1.]
#  [1. 1.]
#  [1. 1.]
#  [1. 1.]]
# 2x2 constant array
f = np.full((2,2), 9)
# [[9 9]
#  [9 9]]
# 3x3 random array
g = np.random.randn((3,3))
# [[0.36755935 0.37909295 0.18186368]
#  [0.78900426 0.93361392 0.16021468]
#  [0.47262259 0.84857866 0.90491294]]

h = np.array([[1,2,3,4,], [5,6,7,8], [9,10,11,12]])
i = h[:2, 1:3]
# [[2 3]
#  [6 7]]

# operations
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
x + y
np.add(x, y)
[[ 6.  8.]
 [10. 12.]]
x - y
np.subtract(x, y)
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
x * y
np.multiply(x, y)
# [[ 5.0 12.0]
#  [21.0 32.0]]
x / y
np.divide(x, y)
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
np.sqrt(x)
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]

x = np.array([[1,2],[3,4]])
[[1 2]
 [3 4]]
np.sum(x)
# 10
np.sum(x, axis=0)
# [4 6]       # [1+3 2+4]
np.sum(x, axis=1)
# [3 7]       # [1+2 3+4]

np.mean(x)
# 2.5
np.mean(x, axis=0)
# [2. 3.]   # [1n3 2n4]
np.mean(x, axis=1)
# [1.5 3.5] # [1n2 3n4]

# =================================== SciPy ====================================
# Scipy.Stats
from scipy import stats
import numpy as np

stats.norm.rvs(size=A)       # Normal distribution random variables
stats.t.rvs(10, size=A)     # T-distribution continuous random variables
stats.describe(x)           # (nobs=unique values, (min,max), mean, variance,
                            #  skewness, kurtosis)
# ==============================================================================

stats.norm.rvs(size=10)
# [-0.3218403   0.77303307 -0.97840903  0.05980813  0.33827322  0.7564039
#  -0.73308474  0.23376475  2.29916377 -1.17922033]


# ------------------------------------------------ #
# PDF - Probability Density function
# CDF - Cumulative Density function
from pylab import *
# create test data
dx = .01
X = np.arrange(-2, 2, dx)
Y = exp(-X ** 2)

# normalize the data to a proper PDF
Y /= (dx*Y).sum()

# compute the CDF
CY = np.cumsum(Y * dx)

# plot both
plot(X, Y)
plot(X, CY, 'r--')

show()
# ------------------------------------------------ #
# compute normal CDF of values
# return the probability for that value
stats.norm.cdf(np.array([1, -1., 0, 1, 3, 4, -2, 6]))
# [0.84134475 0.15865525 0.5        0.84134475 0.9986501  0.99996833
#  0.02275013 1.        ]

# Descriptive Statistics
# reproductible analysis
np.random.seed(282629734)
# generate 1000 Student's T continuous random variables
x = stats.t.rvs(10, size=1000)
print(x.min())   # equivalent to np.min(x)
-3.7897557242248197
print(x.max())   # equivalent to np.max(x)
5.263277329807165
print(x.mean())  # equivalent to np.mean(x)
0.014061066398468422
print(x.var())   # equivalent to np.var(x))
1.288993862079285
stats.describe(x)
# DescribeResult(nobs=1000,
#                minmax=(-3.7897557242248197, 5.263277329807165),
#                mean=0.014061066398468422,
#                variance=1.2902841462255106,
#                skewness=0.21652778283120955,
#                kurtosis=1.055594041706331)

# ================================= Matplotlib =================================
import numpy as np
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Title')
plt.legend(['1 legend', '2 legend'])
# subplots
plt.subplot(height, width, active_plot)
plt.xsticks(rotation=45)
plt.hist(data["variable"])

# ==============================================================================

# compute x and y coordinates for points on a sine curve
x = np.arrange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# plot the points
plt.plot(x, y)
plt.show()


# compute x and y coordinates for points on a sine and cosine curve
x = np.arrange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# plot the points
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()


# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# set up a subplot grid of height 2 and width 1,
# and set the first such subplot as active
plt.subplot(2, 1, 1)

# first plot
plt.plot(x, y_sin)
plt.title('Sine')

# second plot as active,
# make second plot
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()

# ================================== Seaborn ===================================
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

.set_title("Title")
# scatterplot
sns.lmplot(x='x-variable',
           y='y-variable',
           data=dataframe
           fit_reg=True         # regression line, True as default
           hue='other variable' # color by other variable
           )
# disctance plot
sns.swarmplot(x='x-variable', y='y-variable', data=dataframe)

# boxplot
sns.boxplot(data=dataframe.loc[:, ['variable1', "variable2", "variableN"]])

# histogram
sns.distribution(dataframe.variable)
sns.distplot(dataframe["variable"], kde=('density = True'))

# count plot
sns.countplot(x='variable', data=dataframe)

# ==============================================================================

# create a Data Frame
url = "Cartwheeldata.csv"
df = pd.read_csv(url)

# create scatterplot
sns.lmplot(x='Wingspan', y='CWDistance', data=df)

plt.show()


sns.lmplot(x='Wingspan', y='CWDistance', data=df,
           fit_reg=False,   # no regression line
           hue='Gender')    # color by gender
plt.show()


# Cartwheel distance plot
sns.swarmplot(x='Gender', y='CWDistance', data=df)


# boxplot
sns.boxplot(data=df.loc[:, ['Age', "Height", "Wingspan", "CWDistance", "Score"]])
plt.show()

# male boxplot
sns.boxplot(data=df.loc([df['Gender'] == 'M', ['Age', "Height", "Wingspan", "CWDistance", "Score"]]))
# female boxplot
sns.boxplot(data=df.loc([df['Gender'] == 'F', ['Age', "Height", "Wingspan", "CWDistance", "Score"]]))

# male score boxplot
sns.boxplot(data=df.loc([df['Gender'] == 'M', ["Score"]]))
# female score boxplot
sns.boxplot(data=df.loc([df['Gender'] == 'F', ["Score"]]))


# histogram
sns.distribution(df.CWDistance)
plt.show()


# cout plot
sns.countplot(x='Gender', data=df)
plt.xticks(rotation=-45)
plt.show()




# ============================ Data Visualization ==============================

import seaborn as sns
import matplotlib.pyplot as plt

# load the data
tips_data = sns.load_dataset("tips")    # is a default dataset within seaborn

# print the first few rows
tips_data.head()

# print the summary statistics of the quantitative variables
tips_data.describe()
# 	         total_bill	   tip	        size
# count	    244.000000	   244.000000	244.000000
# mean	    19.785943	   2.998279	    2.569672
# std	    8.902412	   1.383638	    0.951100
# min	    3.070000	   1.000000	    1.000000
# 25%	    13.347500	   2.000000	    2.000000
# 50%	    17.795000	   2.900000	    2.000000
# 75%	    24.127500	   3.562500	    3.000000
# max	    50.810000	   10.000000	6.000000

# creating histograms
## total_bill
sns.distplot(tips_data["total_bill"], kde=False).set_title("Histogram of Total Bill")
plt.show()

## total tip
sns.distplot(tips_data["tips"], kde=False).set_title("Histogram of Total Tip")
plt.show()

## both total_bill and tips
sns.distplot(tips_data["total_bill"], kde = False)
sns.distplot(tips_data["tip"], kde=False).set_title("Histogram of Both Tip Size and Total Bill")
plt.show()

# creating boxplots
## total_bill
sns.boxplot(tips_data["total_bill"]).set_title("Box plot of the Total Bill")
plt.show()

## tips
sns.boxplot(tips_data["tip"]).set_title("Box plot of the Tip")
plt.show()

# histograms and boxplots by groups
## tip if smoker
sns.boxplot(x = tips_data["tip"], y = tips_data["smoker"])
plt.show()

## boxplots and histograms of tips by time of time of day
sns.boxplot(x = tips_data["tip"], y = tips_data["time"])
# two similar boxes for types of time
g = sns.FacetGrid(tips_data, row = "time")
g = g.map(plt.hist, "tip")
plt.show()

## boxplots and histograms of tips by the day
sns.boxplot(x = tips_data["tip"], y = tips_data["day"])

g = sns.FacetGrid(tips_data, row = "day")
g = g.map(plt.hist, "tip")
plt.show()
