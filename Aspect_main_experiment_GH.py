#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:55:11 2019

@author: ivanacvekic
"""

###---------- ASPECT DATA ANALYSIS

# import pandas and numpy packages
import pandas as pd
import numpy as np
# importing packages for the linear mixed effects model, ANOVA and Mann-Whitney U tests
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu
from scipy import stats
# import seaborn for visualization
import seaborn as sns
from matplotlib import pyplot as plt


## definition of a function that helps us to calculate the
# avegerade and standard deviation per participant

def average_And_deviation(df, participand_id):
    
    # extracting only the part of the original data frame containing the
    # information about the pariticipant of interest
    local_df = df[df["Participant"]==participand_id]
    
    # create the list of columns we waant to look at
    list_of_columns = ["S1.RT","S2.RT","S3.RT","S4.RT","S5.RT",
                       "S6.RT","S7.RT","S8.RT","S9.RT","S10.RT","S11.RT","S12.RT",
                       "S13.RT","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
    # creating empty list
    raw_list = []
    
    
    
    # the cutout (local_df) for our huge dataframe still has the same index as the original data frame
    # we have get the indices of the lines, which are in the cutout
    # my_index lists of indices for the cutout
    myIndex = local_df.index
    
    # outer for-loop: going through all the lines of the local data frame,
    # which contain only the information about one participant
    for i in myIndex:
    
        # inner for-loop: going through the columns of interest
        for j in (list_of_columns):
           
            raw_list.append(local_df[j][i])
     
    # now, raw_list contains all values of interest of the participant
        
    # calcuating the average and standard deviation, excluding NaN values
    average = np.nanmean(raw_list)    
    standard_deviation = np.nanstd(raw_list)
    
    print (raw_list)
    return average, standard_deviation
    
###---------- definition of average calulation completed




###---------- GERMAN GROUP ANALYSIS
    
# Reading the participant file in .csv format
df1 = pd.read_csv("A_GER_ordered.csv", encoding = "ISO-8859-1", header=0, sep=",")

# creating 'Accuracy' column based on the correct responses and participants' answers
conditions = [
    (df1['Correct'] == 'x') & (df1['Response'] == 'x'),
    (df1['Correct'] == 'm') & (df1['Response'] == 'm')]
choices = [1, 1]
df1['Accuracy'] = np.select(conditions, choices, default=0)
print(df1)
# calculating the percentage of accurate responses
(df1.Accuracy.mean())*100
# calculating the percentage of inacurate responses
100-(df1.Accuracy.mean())*100
# saving the data frame with accuracy data before deleting inaccurate responses for analysis
Accuracy_G = df1
# dropping inacurate responses, because
# only the participants who paid attention during the experiment will be analyzed
df1 = df1[df1.Accuracy == 1]


# splitting the Conditions into factors for lmer
df1[['Aspect', 'Type']] = df1.Condition.str.rsplit("_", n = 1, expand=True,)

# replacing 0 with missing data
# a zero would falsely indicate the speed of the response
# and it would affect the analysis of reaaction times data
df1.replace(0, np.nan, inplace=True)

# adding index for columns
idx = df1.columns 
label = df1.columns[0] 
lst = df1.columns.tolist()

# checking data types
d1types = df1.dtypes

# changing column types from integer to object
df1.Participant = df1.Participant.astype(object)


#----- calulating average (M) and standard deviation (SD) for each participant

# getting the participant numbers from the data frame
participant_list = df1["Participant"].unique()

# create new, empty columns
df1["average"] = 0
df1["StandardDeviation"] = 0


# iteration for all participants
for i in participant_list:
    
    
    # call the function and calculate M and SD for the participant
    # with the number stored in variable i
    my_result  = average_And_deviation(df1,i)
    current_M  = my_result[0]
    current_SD = my_result[1] 
    
    
    # get the index that belongs to the current participant
    my_index = df1[df1["Participant"]==i].index

    # assinging to all lines of the M and SD column the value 
    # that the function gave us of the current participants
    df1.loc[my_index,"average"]           = current_M
    df1.loc[my_index,"StandardDeviation"] = current_SD

#----- outlier correction
list_of_columns = ["S1.RT","S2.RT","S3.RT","S4.RT","S5.RT",
                       "S6.RT","S7.RT","S8.RT","S9.RT","S10.RT","S11.RT","S12.RT",
                       "S13.RT","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
list_of_rows = df1.index

df1["outlier"] = 0

# iteration for all rows of the data frame

for i in list_of_rows:
    # iteration for all relevant columns
    for j in list_of_columns:
        
        # calculation of reference value of outlier correction: M + 2xSD
        reference_value = df1["average"][i] + 2*df1["StandardDeviation"][i]
        
        # if the specific value is larger than the reference, it is replaced by the reference
        if df1[j][i] > reference_value:
            df1.loc[i,j]           = reference_value
#----- outlier correction completed

## LMER
# Reading the Proficinecy file with 'LexTALE' and 'Cloze Test' columns in a .csv format
LexTALE = pd.read_csv("A_LexTALE.csv", encoding = "ISO-8859-1", header=0, sep=",")
# Reading the file with three segments (after outlier correction) for analysis in a .csv format
S1 = pd.read_csv("GER_with_segments.csv", encoding = "ISO-8859-1", header=0, sep=",")
# replacing 0 with missing data
S1.replace(0, np.nan, inplace=True)

# Merging a file with segments with the proficinecy information
S1 = pd.merge(S1, LexTALE, on='Participant')
# renaming column as 'Group'
S1.rename(columns={"Group_x": "Group"}, inplace=True)
#checking data types
S1.info(verbose=True)
# changing column types
S1.Item = S1.Item.astype(object)
S1.Participant = S1.Participant.astype(object)
S1.Trial = S1.Trial.astype(object)

## LMER models for L1 German group
# Region: Disambiguation
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Disambiguation ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S1)
mdf = md.fit(method="nm")
print(mdf.summary())
# Region: Spillover
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Spillover ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S1)
mdf = md.fit()
print(mdf.summary())
# Region: FInal
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Final ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S1)
mdf = md.fit()
print(mdf.summary())


## creating a new file out of an old one with only 3 areas or interest
# i.e., 'Disambiguation', 'Spillover' and 'Final'
df1_only_segments = S1.filter(['Group', 'Trial', 'Participant','Item', 'LexTALE', 'ClozeTestP', 'Aspect',
                      'Type', 'Disambiguation', 'Spillover', 'Final'])

# melt the segments columns into one column 
df1_all = pd.melt(df1_only_segments, id_vars =['Group', 'Trial', 'Participant', 'Item', 'ClozeTestP', 'LexTALE', 'Aspect',
                      'Type'], 
                  value_vars =['Disambiguation', 'Spillover', 'Final'])
# rename columns into: 'Segments' and 'RT' for lmer analysis
df1_all.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# Mixed model for all segments
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("RT ~ Segments + Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=df1_all)
mdf = md.fit()
print(mdf.summary())

## getting ready for visualizing the data
# new file with only 4 columns necessary for the graph
G_GraphA = S1.filter(['LexTALE', 'ClozeTestP', 'Condition', 'Disambiguation', 'Spillover', 'Final'])
# mean information per condition
meanGGA = G_GraphA.groupby('Condition').mean()
# all infromation (M, SD, min, max, etc.) for the paper
allGGA = G_GraphA.groupby('Condition').describe()
# 'Disambiguation', 'Spillover' and 'Final' melted into one column for plotting 
graph1 = pd.melt(G_GraphA, id_vars =['Condition', 'ClozeTestP', 'LexTALE'], value_vars =['Disambiguation', 'Spillover', 'Final'])
# renaming columns
graph1.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# a line plot for the L1 German group
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=graph1, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()


## median split of LexTALE  and Cloze Test because they had an effect on the results
# Lower LexTALE proficiency:
# dividing participants into low proficiency acording to the LexTALE median
lowG_LT = graph1[ graph1['LexTALE'] < 86.25 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=lowG_LT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# Higher LexTALE proficiency:
# dividing participants into high proficiency acording to the LexTALE median
highG_LT = graph1[ graph1['LexTALE'] > 86.25 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=highG_LT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

## median split of ClozeTest because it had an effect on the results

# Lower Cloze test proficiency:
# dividing participants into low proficiency acording to the Cloze Test media
lowG_CT = graph1[ graph1['ClozeTestP'] < 71.43 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=lowG_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# Higher Cloze test proficiency:
# dividing participants into high proficiency acording to the Cloze Test media
highG_CT = graph1[ graph1['ClozeTestP'] > 71.43 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=highG_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()
###---------- GERMAN GROUP ANALYSIS COMPLETED




###---------- CROATIAN GROUP ANALYSIS
# Reading the participant file in .csv format
df2 = pd.read_csv("A_CRO_ordered.csv", encoding = "ISO-8859-1", header=0, sep=",")

# creating 'Accuracy' column based on the correct responses and participants' answers
conditions = [
    (df2['Correct'] == 'x') & (df2['Response'] == 'x'),
    (df2['Correct'] == 'm') & (df2['Response'] == 'm')]
choices = [1, 1]
df2['Accuracy'] = np.select(conditions, choices, default=0)
print(df2)
# calculating the percentage of accurate responses
(df2.Accuracy.mean())*100
# calculating the percentage of inaccurate responses
100-(df2.Accuracy.mean())*100
# saving the data frame with accuracy data before deleting inaccurate responses for analysis
Accuracy_C = df2
# dropping inacurate responses, because
# only the participants who paid attention during the experiment will be analyzed
df2 = df2[df2.Accuracy == 1]


# splitting the Conditions into factors for lmer
df2[['Aspect', 'Type']] = df2.Condition.str.rsplit("_", n = 1, expand=True,)

# replacing 0 with missing data
# a zero would falsely indicate the speed of the response
# and it would affect the analysis of reaaction times data
df2.replace(0, np.nan, inplace=True)

# adding index for columns
idx = df2.columns 
label = df2.columns[0] 
lst = df2.columns.tolist()

# checking data types
d1types = df2.dtypes

# changing column types from integer to object
df2.Participant = df2.Participant.astype(object)


#----- calulating average (M) and standard deviation (SD) for each participant

# getting the participant numbers from the data frame
participant_list = df2["Participant"].unique()

# create new, empty columns
df2["average"] = 0
df2["StandardDeviation"] = 0


# iteration for all participants
for i in participant_list:
    
    
    # call the function and calculate M and SD for the participant
    # with the number stored in variable i
    my_result  = average_And_deviation(df2,i)
    current_M  = my_result[0]
    current_SD = my_result[1] 
    
    
    # get the index that belongs to the current participant
    my_index = df2[df2["Participant"]==i].index

    # assinging to all lines of the M and SD column the value 
    # that the function gave us of the current participants
    df2.loc[my_index,"average"]           = current_M
    df2.loc[my_index,"StandardDeviation"] = current_SD


#----- outlier correction
list_of_columns = ["S1.RT","S2.RT","S3.RT","S4.RT","S5.RT",
                       "S6.RT","S7.RT","S8.RT","S9.RT","S10.RT","S11.RT","S12.RT",
                       "S13.RT","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
list_of_rows = df2.index

df2["outlier"] = 0

# iteration over all rows of the data frame
for i in list_of_rows:
    # iteration over all relevant columns
    for j in list_of_columns:
        
        # calculation of reference value of outlier correction: M + 2xSD
        reference_value = df2["average"][i] + 2*df2["StandardDeviation"][i]
        
        # if the specific value is larger than the reference, it is replaced by the reference
        if df2[j][i] > reference_value:
            df2.loc[i,j]           = reference_value
#----- outlier correction completed
            
# Reading the file with three segments (after outlier correction) for analysis in a .csv format
S2 = pd.read_csv("CRO_with_segments.csv", encoding = "ISO-8859-1", header=0, sep=",")
# replacing 0 with missing data
S2.replace(0, np.nan, inplace=True)


# Merging a file with segments with the proficinecy information
S2 = pd.merge(S2, LexTALE, on='Participant')
# renaming column as 'Group'
S2.rename(columns={"Group_x": "Group"}, inplace=True)
# checking all data types
S2.info(verbose=True)
# changing data types
S2.Item = S2.Item.astype(object)
S2.Participant = S2.Participant.astype(object)
S2.Final = S2.Final.astype(float)

## LMER models for L1 Croatian group
# Region: Disambiguation
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Disambiguation ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S2)
mdf = md.fit()
print(mdf.summary())
# Region: Spillover
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Spillover ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S2)
mdf = md.fit()
print(mdf.summary())
# Region: Final
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Final ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S2)
mdf = md.fit()
print(mdf.summary())

## creating a new file out of an old one with only 3 areas or interest
# i.e., 'Disambiguation', 'Spillover' and 'Final'
df2_only_segments = S2.filter(['Group', 'Trial', 'Participant', 'Item', 'LexTALE', 'ClozeTestP', 'Aspect',
                      'Type', 'Disambiguation', 'Spillover', 'Final'])
# melt the segments columns into one column 
df2_all = pd.melt(df2_only_segments, id_vars =['Group', 'Trial', 'Participant', 'Item', 'ClozeTestP', 'LexTALE', 'Aspect',
                      'Type'], 
                  value_vars =['Disambiguation', 'Spillover', 'Final'])
# rename columns into: 'Segments' and 'RT'
df2_all.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)


# Mixed model for all segments
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("RT ~ Segments + Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=df2_all)
mdf = md.fit()
print(mdf.summary())


## getting ready for visualizing the data
# new file with only 4 columns necessary for the graph
C_GraphA = S2.filter(['LexTALE', 'ClozeTestP', 'Condition', 'Disambiguation', 'Spillover', 'Final'])
# mean information per condition
meanCGA = C_GraphA.groupby('Condition').mean()
# all infromation (M, SD, min, max, etc.) for the paper
allCGA = C_GraphA.groupby('Condition').describe()
# 'Disambiguation', 'Spillover' and 'Final' melted into one column for plotting 
graph2 = pd.melt(C_GraphA, id_vars =['Condition','LexTALE', 'ClozeTestP'], value_vars =['Disambiguation', 'Spillover', 'Final'])
# renaming columns
graph2.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# a line plot for the L1 Croatian group
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=graph2, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

## median split of LexTALE  and Cloze Test because they had an effect on the results
# Lower LexTALE proficiency:
# dividing participants into low proficiency acording to the LexTALE median
lowC_LT = graph2[ graph2['LexTALE'] < 85 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=lowC_LT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# Higher LexTALE proficiency:
# dividing participants into high proficiency acording to the LexTALE median
highC_LT = graph2[ graph2['LexTALE'] > 85 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=highC_LT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()


# Lower Cloze Test proficiency:
# dividing participants into low proficiency acording to the Cloze Test median
lowC_CT = graph2[ graph2['ClozeTestP'] < 85.71 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=lowC_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# Lower Cloze Test proficiency:
# dividing participants into low proficiency acording to the Cloze Test median
highC_CT = graph2[ graph2['ClozeTestP'] > 85.71 ]
#graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=highC_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()
###---------- CROATIAN GROUP ANALYSIS COMPLETED




###---------- SPANISH GROUP ANALYSIS
# Reading the participant file in .csv format
df3 = pd.read_csv("A_ESP_ordered.csv", encoding = "ISO-8859-1", header=0, sep=",")


# creating 'Accuracy' column based on the correct responses and participants' answers
conditions = [
    (df3['Correct'] == 'x') & (df3['Response'] == 'x'),
    (df3['Correct'] == 'm') & (df3['Response'] == 'm')]
choices = [1, 1]
df3['Accuracy'] = np.select(conditions, choices, default=0)
print(df3)
# calculating the percentage of accurate responses
(df3.Accuracy.mean())*100
# calculating the percentage of inaccurate responses
100-(df3.Accuracy.mean())*100
# saving the data frame with accuracy data before deleting inaccurate responses for analysis
Accuracy_S = df3
# dropping inacurate responses, because
# only the participants who paid attention during the experiment will be analyzed
df3 = df3[df3.Accuracy == 1]


# splitting the Conditions into factors for lmer
df3[['Aspect', 'Type']] = df3.Condition.str.rsplit("_", n = 1, expand=True,)

# replacing 0 with missing data
# a zero would falsely indicate the speed of the response
# and it would affect the analysis of reaaction times data
df3.replace(0, np.nan, inplace=True)

# adding index for columns
idx = df3.columns 
label = df3.columns[0] 
lst = df3.columns.tolist()

# checking data types
d3types = df3.dtypes
# changing column types from integer to object
df3.Participant = df3.Participant.astype(object)


#----- calulating average (M) and standard deviation (SD) for each participant

# getting the participant number from the data frame
participant_list = df3["Participant"].unique()

# create new, empty columns
df3["average"] = 0
df3["StandardDeviation"] = 0


# iteration for all participants
for i in participant_list:
    
    
    # call the function and calculate M and SD for the participant
    # with the number stored in variable i
    my_result  = average_And_deviation(df3,i)
    current_M  = my_result[0]
    current_SD = my_result[1] 
    
    
    # get the index belongs to the current participant
    my_index = df3[df3["Participant"]==i].index

    # assinging to all lines of the M and SD column the value 
    # that the function gave us of the current participants
    df3.loc[my_index,"average"]           = current_M
    df3.loc[my_index,"StandardDeviation"] = current_SD

df3.to_excel("text_results_CRO.xlsx")

#----- outlier correction
list_of_columns = ["S1.RT","S2.RT","S3.RT","S4.RT","S5.RT",
                       "S6.RT","S7.RT","S8.RT","S9.RT","S10.RT","S11.RT","S12.RT",
                       "S13.RT","S14.RT","S15.RT","S16.RT","S17.RT",
                       "S18.RT","S19.RT","S20.RT","S21.RT",
                       "S22.RT","S23.RT","S24.RT","S25.RT","S26.RT",
                       "S27.RT","S28.RT","S29.RT","S30.RT","S31.RT",
                       "S32.RT","S33.RT","S34.RT","S35.RT"]
list_of_rows = df3.index

df3["outlier"] = 0

# iteration over all rows of the data frame
for i in list_of_rows:
    # iteration over all relevant columns
    for j in list_of_columns:
        
        # calculation of reference value: M + 2xSD
        reference_value = df3["average"][i] + 2*df3["StandardDeviation"][i]
        
        # if the specific value is larger than the reference, it is replaced by the reference
        if df3[j][i] > reference_value:
            df3.loc[i,j]           = reference_value
#----- outlier correction completed
            
# Reading the file with three segments (after outlier correction) for analysis in a .csv format
S3 = pd.read_csv("SPA_with_segments.csv", encoding = "ISO-8859-1", header=0, sep=",")
# replacing 0 with missing data
S3.replace(0, np.nan, inplace=True)

# Merging a file with segments with the proficinecy information
S3 = pd.merge(S3, LexTALE, on='Participant')
# renaming a column
S3.rename(columns={"Group_x": "Group"}, inplace=True)
# checking data types
S3.info(verbose=True)
# changing data types
S3.Item = S3.Item.astype(object)
S3.Participant = S3.Participant.astype(object)

## LMER models for L1 Croatian group
# Region: Disambiguation
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Disambiguation ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S3)
mdf = md.fit()
print(mdf.summary())
# Region: Spillover
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Spillover ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S3)
mdf = md.fit()
print(mdf.summary())
# Region: Final
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("Final ~ Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=S3)
mdf = md.fit()
print(mdf.summary())

## creating a new file out of an old one with only 3 areas or interest
# i.e., 'Disambiguation', 'Spillover' and 'Final'
df3_only_segments = S3.filter(['Group', 'Trial', 'ClozeTestP', 'Participant', 'Item', 'LexTALE', 'Aspect',
                      'Type', 'Disambiguation', 'Spillover', 'Final'])
df3_only_segments= pd.DataFrame(df3_only_segments)

# melt the segments columns into one column 
df3_all = pd.melt(df3_only_segments, id_vars =['Group', 'Trial', 'ClozeTestP', 'Participant', 'Item', 'LexTALE', 'Aspect',
                      'Type'], 
                  value_vars =['Disambiguation', 'Spillover', 'Final'])
#  rename columns into: 'Segments' and 'RT'
df3_all.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)

# Mixed model for all segments
vcf = {"Item": "0 + C(Item)", "Participant": "0 + C(Participant)"}                                                         
md = smf.mixedlm("RT ~ Segments + Aspect*Type*scale(LexTALE)*scale(ClozeTestP)", groups="Group",                                                    
                                vc_formula=vcf, re_formula="0", missing='drop', data=df3_all)
mdf = md.fit()
print(mdf.summary())


## getting ready for visualizing the data
# new file with only 4 columns necessary for the graph
S_GraphA = S3.filter(['LexTALE', 'ClozeTestP', 'Condition', 'Disambiguation', 'Spillover', 'Final'])
# mean information per condition
meanSGA = S_GraphA.groupby('Condition').mean()
# all infromation (M, SD, min, max, etc.) for the paper
allSGA = S_GraphA.groupby('Condition').describe()
# 'Disambiguation', 'Spillover' and 'Final' melted into one column for plotting 
graph3 = pd.melt(S_GraphA, id_vars =['LexTALE', 'ClozeTestP', 'Condition'], value_vars =['Disambiguation', 'Spillover', 'Final'])
# renaming columns
graph3.rename(columns={'variable':'Segments', 'value': 'RT'}, inplace=True)


# a line plot for the L1 Spanish group
Legend = sns.lineplot(y='RT', x='Segments', 
                 data=graph3, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

## median split of LexTALE  and Cloze Test because they had an effect on the results
# Lower LexTALE proficiency:
# dividing participants into low proficiency acording to the LexTALE median
lowS_LT = graph3[ graph3['LexTALE'] < 71.25 ]
legend = sns.lineplot(y='RT', x='Segments', 
                 data=lowS_LT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()


# High LexTALE proficiency:
# dividing participants into high proficiency acording to the LexTALE median
highS_LT = graph3[ graph3['LexTALE'] > 71.25 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=highS_LT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()


# median split of ClozeTest because it had an effect on the results
# Lower Cloze Test proficiency:
lowS_CT = graph3[ graph3['ClozeTestP'] < 78.57 ]
legend = sns.lineplot(y='RT', x='Segments', 
                 data=lowS_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# Higher Cloze Test proficiency:
highS_CT = graph3[ graph3['ClozeTestP'] > 78.57 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=highS_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()
###---------- SPANISH GROUP ANALYSIS COMPLETED



###---------- PLOTTING THE RESULTS OF ALL GROUPS TOGETHER
# merging the three Group files
all_graph = pd.concat([graph1,graph2,graph3], axis=0)
# graph 
legend = sns.lineplot(y='RT', x='Segments', 
                 data=all_graph, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()


# Lower Cloze Test proficiency median split
low_all_CT = all_graph[ all_graph['ClozeTestP'] < 82.14 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=low_all_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()

# Higher Cloze Test proficiency median split
high_all_CT = all_graph[ all_graph['ClozeTestP'] > 82.14 ]
# graph
legend = sns.lineplot(y='RT', x='Segments', 
                 data=high_all_CT, 
                 palette="colorblind",
                 hue='Condition', err_style="bars", sort=False)
Legend.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=False, ncol=1)
plt.show()
###---------- PLOTTING THE RESULTS OF ALL GROUPS COMPLETED



###---------- ANALYZING THE ACCURACY PERCENTAGE BETWEEN-GROUPS
# combine the three groups and their accuracies in one dataset
Accuracy = pd.concat([Accuracy_G, Accuracy_C,Accuracy_S], axis=0)
# all infromation (M, SD, min, max, etc.)
descriptives = Accuracy.groupby('Group').describe()
# ANOVA
mod = ols('Accuracy ~ Group', data=Accuracy).fit()
ACC_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(ACC_ANOVA)
# testing normality
AccuracyNORM = stats.shapiro(Accuracy['Accuracy'])
AGC = mannwhitneyu(Accuracy_G['Accuracy'],Accuracy_C['Accuracy'])
AGS = mannwhitneyu(Accuracy_G['Accuracy'],Accuracy_S['Accuracy'])
ACS = mannwhitneyu(Accuracy_C['Accuracy'],Accuracy_S['Accuracy'])
###---------- ANALYZING ACCURACY COMPLETED


###---------- End of the analysis