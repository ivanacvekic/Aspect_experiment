#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:24:54 2019

@author: ivanacvekic
"""
###---------- ASPECT PARTICIPANT INFORMATION ANALYSIS

## Importing pandas package
import pandas as pd
# importing scipy locally for the Shapiro-Wilk Test
from scipy import stats
# importing statistical models (ANOVA & Mann-Whitney U)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import mannwhitneyu

# Reading the participant file in .csv format
df = pd.read_csv("A_participants.csv", header=0, sep=",")

# find the keys of the dataframe
keys_of_dataFrame = df.keys()

# checking data types of all columns
df.dtypes

# changing 'Participant' column from integer to object
df.Participant = df.Participant.astype(object)

## Descriptives
# calculating descriptives for all columns
meanAll = df.describe()
# calculating descriptives (M, SD, min, max, etc.) by experimental Group
descriptives = df.groupby('Group').describe()
# calculating only means for all columns per Group
mean = df.groupby('Group').mean()

## Significance testing
# Using ANOVA to test significane between groups for:
# LexTALE - proficiency test in English
mod = ols('LexTALE ~ Group', data=df).fit()
LextTALE_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(LextTALE_ANOVA)
# CFT - category fluency task for proficiency in English
mod = ols('CFT ~ Group', data=df).fit()
CFT_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(CFT_ANOVA)
# Age of participants
mod = ols('Age ~ Group', data=df).fit()
Age_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(Age_ANOVA)
# Age of Acquisition of participants
mod = ols('AoA ~ Group', data=df).fit()
AoA_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(AoA_ANOVA)
# Years of learning English
mod = ols('Years ~ Group', data=df).fit()
Years_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(Years_ANOVA)
# Cloze Test - testing participants' knowledge of Aspect
mod = ols('ClozeTestP ~ Group', data=df).fit()
ClozeTestP_ANOVA = sm.stats.anova_lm(mod, typ=2)
print(ClozeTestP_ANOVA)

## Testing normality of the data for significant results
# using the Shapiro–Wilk test for normal distribution testing
LexTALE_N = stats.shapiro(df['LexTALE'])
Age_N = stats.shapiro(df['Age'])
AoA_N = stats.shapiro(df['AoA'])
ClozeTestP_N = stats.shapiro(df['ClozeTestP'])

# Creating subsets by Group for significance testing between-groups
S = df[df.Group == 'Spanish']
C = df[df.Group == 'Croatian']
G = df[df.Group == 'German']

## Significance testing between each Group pair
# in order to find out where the difference is

# German - Croatian
GC1 = mannwhitneyu(G['LexTALE'],C['LexTALE'])
GC2 = mannwhitneyu(G['Age'],C['Age'])
GC3 = mannwhitneyu(G['AoA'],C['AoA'])
GC4 = mannwhitneyu(G['ClozeTestP'],C['ClozeTestP'])

# German - Spanish
GS1 = mannwhitneyu(G['LexTALE'],S['LexTALE'])
GS2 = mannwhitneyu(G['Age'],S['Age'])
GS3 = mannwhitneyu(G['AoA'],S['AoA'])
GS4 = mannwhitneyu(G['ClozeTestP'],S['ClozeTestP'])

# Croatian - Spanish
CS1 = mannwhitneyu(C['LexTALE'],S['LexTALE'])
CS2 = mannwhitneyu(C['Age'],S['Age'])
CS3 = mannwhitneyu(C['AoA'],S['AoA'])
CS4 = mannwhitneyu(C['ClozeTestP'],S['ClozeTestP'])

#save in a fast format
df.to_pickle("A_GH.pkl")

###---------- End of the analysis