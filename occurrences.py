import pandas as pd

def occurrences(df):

	df["% Plain"] = df["Plain Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Cool"] = df["Cool Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Warm"] = df["Warm Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Cold"] = df["Cold Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Heavy Cold"] = df["Heavy Cold Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Heavy Warm"] = df["Heavy Warm Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Hot"] = df["Hot Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

	df["% Heavy Hot"] = df["Heavy Hot Occurences"]/(df["Plain Occurences"] + df["Cool Occurences"] + df["Warm Occurences"] + df["Cold Occurences"] + df["Heavy Cold Occurences"] + df["Heavy Warm Occurences"] + df["Hot Occurences"] + df["Heavy Hot Occurences"])

def linear_scale(df):

	df["Linear Scale"] = (df["% Plain"] * (3.0/7.0)) + (df["% Cool"] * (2.0/7.0)) + (df["% Warm"] * (4.0/7.0)) + (df["% Cold"] * (1.0/7.0)) + (df["% Heavy Cold"] * (0.0/7.0)) + (df["% Heavy Warm"] * (5.0/7.0)) + (df["% Hot"] * (6.0/7.0)) + (df["% Heavy Hot"] * (7.0/7.0))



