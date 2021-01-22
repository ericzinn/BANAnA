"""Main workflow of the BANAnA pipeline

Copyright (c) 2020 Eric Zinn

This code is free software;  you can redstribute it and/or modify it under the terms of the AGPL
license (see LICENSE file included in the distribution)
"""


# imports
import argparse
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, integrate
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
import arviz as az
from matplotlib import cm
from sklearn.metrics import r2_score
import yaml
import pickle

# import BANAnA modules
import MCMC
import figurePlotting
import dataLogging
import correlationHelper

# Begin defining useful functions


def filterBarcodesByMean(df):
    """Return a filter which is True if there are more counts in a barcode than 
     there are average counts of every non-zero barcode."""
    return (df).sum(axis = 1) >= (df.sum(axis = 1).sum() / len(df.index))

# Function to makeFCsByPosition


def makeFCByPosition(dataFrame, prefix, method = 'Average'):
    '''Returns a transposed dataframe where rows represent experimental conditions and columns
        represent statistical moments (Arithmetic Mean and SEM) of viruses grouped by position and
        amino acid identity'''

    # Cache a list of experimental conditions from the passed dataframe
    listOfConditions = [column for column in dataFrame.columns.values.tolist() if column.endswith("FC")]
    # Initialize an empty dataframe with rows = experimental conditions
    tempNetFC = pd.DataFrame(index = listOfConditions)

    # Make a new list and propagate it with all unique states at every
    # position in the dataFrame's barcod (as lists)
    # Example (Anc80): [['K', 'R'], ['A', 'S'],...etc.
    listOfSets = []
    for i in range(len(dataFrame['Barcode'].iloc[0])):
        listOfSets.append(list(set([barcode[i] for barcode in dataFrame['Barcode']])))

    if method == 'Average':
        # Iterate over each possible state in each possible position and
        # make two new columns to hold the Mean and SEM of each state/position
        # for every experimental condition in the dataframe
        # Eg. All PS_Means averaged for P1K, P1R, P2A, P2S, etc.
        for position, listOfStates in enumerate(listOfSets):
            for state in listOfStates:
                listOfBarcodes = [prefix + str(barcode) for barcode in dataFrame['Barcode'] if barcode[position] == state]
                tempNetFC['p' + str(position + 1) + state + '_Mean'] = [dataFrame[condition].loc[listOfBarcodes].mean() for condition in listOfConditions]
                tempNetFC['p' + str(position + 1) + state + '_SEM'] = [dataFrame[condition].loc[listOfBarcodes].sem() for condition in listOfConditions]
                tempNetFC['p' + str(position + 1) + state + '_N'] = [dataFrame[condition].loc[listOfBarcodes].count() for condition in listOfConditions]

        # Add two final columns to the dataframe averaging over ALL barcodes
        # regardless of state/position for each condition in the dataframe
        tempNetFC['net_Mean'] = [dataFrame[condition].mean() for condition in listOfConditions]
        tempNetFC['net_SEM'] = [dataFrame[condition].sem() for condition in listOfConditions]
        tempNetFC['net_N'] = [dataFrame[condition].count() for condition in listOfConditions]

    elif method == 'Total':
        listOfConditions = configYAML['listOfDilutions']
        # Initialize an empty dataframe with rows = experimental conditions
        tempNetFC = pd.DataFrame(index = listOfConditions)

        listOfSets = []
        for i in range(len(dataFrame.index.values[0].split('_')[-1])):
            listOfSets.append(list(set([barcode[i] for barcode in [x.split('_')[-1] for x in dataFrame.index.values]])))

        # Iterate over each possible state in each possible position and
        # make a new column to hold the summed total counts of each state/posiiton
        # then subtract from a reference
        for position, listOfStates in enumerate(listOfSets):
            for state in listOfStates:
                listOfBarcodes = [prefix + str(barcode) for barcode in [x.split('_')[-1] for x in dataFrame.index.values] if barcode[position] == state]
                #tempNetFC['p' + str(position + 1) + state + '_Total'] = [np.log2(np.nanmean(np.nansum(2**dataFrame[[x for x in dataFrame.columns if condition in x]].loc[listOfBarcodes]))) for condition in listOfConditions]
                tempNetFC['p' + str(position + 1) + state + '_Total'] = [np.log2((2**dataFrame[[x for x in dataFrame.columns if condition in x]].loc[listOfBarcodes]).sum() + 1e-10).mean() for condition in listOfConditions]
        tempNetFC['net_Total'] = [np.log2((2**dataFrame[[x for x in dataFrame.columns if condition in x]]).sum() + 1e-10).mean() for condition in listOfConditions]        
        tempNetFC = (tempNetFC - tempNetFC.T["NS"]).drop('NS')
    return tempNetFC


# Start Parser stuff    
parser = argparse.ArgumentParser()

parser.add_argument("-p", "--positional_variation", type = str, default = "No",
                    help = "Perform analysis considering positions of variation in libraries?")
parser.add_argument("--pickles", action='store_true', 
                    help = "Load the pickled models/fits from a previous fit")
parser.add_argument("--debug", action = "store_true",
                    help = "Eric's debugging flag")

requiredNamed = parser.add_argument_group("Required Named Arguments")
requiredNamed.add_argument("-i", "--input_directory", type = str,
                           help = "Path of the input DIRECTORY (with subdirs of samples)")
requiredNamed.add_argument("-c", "--config_file", type = str,
                           help = "Path to experiment config file (see README)")
requiredNamed.add_argument("-o", "--output_directory", type = str,
                           help = "Path to where you want the output to live")

args = vars(parser.parse_args())

# Load experimental file into memory
with open(args["config_file"]) as experimentConfig:
    configYAML = yaml.load(experimentConfig, Loader=yaml.FullLoader)

# Initialize some empty dataframes and lists for directory walking
bigDataFrame = pd.DataFrame()
totalDataFrame = pd.DataFrame()
tempTotalDataFrame = pd.DataFrame()

barcodeFileList = []
dataFrameList = []

# Set up necessary subdirectories within output folder
dataLogging.makeDataSubfolders(args["output_directory"])
figurePlotting.makeFigureSubfolders(args["output_directory"])

# Walk through experimental directory, loading and aggregating csv files into
# larger dataframes
for library in configYAML["listOfLibraries"]:
    for dirpath, dirnames, filenames in os.walk(args["input_directory"]):
        for filename in [f for f in filenames if library in f]:
            barcodeFileList.append(os.path.join(dirpath, filename))

    for libraryCSV in barcodeFileList:
        if os.path.getsize(libraryCSV) > 0:
            dataFrameList.append(pd.read_csv(libraryCSV, index_col=0, names = ['Barcode', os.path.basename(libraryCSV).split(".")[0]]))

    if dataFrameList:
        tempDataFrame = pd.concat(dataFrameList, axis=1)
        # Sum all of the columns from the same condition (different lanes) together
        tempDataFrame = tempDataFrame.groupby(tempDataFrame.columns, axis=1).sum().replace(0, np.nan)
        for column in tempDataFrame:
            tempTotalDataFrame = pd.concat([tempTotalDataFrame, pd.DataFrame(columns=[column], index=[library], data=[tempDataFrame[column].sum()])], axis = 1)
        bigDataFrame = bigDataFrame.append(tempDataFrame)
        totalDataFrame = totalDataFrame.append(tempTotalDataFrame)

    # Reset/Clear variables for next lib
    tempTotalDataFrame = pd.DataFrame()
    barcodeFileList = []
    dataFrameList = []

# drop bad samples (if there are any)
if configYAML['badSamples']:
    totalDataFrame.drop(columns = configYAML['badSamples'], inplace = True)
    bigDataFrame.drop(columns = configYAML['badSamples'], inplace = True)

# Prune 'control' barcodes to only those which were in the yaml file
barcodesToDrop = [bc for bc in bigDataFrame.index.to_list() if "Control" in bc and bc not in configYAML["listOfControls"]]

if barcodesToDrop:
    bigDataFrame.drop(barcodesToDrop, inplace = True)

# Data Logging
dataLogging.logBigDf(args["output_directory"], bigDataFrame)
dataLogging.logTotalDf(args["output_directory"], totalDataFrame)

# Initialize two empty lists.  One to hold all of the unique indices, the
# other to hold all of the dataframes associated with those indices

indexList, rawCountsDFList = [], []

for iterator, library in enumerate(configYAML['listOfLibraries']):
    # Add list of indices
    indexList.append([x for x in bigDataFrame.index.tolist() if x.startswith(library)])
    # Make a temporary dataframe which is a subset of bigDataFrame
    # for the sake of code-readability
    tmpDf = bigDataFrame.loc[indexList[iterator]]

    if library == "Control":
        rawCountsDFList.append(tmpDf)
    else:
        # print("Filtering out counts from "+library+" if barcode counted less than mean ("+str(tmpDf.sum(axis = 1).sum()/len(tmpDf.index))+")")
        # rawCountsDFList.append(tmpDf[filterBarcodesByMean(tmpDf)])
        rawCountsDFList.append(tmpDf)

filteredDataframe = pd.concat(rawCountsDFList)

# Data Logging here...
dataLogging.logFilteredDf(args["output_directory"], filteredDataframe)

# Normalize everything to the relative abundances of Spike in Controls...and convert to pseudocounts
cpSpike = (filteredDataframe + 0.5) / (filteredDataframe.loc[configYAML["spikeInBarcode"]] + 1)
cpSpike = cpSpike.drop(configYAML["spikeInBarcode"])

# Rename the spike-in barcode (row) to "Spike-in"
cpSpike = cpSpike.rename(index = {configYAML["spikeInBarcode"]: "Spike-in"})

# Log2 Transform that Data
log2cpSpike = np.log2(cpSpike)
# Drop the spike-in from the filteredDataframe
filteredDataframe.drop(configYAML["spikeInBarcode"])

# Data Logging Here...
dataLogging.logNormalizedDf(args["output_directory"], cpSpike)
dataLogging.logLog2NormalizedDf(args["output_directory"], log2cpSpike)

# Refresh the list of indices
indexList = []
for iterator, library in enumerate(configYAML['listOfLibraries']):
    indexList.append([x for x in log2cpSpike.index.tolist() if x.startswith(library)])

# Make a list with the samples arranged from highest serum to lowest
newOrder = []
for dilution in configYAML["listOfDilutions"]:
    for sampleName in [x for x in sorted(cpSpike.columns.to_list()) if dilution in x]:
        newOrder.append(sampleName)


# Iterate over each of the libs in the list and summarize with the total number of normalized counts in each
dfList = []
for library in configYAML['listOfLibraries']:
    if library != "Control":
        dfList.append(cpSpike.loc[[x for x in cpSpike.index if x.startswith(library)]].sum().to_frame(name = library))
    else:
        for control in [x for x in cpSpike.index if x.startswith(library)]:
            dfList.append(cpSpike.loc[control].to_frame(name = control.split("_")[-1]))

summaryOfNormalizedCounts = pd.concat(dfList, axis = 1).iloc[:, ::-1].loc[newOrder]
# reorder the columns according to list in config
if configYAML['colorDict']:
    summaryOfNormalizedCounts = summaryOfNormalizedCounts[[x[0] for x in configYAML['colorDict'].items()]]

# Data Logging
dataLogging.logSummaryNormDf(args["output_directory"], summaryOfNormalizedCounts)

# Average number of normalized log2 counts over all replicates of a dilution
for dilution in configYAML['listOfDilutions']:
    log2cpSpike[dilution + "_Mean"] = log2cpSpike[[x for x in log2cpSpike.columns.to_list() if dilution in x]].mean(axis = 1)

# Calculate fold changes between conditions by subtracting out the No-Serum controls
for dilution in configYAML['listOfDilutions']:
    if dilution == "NS":
        continue
    else:
        log2cpSpike[dilution + '_FC'] = log2cpSpike[dilution + "_Mean"] - log2cpSpike["NS_Mean"]

# Make a new column with JUST the barcode of each row
log2cpSpike['Barcode'] = [x.split('_')[-1] for x in log2cpSpike.index]

# Load these puppies into a new list of dataFrames for later
log2FCList = []
for iterator, library in enumerate(configYAML['listOfLibraries']):
    log2FCList.append(log2cpSpike[[x for x in log2cpSpike.columns.to_list() if x.endswith('FC')] + ['Barcode']].loc[indexList[iterator]])

# Data Logging
dataLogging.logLog2FCList(args["output_directory"], log2FCList)

# Calculate concentrations of AB from dilutions
concentrationList = [1 / int(x) for x in configYAML['listOfDilutions'] if x != "NS"]

# Make Log2 Fold Change by Position Dataframes:
log2foldChangeByPositionList = []

# Calculate the FCByPosition using the AVERAGE fold change of each
# barcode in each library
for iterator, library in enumerate(configYAML['listOfLibraries']):
    if library != "Control":
        tmpDf = log2cpSpike[[x for x in log2cpSpike.columns.to_list() if x.endswith('FC')] + ['Barcode']].loc[indexList[iterator]]
        log2foldChangeByPositionList.append(makeFCByPosition(tmpDf, library + "BC_"))
        log2foldChangeByPositionList[iterator].index = concentrationList
    else:
        log2foldChangeControls = log2cpSpike[[x for x in log2cpSpike.columns.to_list() if x.endswith('FC')] + ['Barcode']].loc[indexList[iterator]].T.drop('Barcode')
        log2foldChangeControls.index = concentrationList

# Calculate the FCByPosition using the TOTAL counts of each
# barcode in each library
log2foldChangeByPositionTotalList = []
for iterator, library in enumerate(configYAML['listOfLibraries']):
    if library != "Control":
        tmpDf = log2cpSpike[[x for x in log2cpSpike.columns if not x.endswith('Mean') and not x.endswith('FC')]].loc[indexList[iterator]]
        log2foldChangeByPositionTotalList.append(makeFCByPosition(tmpDf, library + "BC_", method="Total").replace(-np.inf, np.nan))
        log2foldChangeByPositionTotalList[iterator].index = concentrationList

# Make a Summary Dataframe for avg fold Changes
log2avgLibsDataFrame = pd.DataFrame(index = log2foldChangeByPositionList[0].index.values)
for iterator, df in enumerate(log2foldChangeByPositionList):
    log2avgLibsDataFrame[configYAML['listOfLibraries'][iterator] + "_Net_mean"] = df.loc[:, "net_Mean"]
    log2avgLibsDataFrame[configYAML['listOfLibraries'][iterator] + "_Net_SEM"] = df.loc[:, "net_SEM"]
    log2avgLibsDataFrame[configYAML['listOfLibraries'][iterator] + "_Net_N"] = df.loc[:, "net_N"]

# Make a Summary Dataframe for total fold Changes
log2totalLibsDataFrame = pd.DataFrame(index = log2foldChangeByPositionTotalList[0].index.values)
for iterator, df in enumerate(log2foldChangeByPositionTotalList):
    log2totalLibsDataFrame[configYAML['listOfLibraries'][iterator] + "_Total"] = df.loc[:, "net_Total"]

# Data Logging 
dataLogging.logLog2AvgLibsDf(args["output_directory"], log2avgLibsDataFrame) 
dataLogging.logTotalLibsDf(args["output_directory"], log2totalLibsDataFrame)
dataLogging.logLog2FCControls(args["output_directory"], log2foldChangeControls)

# Now we raise 2**everything for the sake of fitting a curve (note: could also fit the log transform in theory by altering the func)
netLibsDataFrame = pd.DataFrame()
for column in [x for x in log2avgLibsDataFrame if x.endswith("mean")]:
    netLibsDataFrame[column] = 2**log2avgLibsDataFrame[column]


foldChangeControls = 2**log2foldChangeControls
totalLibsDataFrame = 2**log2totalLibsDataFrame

totalFCDataFrame = pd.concat([totalLibsDataFrame, foldChangeControls.astype(float)], axis = 1)
netFCDataFrame = pd.concat([netLibsDataFrame, foldChangeControls.astype(float)], axis = 1)
# Data Logging
dataLogging.logNetFCDf(args["output_directory"], netFCDataFrame)
dataLogging.logTotalFCDf(args["output_directory"], totalFCDataFrame)


if args["pickles"]:
    # Check to see if we're expecting this model to already have been run/fit:
    fitList = []
    for column in totalFCDataFrame.columns.to_list():
        with open(args["output_directory"] + "/intermediateData/curveFitting/pickles/" + column + ".pickle", "rb") as f:
            print("Loading pickle: " + args["output_directory"] + "/intermediateData/curveFitting/pickles/" + column + ".pickle")
            model_fit_dict = pickle.load(f)

        fitList.append(model_fit_dict["fit"])
else:
    # Initialize the model
    MCMCModel = MCMC.initializeInvLogitModel()
    # Now let's iterate through netFCDataFrame and fit for each column
    fitList = [MCMC.fitData(totalFCDataFrame[column].dropna(), totalFCDataFrame[column].dropna().index, MCMCModel) for column in totalFCDataFrame.columns.to_list()]

# Log These Fits
dataLogging.outputListOfFits(args["output_directory"], fitList, totalFCDataFrame.columns.to_list())
# Pickle the fits
if not args["pickles"]:
    # If we aren't loading pickles, then dump the models and fits to pickles
    dataLogging.pickleModelAndFits(MCMCModel, fitList, totalFCDataFrame.columns.to_list(), args["output_directory"])

# Export the BayesEstaimtes as a single table
parameterDF = dataLogging.logBayesEstimates(fitList, totalFCDataFrame, args["output_directory"])


# Now for correlation stuff...
print("Loading user specified Alignment...")
# Load the user-specified alignment
alignment = correlationHelper.loadFastaAlignment(configYAML["pathToAlignment"])
# Split the alignment into indiv columns
alignment = correlationHelper.splitAlignment(alignment)
# Make dummy variables using pandas' built-in routine
dum = pd.get_dummies(alignment)
# Make a pandas series containing the column names and the number of unique values per column
nunique = dum.apply(pd.Series.nunique)
# Drop all columns with only 1 unique entry 
dum.drop(nunique[nunique == 1].index, axis=1, inplace= True)
# Add EC50 to dum
dum["EC50"] = parameterDF['Mean EC50']
# Log Transform
dum["log10(EC50)"] = np.log10(dum["EC50"])
# Make list of correlations
correlationList = correlationHelper.makeCorrelationList(dum)
# Drop the uninformative entries in the list
correlationList.drop(["EC50", "log10(EC50)"], inplace = True)
# Log list of correlations
dataLogging.logCorr(correlationList, args["output_directory"])

# Generate a list of B-Factors based on the correlations
bFactors = correlationHelper.exportListOfBFactors(
    configYAML["pathToByPosAlignment"], configYAML["referenceSequence"], configYAML["sampleName"], correlationList["corr"],
    firstPositionInStructure = configYAML["firstPosInStruc"])
# And generate a pymol script to visualize those b-factors
dataLogging.makePyMolScript(bFactors, configYAML["referenceStructure"], configYAML["referenceSequence"], 
                            configYAML["sampleName"], args["output_directory"])

if not args["debug"]:
    # Figure Plot: Stacked Bargraph
    figurePlotting.plotStackedBargraph(summaryOfNormalizedCounts, args["output_directory"], 
                                       colorlist = [x[1] for x in configYAML['colorDict'].items()])
    # Make a pretty boxplot of the BayesEstimates
    figurePlotting.makeBoxPlot(fitList, totalFCDataFrame.columns.to_list(), "logEC50", [x[1] for x in configYAML['colorDict'].items()], args["output_directory"])
    figurePlotting.makeBoxPlot(fitList, totalFCDataFrame.columns.to_list(), "Hillslope", [x[1] for x in configYAML['colorDict'].items()], args["output_directory"])

    # Figure PLot: Observed Library Neutralization
    figurePlotting.plotLibraryNeutralization(totalFCDataFrame, args["output_directory"], 
                                             colorDict = configYAML["colorDict"], cleanNames = True) 
    # Figure Plot: Fit Curves
    for iterator, column in enumerate(totalFCDataFrame.columns.to_list()):
        if configYAML["colorDict"]:
            if "Total" in column:
                baseName = column.split("_")[0]
            if "Control" in column:
                baseName = column.split("_")[1]

            figurePlotting.plotMCMCFit(fitList[iterator], totalFCDataFrame[column], 
                                       totalFCDataFrame.index.values, color = configYAML["colorDict"][baseName], 
                                       saveName = args["output_directory"] + "/Figures/Fitting Curves/" + column)
        else:
            figurePlotting.plotMCMCFit(fitList[iterator], totalFCDataFrame[column], 
                                       totalFCDataFrame.index.values, color = cm.tab20(iterator), 
                                       saveName = args["output_directory"] + "/Figures/Fitting Curves/" + column)

        # Make ARVIZ Plots
        figurePlotting.plotMCMCTraces(fitList, totalFCDataFrame, args["output_directory"])

# Check to see if we're doing positional variation fits
if args["positional_variation"] == "Yes":
    # Let's start by combining all of the dataframes and transforming the fold changes
    # from log to linear (by rasing 2**(log2FC))

    tmpDfList = []

    for iterator, frame in enumerate(log2foldChangeByPositionTotalList):
        tmpDfList.append(log2foldChangeByPositionTotalList[iterator][[x for x in log2foldChangeByPositionTotalList[iterator] if x.endswith('Total')]].add_prefix(configYAML['listOfLibraries'][iterator] + "_"))

    log2foldChangeByPositionTotal = pd.concat(tmpDfList, axis =1)
    foldChangeByPositionTotal = 2**log2foldChangeByPositionTotal

    # Data Logging
    dataLogging.logLogFCByPosTotal(args["output_directory"], log2foldChangeByPositionTotal)
    dataLogging.logFCByPosTotal(args["output_directory"], foldChangeByPositionTotal)
    if args["pickles"]:
        # Check to see if we're expecting this model to already have been run/fit:
        fitListByPos = []
        for column in foldChangeByPositionTotal.columns.to_list():
            with open(args["output_directory"] + "/intermediateData/curveFittingByLibraryPosition/pickles/" + column + ".pickle", "rb") as f:
                print("Loading pickle: " + args["output_directory"] + "/intermediateData/curveFittingByLibraryPosition/pickles/" + column + ".pickle")
                model_fit_dict = pickle.load(f)

            fitListByPos.append(model_fit_dict["fit"])
    else:
        # Now let's iterate through foldChangeLibrariesByPosition and fit for each column
        # Note that this block of code will take some SERIOUS time to run depending on the hardware you're running it on.
        fitListByPos = [MCMC.fitData(foldChangeByPositionTotal[column].dropna(), foldChangeByPositionTotal[column].dropna().index, MCMCModel) for column in foldChangeByPositionTotal.columns.to_list()]

    # Data Logging
    dataLogging.outputListOfFitsByPos(args["output_directory"], fitListByPos, foldChangeByPositionTotal.columns.to_list())

    # Pickle the fits
    if not args["pickles"]:
        # If we aren't loading pickles, then dump the models and fits to pickles
        dataLogging.pickleModelAndFitsByPos(MCMCModel, fitListByPos, foldChangeByPositionTotal.columns.to_list(), args["output_directory"])

    # Export BayesEstaimes as a single table
    print("Exporting estimated parameters to csv file...")
    parameterDFByPos = dataLogging.logBayesEstimatesByPos(fitListByPos, foldChangeByPositionTotal, args["output_directory"])

    # Now for correlation stuff (by pos)...
    print("Loading user specified Alignment...")
    # Load the user-specified alignment
    byPosAlignment = correlationHelper.loadFastaAlignment(configYAML["pathToByPosAlignment"])
    # Split the alignment into indiv columns
    byPosAlignment = correlationHelper.splitAlignment(byPosAlignment)
    # Make dummy variables using pandas' built-in routine
    dum = pd.get_dummies(byPosAlignment)
    # Make a pandas series containing the column names and the number of unique values per column
    nunique = dum.apply(pd.Series.nunique)
    # Drop all columns with only 1 unique entry 
    dum.drop(nunique[nunique == 1].index, axis=1, inplace= True)
    # Combine parameterDF and parameterDFByPos
    combinedParams = pd.concat([parameterDF, parameterDFByPos])
    # Add EC50 to dum
    dum["EC50"] = combinedParams['Mean EC50']
    # Log Transform
    dum["log10(EC50)"] = np.log10(dum["EC50"])
    # Make list of correlations
    correlationList = correlationHelper.makeCorrelationListByPos(dum)
    # Drop the uninformative entries in the list
    correlationList.drop(["EC50", "log10(EC50)"], inplace = True)
    # Log list of correlations
    dataLogging.logCorrByPos(correlationList, args["output_directory"])

    # Log alignment (for debugging)
    dataLogging.logAlignmentByPos(byPosAlignment, dum, args["output_directory"])

    # Generate a list of B-Factors based on the correlations
    bFactors = correlationHelper.exportListOfBFactors(
        configYAML["pathToByPosAlignment"], configYAML["referenceSequence"], configYAML["sampleName"], correlationList["corr"],
        firstPositionInStructure = configYAML["firstPosInStruc"])
    # And generate a pymol script to visualize those b-factors
    dataLogging.makePyMolScriptByPos(bFactors, configYAML["referenceStructure"], configYAML["referenceSequence"], 
                                     configYAML["sampleName"], args["output_directory"])

    figurePlotting.correlationHeatmap(byPosAlignment, dum, correlationList, args["output_directory"])
    dataLogging.logCorrelationsByCapsid(byPosAlignment, dum, correlationList, args["output_directory"])
    if not args["debug"]:
        # Plot Fits
        for iterator, column in enumerate(foldChangeByPositionTotal.columns.to_list()):
            figurePlotting.plotMCMCFit(fitListByPos[iterator], foldChangeByPositionTotal[column], 
                                       foldChangeByPositionTotal.index.values, color = cm.tab20(iterator), 
                                       saveName = args["output_directory"] + "/Figures/Fitting Curves By Position/" + column)
        # Consider adding ARVIZ Plots...
