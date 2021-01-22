"""Data Logging Module of the BANAnA pipeline
Copyright (c) 2020 Eric Zinn

Contains definitions for exporting data at various points in the BANAnA process to files (mostly
.csv files) within a user defined folder.  These files are intended to be referenced for the
sake of troubleshooting as well as manual analysis if the user wishes.

This code is free software;  you can redstribute it and/or modify it under the terms of the AGPL
license (see LICENSE file included in the distribution)
"""

from pathlib import Path
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pickle


def makeDataSubfolders(outputBasePath):
    """Given a specific directory, ensure that all of the required
    subfolders for a given analysis are present"""

    print(outputBasePath)

    Path(outputBasePath + "/intermediateData/AggregatedRawCounts").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/intermediateData/averageFoldChange").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/intermediateData/FilteredRawCounts").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/intermediateData/normalizedCounts").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/intermediateData/curveFitting/pickles").mkdir(parents = True, exist_ok = True)
    Path(outputBasePath + "/intermediateData/curveFittingByLibraryPosition/pickles").mkdir(parents = True, 
                                                                                           exist_ok = True)
    Path(outputBasePath + "/intermediateData/correlations/").mkdir(parents = True, exist_ok = True)
    Path(outputBasePath + "/intermediateData/correlationsByPos/").mkdir(parents = True, exist_ok = True)


def outputListOfFits(outputBasePath, fitList, listOfNames):
    for iterator, fit in enumerate(fitList):
        with open(outputBasePath + "/intermediateData/curveFitting/" + listOfNames[iterator] + ".txt", 'w') as f:
            print("Writing summary of fit for " + listOfNames[iterator] + "...")
            print(fit, file = f)


def logBigDf(outputBasePath, bigDf):
    bigDf.to_csv(outputBasePath + "/intermediateData/AggregatedRawCounts/bigDataFrame.csv")


def logTotalDf(outputBasePath, totalDf):
    totalDf.to_csv(outputBasePath + "/intermediateData/AggregatedRawCounts/totalDataFrame.csv")


def logFilteredDf(outputBasePath, filteredDf):
    filteredDf.to_csv(outputBasePath + "/intermediateData/FilteredRawCounts/filteredDataframe.csv")


def logNormalizedDf(outputBasePath, normalizedDf):
    normalizedDf.to_csv(outputBasePath + "/intermediateData/normalizedCounts/cpSpike.csv")


def logLog2NormalizedDf(outputBasePath, log2normalizedDf):
    log2normalizedDf.to_csv(outputBasePath + "/intermediateData/normalizedCounts/log2cpSpike.csv")


def logSummaryNormDf(outputBasePath, summaryDf):
    summaryDf.to_csv(outputBasePath + "/intermediateData/normalizedCounts/summaryOfNormalizedCounts.csv")


def logLog2FCList(outputBasePath, log2FCList):
    for frame in log2FCList:
        frame.to_csv(outputBasePath + "/intermediateData/averageFoldChange/log2FC_" + frame.index.to_list()[0].split('_')[0] + ".csv")


def logLog2AvgLibsDf(outputBasePath, log2AvgLibsDf):
    log2AvgLibsDf.to_csv(outputBasePath + "/intermediateData/averageFoldChange/log2avgLibsDataFrame.csv") 


def logTotalLibsDf(outputBasePath, totalLibsDf):
    totalLibsDf.to_csv(outputBasePath + "/intermediateData/averageFoldChange/log2totalLibsDataFrame.csv")


def logLog2FCControls(outputBasePath, log2FCControlsDf):
    log2FCControlsDf.to_csv(outputBasePath + "/intermediateData/averageFoldChange/log2foldChangeControls.csv")


def logNetFCDf(outputBasePath, netFCDf):
    netFCDf.to_csv(outputBasePath + "/intermediateData/averageFoldChange/netFCDataFrame.csv")


def logTotalFCDf(outputBasePath, totalFCDf):
    totalFCDf.to_csv(outputBasePath + "/intermediateData/averageFoldChange/totalFCDataFrame.csv")


def doseCurve(x, EC50, Hillslope):
    return (x**Hillslope) / (x**Hillslope + EC50**Hillslope)


def logBayesEstimates(fitList, FCdf, outputBasePath):
    columnsList = ["5% EC50", "95% EC50", "Mean EC50", "5% Hillslope", "95% Hillslope", "Mean Hillslope", "R^2"]
    seriesList = []

    for iterator, fit in enumerate(fitList):
        df = fit.to_dataframe()
        Vector = FCdf.columns.to_list()[iterator]
        EC50Percentiles, HillslopePercentiles = 10**np.percentile(df['logEC50'], [5, 95]), np.percentile(df['Hillslope'], [5, 95])
        EC50Mean, HillslopeMean = 10**df['logEC50'].mean(), df['Hillslope'].mean()
        y_true = FCdf[Vector].values
        y_pred = doseCurve(FCdf.index.values, EC50Mean, HillslopeMean)
        r2 = r2_score(y_true, y_pred)

        data = [EC50Percentiles[0], EC50Percentiles[1], EC50Mean, HillslopePercentiles[0], HillslopePercentiles[1], HillslopeMean, r2]

        seriesList.append(pd.Series(data, name = Vector, index=columnsList))

    parameterDF = pd.concat(seriesList, axis = 1).T
    parameterDF.to_csv(outputBasePath + "/intermediateData/curveFitting/bayesianParameterEstimation.csv")

    return parameterDF


def logLogFCByPosTotal(outputBasePath, log2FCByPosTotal):
    log2FCByPosTotal.to_csv(outputBasePath + "/intermediateData/averageFoldChange/log2FoldChangeLibrariesByPositionTotal.csv")


def logFCByPosTotal(outputBasePath, FCByPostTotal):
    FCByPostTotal.to_csv(outputBasePath + "/intermediateData/averageFoldChange/foldChangeLibrariesByPositionTotal.csv")


def outputListOfFitsByPos(outputBasePath, fitList, listOfNames):
    for iterator, fit in enumerate(fitList):
        with open(outputBasePath + "/intermediateData/curveFittingByLibraryPosition/" + listOfNames[iterator] + ".txt", 'w') as f:
            print("Writing summary of fit for " + listOfNames[iterator] + "...")
            print(fit, file = f)


def logBayesEstimatesByPos(fitList, FCdf, outputBasePath):
    columnsList = ["5% EC50", "95% EC50", "Mean EC50", "5% Hillslope", "95% Hillslope", "Mean Hillslope", "R^2"]
    seriesList = []

    for iterator, fit in enumerate(fitList):
        df = fit.to_dataframe()
        Vector = FCdf.columns.to_list()[iterator]
        EC50Percentiles, HillslopePercentiles = 10**np.percentile(df['logEC50'], [5, 95]), np.percentile(df['Hillslope'], [5, 95])
        EC50Mean, HillslopeMean = 10**df['logEC50'].mean(), df['Hillslope'].mean()
        y_true = FCdf[Vector].dropna()
        y_pred = doseCurve(FCdf[Vector].dropna().index.values, EC50Mean, HillslopeMean)

        try:
            r2 = r2_score(y_true, y_pred)
        except:
            r2 = np.nan

        data = [EC50Percentiles[0], EC50Percentiles[1], EC50Mean, HillslopePercentiles[0], HillslopePercentiles[1], HillslopeMean, r2]

        seriesList.append(pd.Series(data, name = Vector, index=columnsList))

    parameterDF = pd.concat(seriesList, axis = 1).T
    parameterDF.to_csv(outputBasePath + "/intermediateData/curveFittingByLibraryPosition/bayesianParameterEstimation.csv")

    return parameterDF


def pickleModelAndFits(model, fitList, listOfNames, outputBasePath):
    for iterator, fit in enumerate(fitList):
        with open(outputBasePath + "/intermediateData/curveFitting/pickles/" + listOfNames[iterator] + ".pickle", 'wb') as f:
            print("Pickling the model and fit for " + listOfNames[iterator] + "...")
            pickle.dump({"model": model, "fit": fit}, f)


def pickleModelAndFitsByPos(model, fitList, listOfNames, outputBasePath):
    for iterator, fit in enumerate(fitList):
        with open(outputBasePath + "/intermediateData/curveFittingByLibraryPosition/pickles/" + listOfNames[iterator] + ".pickle", 'wb') as f:
            print("Pickling the model and fit for " + listOfNames[iterator] + "...")
            pickle.dump({"model": model, "fit": fit}, f)


def logCorrByPos(corrList, outputBasePath):
    corrList.to_csv(outputBasePath + "/intermediateData/correlationsByPos/correlationListByPos.csv")


def logCorr(corrList, outputBasePath):
    corrList.to_csv(outputBasePath + "/intermediateData/correlations/correlationList.csv")


def logAlignmentByPos(alignment, dum, outputBasePath):
    alignment.to_csv(outputBasePath + "/intermediateData/correlationsByPos/alignmentByPos.csv")
    dum.to_csv(outputBasePath + "/intermediateData/correlationsByPos/dumByPos.csv")


def makePyMolScript(bFactors, referenceStructure, referenceSequence, sampleName, outputBasePath):
    # Start generating a pymol script...
    outputScript = open(outputBasePath + "/intermediateData/correlations/" + sampleName + ".pml", 'w')
    outputScript.write("reinitialize\n")
    outputScript.write("fetch " + referenceStructure + ", type=pdb1, async=0\n")
    outputScript.write("newB = " + str(list(np.nan_to_num(bFactors))) + "\n")
    outputScript.write("alter " + referenceStructure + ", b = 0.0\n")
    outputScript.write("alter " + referenceStructure + " and n. CA, b = newB.pop(0)\n")
    outputScript.write(
        "cmd.spectrum('b', 'red_white_blue', '" + referenceStructure + " and n. CA', minimum=-1, maximum=1)\n")
    outputScript.write("create ca_obj, " + referenceStructure + " and name ca\n")
    outputScript.write("ramp_new ramp_obj, ca_obj, [0, 10], [-1, -1, 0]\n")
    outputScript.write("set surface_color, ramp_obj, " + referenceStructure + "\n")
    outputScript.write("disable ramp_obj\n")
    outputScript.write("disable ca_obj\n")
    outputScript.write("set all_states, on\n")
    outputScript.write("set_view (0.715355754, 0.696245492, -0.059274726, -0.696687281, 0.717189074, 0.016255794, 0.053828631, 0.029667672, 0.998105764, 0.000000000, 0.000000000, -951.704284668, 0.000000000, 0.000000000, 0.000000000, -581.101867676, 2484.510498047, -20.000000000)\n")
    outputScript.write("show surface, " + referenceStructure)

    outputScript.close()


def makePyMolScriptByPos(bFactors, referenceStructure, referenceSequence, sampleName, outputBasePath):
    # Start generating a pymol script...
    outputScript = open(outputBasePath + "/intermediateData/correlationsByPos/" + sampleName + ".pml", 'w')
    outputScript.write("reinitialize\n")
    outputScript.write("fetch " + referenceStructure + ", type=pdb1, async=0\n")
    outputScript.write("newB = " + str(list(np.nan_to_num(bFactors))) + "\n")
    outputScript.write("alter " + referenceStructure + ", b = 0.0\n")
    outputScript.write("alter " + referenceStructure + " and n. CA, b = newB.pop(0)\n")
    outputScript.write(
        "cmd.spectrum('b', 'red_white_blue', '" + referenceStructure + " and n. CA', minimum=-1, maximum=1)\n")
    outputScript.write("create ca_obj, " + referenceStructure + " and name ca\n")
    outputScript.write("ramp_new ramp_obj, ca_obj, [0, 10], [-1, -1, 0]\n")
    outputScript.write("set surface_color, ramp_obj, " + referenceStructure + "\n")
    outputScript.write("disable ramp_obj\n")
    outputScript.write("disable ca_obj\n")
    outputScript.write("set all_states, on\n")
    outputScript.write("set_view (0.715355754, 0.696245492, -0.059274726, -0.696687281, 0.717189074, 0.016255794, 0.053828631, 0.029667672, 0.998105764, 0.000000000, 0.000000000, -951.704284668, 0.000000000, 0.000000000, 0.000000000, -581.101867676, 2484.510498047, -20.000000000)\n")
    outputScript.write("show surface, " + referenceStructure)

    outputScript.close()


def logCorrelationsByCapsid(alignmentDf, dum, correlationList, outputBasePath):
    rowsToPlot = [x for x in dum.index.values if (x.startswith("Control")) or (x.endswith("net_Total"))]
    alignmentDf = alignmentDf.loc[rowsToPlot]
    correlationDict = correlationList.to_dict()

    for index, row in alignmentDf.iterrows():
            # Empty list to hold the mapped values of a single capsid
        correlationRow = []
        aminoAcidRow = alignmentDf.loc[index].to_list()
        # Clean up the names (optional)

        if "Control" in index:
            name = index.split("_")[1]
        else:
            name = index.split("_")[0]

        for columnNumber, aminoAcid in enumerate(row):
            lookUpItem = str(columnNumber) + "_" + aminoAcid
            try: 
                correlationRow.append(correlationDict['corr'][lookUpItem])
            except:
                correlationRow.append(0)

        tmpDf = pd.DataFrame(dict(aa = aminoAcidRow, corr = correlationRow))
        tmpDf = tmpDf[tmpDf.aa != '-']
        tmpDf.reset_index(inplace = True, drop = True)

        tmpDf.to_csv(outputBasePath + "/intermediateData/correlationsByPos/" + name + "correlations.csv")
