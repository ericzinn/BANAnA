"""Helper functions for generating capsid/neutralization correlations of the BANAnA pipeline
Copyright (c) 2020 Eric Zinn

Contains definitions for correlating amino acid identity with neutralization.

This code is free software;  you can redstribute it and/or modify it under the terms of the AGPL
license (see LICENSE file included in the distribution)
"""

import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import argparse

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from pathlib import Path


def loadFastaAlignment(pathToFasta):

    try:
        alignment = AlignIO.read(pathToFasta, "fasta")
    except:
        print("Error: Could not load " + pathToFasta)
        quit()

    return(alignment)


def splitAlignment(alignment):
    # Convert alignment into dataframe
    alignDf = pd.DataFrame(columns=['Vector', 'Sequence'],
                           data=[[item.id, item.seq] for item in alignment]).set_index('Vector')
    # Split characters in alignment into individual (numbered) columns
    splitDf = pd.DataFrame(alignDf['Sequence'].apply(
        lambda x: ' '.join([c for c in x]))).Sequence.str.split(expand=True)

    return(splitDf)


def makeCorrelationList(dum):
    byPosCorrelationDf = pd.DataFrame()

    # drop rows for which there are no EC50s
    dum.drop(dum.loc[dum["log10(EC50)"].isnull()].index, inplace = True)

    for column in dum.columns.to_list():
         # If there is a dummy-charcter (X) in the column, skip it and go to the next column
        if column.endswith("_X"):
            continue
        # First, get the position in the alignment corresponding to the column
        position = (''.join(filter(str.isdigit, column)))
        # Next, let's get a list of all observed amino acids for this position
        stateListAtPosition = [x.split("_")[1] for x in dum.columns.to_list() if x.startswith(str(position) + '_')]

        # Propagate a list of rows to correlate...first with the default values
        rowsToCorrelate = dum.index.to_list()
        # If there is an 'x' in the list of states, then it's a variable position in one or more libraries.
        if "X" in stateListAtPosition:
            # We start by looking at all of the libraries within our dataframe
            listOfLibraries = [x for x in dum.index.to_list() if x.endswith("_Total")]
            # And then find out which libraries include varaibility at this position
            listOfLibrariesWithVariability = [b for a, b in zip(dum.loc[listOfLibraries, position + "_X"] == 1, listOfLibraries) if a]

            # Now we drop the "_Total" for each library which is variable at this position...
            # Since we can't separate states...
            for lib in listOfLibrariesWithVariability:
                rowsToCorrelate.remove(lib)

        # Calculate pearsonr (or another metric I gues...) and append it to
        # the dataframe
        correlation, pvalue = stats.pearsonr(dum.loc[rowsToCorrelate, column], 
                                             dum.loc[rowsToCorrelate, "log10(EC50)"])
        numberOfDatapoints = len(rowsToCorrelate)

        tmpDf = pd.DataFrame(dict(corr = correlation, pval = pvalue, n = numberOfDatapoints), index = [column])
        byPosCorrelationDf = byPosCorrelationDf.append(tmpDf)

    return byPosCorrelationDf


def makeCorrelationListByPos(dum):
    byPosCorrelationDf = pd.DataFrame()

    # drop rows for which there are no EC50s
    dum.drop(dum.loc[dum["log10(EC50)"].isnull()].index, inplace = True)

    for column in dum.columns.to_list():
         # If there is a dummy-charcter (X) in the column, skip it and go to the next column
        if column.endswith("_X"):
            continue
        # First, get the position in the alignment corresponding to the column
        position = (''.join(filter(str.isdigit, column)))
        # Next, let's get a list of all observed amino acids for this position
        stateListAtPosition = [x.split("_")[1] for x in dum.columns.to_list() if x.startswith(str(position) + '_')]

        # Propagate a list of rows to correlate...first with the default values
        rowsToCorrelate = ([x for x in dum.index.to_list() if (x.endswith("_net_Total")) or (x.startswith("Control"))])
        # If there is an 'x' in the list of states, then it's a variable position in one or more libraries.
        if "X" in stateListAtPosition:
            # We start by looking at all of the libraries within our dataframe
            listOfLibraries = [x for x in dum.index.to_list() if x.endswith("_net_Total")]
            # And then find out which libraries include varaibility at this position
            listOfLibrariesWithVariability = [b for a, b in zip(dum.loc[listOfLibraries, position + "_X"] == 1, listOfLibraries) if a]

            # Now we drop the "net_Total" for each library which is variable at this position...
            for lib in listOfLibrariesWithVariability:
                rowsToCorrelate.remove(lib)
            # And add the individual fits for each library...
            for lib in listOfLibrariesWithVariability:
                libPrefix = lib.split("_")[0]
                listOfPositionFits = dum.loc[:, position + "_X"].filter(like = libPrefix + "_p").index.to_list()
                rowsToAdd = [b for a, b in zip(dum.loc[listOfPositionFits, position + "_X"] == 0, listOfPositionFits) if a]

                rowsToCorrelate = rowsToAdd + rowsToCorrelate
        # Calculate pearsonr (or another metric I gues...) and append it to
        # the dataframe
        correlation, pvalue = stats.pearsonr(dum.loc[rowsToCorrelate, column], 
                                             dum.loc[rowsToCorrelate, "log10(EC50)"])
        numberOfDatapoints = len(rowsToCorrelate)

        tmpDf = pd.DataFrame(dict(corr = correlation, pval = pvalue, n = numberOfDatapoints), index = [column])
        byPosCorrelationDf = byPosCorrelationDf.append(tmpDf)

    return byPosCorrelationDf


def exportListOfBFactors(alignmentFile, sequenceOfInterest, sampleName, listOfValues, 
                         typeOfValue="coef", firstPositionInStructure=219):
    """The Following function will convert a series of computed values (e.g.
    correlation coefficients, p-values, conservation score), an alignment and 
    the name of an AAV within that alignment.  It will output a single column of
    numbers corresponding to the values at that position within the sequence"""
    alignment = AlignIO.read(alignmentFile, "fasta")

    # Import alignment into dataframe
    tmpDf = pd.DataFrame(columns=['Vector', 'Sequence'], data=[
                         [item.id, item.seq] for item in alignment]).set_index('Vector')
    # Split characters in alignment into individual (numbered) columns
    tmpDf = pd.DataFrame(tmpDf['Sequence'].apply(
        lambda x: ' '.join([c for c in x]))).Sequence.str.split(expand=True)

    vp1Numbering = []
    counter = 1

    # Convert the alignment numbering to the VP1 numbering for a given sequence
    # and add it as a new row to the temporary dataframe
    for columnName, columnData in tmpDf.loc[sequenceOfInterest].iteritems():
        if columnData != "-":
            vp1Numbering.append(counter)
            counter = counter + 1
        else:
            vp1Numbering.append("-")

    tmpDf.loc["mapPositions"] = vp1Numbering

    # Find the values corresponding to the sequence of interest and append them
    # to the dataframe as a new row
    allBFactors = []
    for columnName, columnData in tmpDf.loc[sequenceOfInterest].iteritems():
        try:
            tmpBFactor = listOfValues.loc[str(columnName) + "_" + columnData]
        except:
            # If you have a list of correlation coef.s, add '0' if you can't
            # find a corresponding value in the passed list (no correlation)
            if typeOfValue == "coef":
                tmpBFactor = 0
            # If it's a p value, then add 1 (no significance)
            elif typeOfValue == "p value":
                tmpBFactor = 1
            # If it's some other type of value, then add NaN (not a number)
            else:
                tmpBFactor = np.nan
        allBFactors.append(tmpBFactor)

    tmpDf.loc["allBFactors"] = allBFactors

    # Now all we need is to make a quick list of the B Factors corresponding
    mappedBFactors = []

    for columnname, columndata in tmpDf.loc["mapPositions"].iteritems():
        if (columndata != "-") and columndata >= firstPositionInStructure:
            mappedBFactors.append(tmpDf.loc['allBFactors', columnname])

    # with open('bFactors_'+sequenceOfInterest+'_'+sampleName+'.txt', 'w') as filehandle:
    #    filehandle.writelines("%s\n" % value for value in mappedBFactors)\

    # print('Success, wrote to file: bFactors_'+sequenceOfInterest+'_'+sampleName+'.txt')

    return(mappedBFactors)
