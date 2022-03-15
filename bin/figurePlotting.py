"""Figure Plotting Module of the BANAnA pipeline
Copyright (c) 2020 Eric Zinn
 
Contains definitions relating to figure plotting and saving.

This code is free software;  you can redstribute it and/or modify it under the terms of the AGPL
license (see LICENSE file included in the distribution)
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import cm
from matplotlib.colors import ListedColormap
import arviz as az
import seaborn as sns
import pandas as pd


def setFigureStyle():
    sns.set_style('whitegrid')
    sns.set_context("paper", font_scale=1.8)


def makeFigureSubfolders(outputBasePath):
    """Given a specific directory, ensure that all of the required
       subfolders for a given analysis are present"""

    Path(outputBasePath + "/Figures/Fitting Curves").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/Figures/Fitting Curves By Position").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/Figures/correlations/").mkdir(parents=True, exist_ok=True)
    Path(outputBasePath + "/Figures/MCMC Info").mkdir(parents=True, exist_ok=True)


def plotStackedBargraph(df, outputBase, w = 20, h = 12, d = 100, order = "reverse",
                        cmap = 'tab20', colorlist = ""):
    """Plot a stacked bargraph from a dataframe containing the sum total of (normalized) barcodes
       from each of the included libraries and individually barcoded controls in the experiment"""
    setFigureStyle()

    # If specified, reverse the dataframe
    if order == "reverse":
        df = df.iloc[::-1]
    # Set up plot, render it 
    fig, axes = plt.subplots(figsize=(w, h), dpi=d)
    if colorlist:
        df.plot(kind = 'bar', stacked = True, color = colorlist, ax = axes, legend = True)
    else:
        df.plot(kind = 'bar', stacked = True, colormap=cmap, ax=axes, legend = True)

    plt.legend(loc = "upper left", bbox_to_anchor=(1.05, 1))
    plt.savefig(outputBase + "/Figures/summaryStackedBargraph.png", bbox_inches = "tight")


def doseCurve(x, EC50, Hillslope):
    return (x**Hillslope) / (x**Hillslope + EC50**Hillslope)


def plotMCMCFit(fit, obsNeutralization, obsConcentration, color = 'C0', saveName = "", 
                h = 10, w = 10, d = 100, xmin = 5e-2, xmax= 1e-4):
    setFigureStyle()
    """Plot a SINGLE MCMC fit"""
    fig, ax = plt.subplots(figsize = (w, h), dpi = d)
    ax.set_xscale('log')
    ax.set_xlim(xmin, xmax)

    x = np.logspace(-6, -1, num = 100)
    logEC50, Hillslope = fit.extract(('logEC50', 'Hillslope')).values()
    curveList = []

    # Compute curves for 10,000 random smaplings of EC50 & Hillslope values from fit
    for i in np.random.randint(0, len(logEC50) - 1, 10000):
        curveList.append(doseCurve(x, 10**logEC50[i], Hillslope[i]))

    # Transpose curveList and compute 5th, 50th, and 95th Percentiles 
    a = list(map(list, zip(*curveList)))
    lowerCurve, medianCurve, upperCurve = np.percentile(a, [5, 50, 95], axis = 1)

    # Plot the curves
    lowerLine, = ax.plot(x, lowerCurve, '--', c = color)
    upperLine, = ax.plot(x, upperCurve, '--', c = color)
    medianLine, = ax.plot(x, medianCurve, c = color,)
    medianLine.set_label("Curve Fit: Median Value (50th Percentile)")
    lowerLine.set_label("Curve Fit: 5th, 95th Percentile")

    dots, = ax.plot(obsConcentration, obsNeutralization, '.--', markersize = 20, color = color)
    dots.set_label("Observed Values: " + obsNeutralization.name)

    ax.legend()

    if saveName:
        plt.savefig(saveName + ".png")

    plt.close(fig)


def plotLibraryNeutralization(neutDf, outputBasePath, w = 10, h = 10, d = 100, colormap = cm.tab20, colorDict = "", cleanNames = False):
    setFigureStyle()

    fig, ax = plt.subplots(figsize = (w, h), dpi = d)

    x = np.arange(1e-4, 1e-1, 5e-4)
    ax.set_xscale('log')
    ax.set_xlim(5e-2, 1e-4)

    for iterator, column in enumerate(neutDf.columns.to_list()):
        # Option to clean up names...at risk of making things
        # less clear
        if cleanNames == True:
            if "Control" in column:
                name = column.split("_")[1]
            else:
                name = column.split("_")[0]
        if colorDict:
            color = colorDict[name]
        else:
            color = colormap(iterator)
        x_t = neutDf.index.values
        y_t = neutDf[column]

        dots, = ax.plot(x_t, y_t, '.--', markersize = 20, color = color)

        if cleanNames == True:
            dots.set_label(name)
        else:
            dots.set_label(column + " Observed")

    ax.legend(loc = "upper left", bbox_to_anchor=(1.05, 1))
    plt.savefig(outputBasePath + "/Figures/observedBarcodedNeutralizationCurve.png", 
                bbox_inches = "tight")


def plotMCMCTraces(fitList, df, outputBasePath):

    # Suppress warnigns from arviz
    warnings.filterwarnings('ignore')

    for iterator, column in enumerate(df.columns.to_list()):
        fig = az.plot_trace(fitList[iterator])
        plt.savefig(outputBasePath + "/Figures/MCMC Info/" + column + ".png")
        plt.close()

    # Turn warnings back on
    warnings.filterwarnings('default')


def plotMCMCTraces(fitList, df, outputBasePath):

    # Suppress warnigns from arviz
    warnings.filterwarnings('ignore')

    for iterator, column in enumerate(df.columns.to_list()):
        fig = az.plot_trace(fitList[iterator])
        plt.savefig(outputBasePath + "/Figures/MCMC Info/" + column + ".png")
        plt.close()

    # Turn warnings back on
    warnings.filterwarnings('default')


def makeBoxPlot(fitList, listOfNames, columnToPlot, colorList, outputBasePath, cleanNames = True):
    setFigureStyle()

    # Make a wide-form to hold all of the data
    df = pd.DataFrame()

    for iterator, vector in enumerate(listOfNames):
        if cleanNames == True:
            if "Control" in vector:
                name = vector.split("_")[1]
            else:
                name = vector.split("_")[0]

        df[name] = fitList[iterator][columnToPlot]

    # Initialize the palette
    pal = sns.color_palette(colorList)

    plt.figure(figsize=(10, 20), dpi = 300)
    sns.boxplot(data = df, orient = "h", showfliers = False, palette = pal, saturation =1)
    sns.despine(left = True)

    plt.savefig(outputBasePath + "/Figures/" + columnToPlot + "BoxPlot.png")


def correlationHeatmap(alignmentDf, dum, correlationList, outputBasePath, cleanNames = True, trimToConsensus = True):
    """Make a pretty heatmap of the correlations corresponding to amino acids
       within the provided alignment"""
    if trimToConsensus:
        rowsToPlot = [x for x in dum.index.values if (x.startswith("Control")) or (x.endswith("net_Total"))]
        alignmentDf = alignmentDf.loc[rowsToPlot]
    # Initialize a new DF to hold the heatmap values
    heatMapDf = pd.DataFrame()
    # Convert our correlationList into a dictionary
    correlationDict = correlationList.to_dict()
    # Now we can map the values (column + amino acid state) for each of the entries
    # in our alignment by our dictionary of correlations
    for index, row in alignmentDf.iterrows():
        # Empty list to hold the mapped values of a single capsid
        heatMapRow = []

        # Clean up the names (optional)
        if cleanNames == True:
            if "Control" in index:
                name = index.split("_")[1]
            else:
                name = index.split("_")[0]

        for columnNumber, aminoAcid in enumerate(row):
            lookUpItem = str(columnNumber) + "_" + aminoAcid

            try: 
                heatMapRow.append(correlationDict['corr'][lookUpItem])
            except:
                heatMapRow.append(0)

        tmpDf = pd.DataFrame([heatMapRow], index = [name])
        heatMapDf = heatMapDf.append(tmpDf)

    # Then we can finally plot the resultant dataframe as a heatmap
    f, ax = plt.subplots(figsize = (30, 0.5 * len(heatMapDf.index)))
    ax = sns.heatmap(heatMapDf, vmin = -1, vmax = 1, cmap = 'bwr_r')

    plt.savefig(outputBasePath + "/Figures/correlations/correlationHeatmap.png")
