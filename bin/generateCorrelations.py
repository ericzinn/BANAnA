import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import argparse

from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from pathlib import Path


# ***Begin Function Definitions***
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

    #print('Success, wrote to file: bFactors_'+sequenceOfInterest+'_'+sampleName+'.txt')

    return(mappedBFactors)
# ***End Function Definitions***

# ***Begin agument parser***


parser = argparse.ArgumentParser()

# parser.add_argument("-n", "--sample_name", type = int,
#                   help = "Name of the Sample (Serum of Ab)", default = "sample")
parser.add_argument("-o", "--output_file", type=str,
                    help="path of the output file (default: correlations.csv)", 
                    default="correlations.csv")
parser.add_argument("-p", "--pymol_script", type=str,
                    help="optional: path of pymol script to output (include .pml please)", 
                    default="")
parser.add_argument("-r", "--reference_sequence", type=str,
                    help="Name of vector (in alignment) to reference when making pymol script", 
                    default="Control_AAV2")

requiredNamed = parser.add_argument_group("Required Named Arguments")
requiredNamed.add_argument("-i", "--input_file", type=str,
                           help="Path to the parameter estimation csv file")
requiredNamed.add_argument("-a", "--alignment_file", type=str,
                           help="Path to the alignment file")

args = vars(parser.parse_args())

# Check Required Arugments
if not args['input_file']:
    print('Please provide a valid .csv file (-i)')
    quit()

if not args['alignment_file']:
    print('Please provide a valid .fasta file (-a)')
    quit()

# ***End agument parser***

# ***Begin loading external data***
# Load the alignment file
try:
    alignment = AlignIO.read(args['alignment_file'], "fasta")
except:
    print("Error: Could not load " + args['alignment_file'])
    quit()

# Load the csv file
try:
    parametersDf = pd.read_csv(args['input_file'], index_col=0)
except:
    print("Error: Could not load " + args['input_file'])
    quit()
# ***End loading external data***

# ***Begin massaging data***
# Convert alignment into dataframe
alignDf = pd.DataFrame(columns=['Vector', 'Sequence'],
                       data=[[item.id, item.seq] for item in alignment]).set_index('Vector')
# Split characters in alignment into individual (numbered) columns
alignDf = pd.DataFrame(alignDf['Sequence'].apply(
    lambda x: ' '.join([c for c in x]))).Sequence.str.split(expand=True)
# One-hot encode (Dummy variables)
alignDum = pd.get_dummies(alignDf)

# Drop bad fits from alignment (R^2 < 0.5 for now...)
badFitsList = []

for index in alignDum.index.to_list():
    if parametersDf.loc[index]["R^2"] < 0.5:
        badFitsList.append(index)

alignDum = alignDum.drop(badFitsList, axis=0)

# Make a pandas series containing the column names and the number of unique values per column
nunique = alignDum.apply(pd.Series.nunique)
# Drop all columns with only 1 unique entry (no variation)
alignDum.drop(nunique[nunique == 1].index, axis=1, inplace=True)
# Add the EC50 column from the parameters csv
alignDum["EC50"] = parametersDf['Mean EC50']


# ***End massaging data***


# ***Begin data manip ***

# Log transform the EC50 values b/c EC50s are log-normal
alignDum["Log10(EC50)"] = np.log10(alignDum["EC50"])
# WARN if log-transformed data are not normal (shapiro test)
shapiro_pval = stats.shapiro(alignDum["Log10(EC50)"])[1]

if shapiro_pval < 0.05:
    print("Warning: Log transformed EC50s may not be normally distributed: p = " + str(shapiro_pval))
else:
    print("OK! EC50s are likely normally distributed: p = " + str(shapiro_pval))

# Perform point-biserial correlation on matrix
pearsonmatrix = alignDum.corr(method="pearson")
# Generate p-values from point-biserial correaltion
pvalue_list = []

for index, item in pearsonmatrix['Log10(EC50)'].items():
    pvalue_list.append(stats.pearsonr(
        alignDum['Log10(EC50)'], alignDum[index])[1])

pearsonmatrix['p'] = pvalue_list
pearsonmatrix = pearsonmatrix.drop(['EC50', 'Log10(EC50)'])

# Generate a new dataframe for logging/ouput
outputTable = pd.DataFrame()

outputTable['pearson'] = pearsonmatrix['Log10(EC50)']
outputTable['p'] = pearsonmatrix['p']
# ***End data manip***

# ***Begin data ouptut***
outputTable.to_csv(args['output_file'])

# Check if we're outputting a pymol script...
if args["pymol_script"]:
    if ".pml" not in args["pymol_script"]:
        print("Error: flag -p must end in .pml")
        quit()

    sample_name = args["pymol_script"].split(".")[0]

    bFactors = np.nan_to_num(exportListOfBFactors(
        args["alignment_file"], args["reference_sequence"], sample_name, pearsonmatrix["Log10(EC50)"]))

    # Start generating a pymol script...

    outputScript = open(args["pymol_script"], 'w')
    outputScript.write("reinitialize\n")
    outputScript.write("fetch 1lp3, type=pdb1, async=0\n")
    outputScript.write("newB = " + str(bFactors) + "\n")
    outputScript.write("alter 1lp3, b = 0.0\n")
    outputScript.write("alter 1lp3 and n. CA, b = newB.pop(0)\n")
    outputScript.write(
        "cmd.spectrum('b', 'red_white_blue', '1lp3 and n. CA', minimum=-1, maximum=1)\n")
    outputScript.write("create ca_obj, 1lp3 and name ca\n")
    outputScript.write("ramp_new ramp_obj, ca_obj, [0, 10], [-1, -1, 0]\n")
    outputScript.write("set surface_color, ramp_obj, 1lp3\n")
    outputScript.write("disable ramp_obj\n")
    outputScript.write("disable ca_obj\n")
    outputScript.write("set all_states, on\n")
    outputScript.write("set_view (0.715355754, 0.696245492, -0.059274726, -0.696687281, 0.717189074, 0.016255794, 0.053828631, 0.029667672, 0.998105764, 0.000000000, 0.000000000, -951.704284668, 0.000000000, 0.000000000, 0.000000000, -581.101867676, 2484.510498047, -20.000000000)\n")
    outputScript.write("show surface, 1lp3")

    outputScript.close()
