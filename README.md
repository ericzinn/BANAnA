# BANAnA

<b>B</b>arcoded <b>A</b>AV <b>N</b>eutralizing <b>An</b>tibody <b>A</b>nalysis


## Dependencies
<li>numpy</li>
<li>pandas</li>
<li>seaborn</li>
<li>matplotlib</li>
<li>arviz</li>
<li>scipy</li>
<li>pystan</li>
<li>pyyaml</li>
<li>pickle</li>


## Usage

BANAnA ustilizes yaml files to configure the analysis for each set of data (i.e. an experiment) that the user desires. An example config file has been provided (demo/demo.yaml).

Simply point the banana.py script to the yaml file of interest (-c "configFile.yaml"), the root directory of the experimental data (-i "path/to/folder/") and provide a designated output folder (-o "path/to/output").  Optionally, ythe user can specify the positional variation flag (-p) to fit additional curves based on library positional variation and can use the "--pickles" flag to indicate that the script should load pickled models.
