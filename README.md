# digiemine

Requirements:

    OS: Fedora 39
    Python: Version 3.9.6
    Python packages:
            pandas 1.5.3
            scikit-learn 1.2.1
            tsfresh 0.18.0
            swifter 1.4.0
            fitz 0.0.1dev2
These packages can be installed using pip, the version is important (especially for tsfresh).

DigiEMine also integrates code from the following repositories (integrated in folders with the same name): 
    edt_ts : https://github.com/bscheibel/edt-ts
    edt: https://github.com/bscheibel/edt
    techdraw: https://github.com/DigiEDraw/extraction

The folder 'data' includes csv files for the use cases. The manufacturing use case data was originally in yaml form, converted to csv during the first part of the algorithm ('get_infos') from the original log data which can be found in the folder 'timesequence'.
The full results - including the PDF file of the highlighted engineering drawing - can be seen in the 'result' folder.

To start the script in terminal: 

        python main.py {synthetic, valve, turm} 
    
Per default, the 'valve' use case is started. If one of the existing use cases is to be reproduced, just enter the use case name e.g. {synthetic, valve, turm}

'Valve': runs the full approaches, including all three parts (extraction of information, mining of decision rules and visualisation on the original drawing).
The use cases 'turm' and 'synthetic' only run the decision rule mining part of the approach.
