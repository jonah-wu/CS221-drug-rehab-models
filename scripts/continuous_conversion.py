'''
Convert binned features to approximate continuous ones by using the mean of each bin. This is needed for
imputation.
'''

import pandas as pd
import numpy as np

DATA_FILE = "/Users/nathanmarks/Documents/Stanford/Sophomore/SophSpring/Project/one-hotted_copy/one_hot_2017_1.csv"

print("Importing data")
data = pd.read_csv(DATA_FILE)

#remove binned columns
##################################################################################################################
# TODO: FRSTUSE3 needs to be added back in once the one-hotting has been redone without removing it.
##################################################################################################################
#reduced_data = data.drop(columns=['AGE', 'EDUC', 'DAYWAIT', 'FRSTUSE1', 'FRSTUSE2', 'FRSTUSE3', 'FREQ_ATND_SELF_HELP'])


reduced_data = data.drop(columns=['AGE', 'EDUC', 'DAYWAIT', 'FRSTUSE1', 'FRSTUSE2', 'FREQ_ATND_SELF_HELP'])


COLUMN_HEADERS = ["age_approx", "educ_approx", "daywait_approx", "firstuse1_approx", "firstuse2_approx", 
                  "firstuse3_approx", "freq_atnd_self_help_approx"]

converted_data = pd.DataFrame(np.zeros((data.shape[0], len(COLUMN_HEADERS))), columns=COLUMN_HEADERS)

ageBinToCont = {
    1:13,
    2:16,
    3:19,
    4:23,
    5:27,
    6:32,
    7:37,
    8:42,
    9:47,
    10:52,
    11:60,
    12:70
}

educBinToCont = {
    1:8,
    2:10,
    3:12,
    4:14,
    5:16,
    -9:-9
}

dayWaitBinToCont = {
    0:0,
    1:4,
    2:11,
    3:23,
    4:31,
    -9:-9
}

firstUseBinToCont = {
    1:11,
    2:13,
    3:16,
    4:19,
    5:23,
    6:27,
    7:30,
    -9:-9
}

freqSelfHelpBinToCont = {
    1:0,
    2:2,
    3:6,
    4:19,
    5:-9,
    -9:-9
}

for i in range(data.shape[0]):
    to_fill = dict.fromkeys(COLUMN_HEADERS, 0)
    row = data.iloc[i]

    #age
    temp = row["AGE"]
    to_fill["age_approx"] = ageBinToCont[temp]

    temp = row["EDUC"]
    to_fill["educ_approx"] = educBinToCont[temp]

    temp = row["DAYWAIT"]
    to_fill["daywait_approx"] = dayWaitBinToCont[temp]

    temp = row["FRSTUSE1"]
    to_fill["firstuse1_approx"] = firstUseBinToCont[temp]

    temp = row["FRSTUSE2"]
    to_fill["firstuse2_approx"] = firstUseBinToCont[temp]

    ##################################################################################################################
    # TODO: FRSTUSE3 needs to be added back in once the one-hotting has been redone without removing it.
    ##################################################################################################################
    # temp = row["FRSTUSE3"]
    # to_fill["firstuse3_approx"] = firstUseBinToCont[temp]

    temp = row["FREQ_ATND_SELF_HELP"]
    to_fill["freq_atnd_self_help_approx"] = freqSelfHelpBinToCont[temp]

    to_fill.update(reduced_data.iloc[[i]].to_dict(orient="records")[0])
    converted_data.iloc[i] = to_fill
    if i % 1000 == 0:
        print(f"Iteration: {i}")
    
f = open("continuous_test.csv", "w")
one_hot_data.to_csv(f, index=False)

