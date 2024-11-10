# DSC-0014-split
# Date: 2024-11-6
# Decision: Before training the model the user should be able to split the training and testsets. 
# Status: Accepted
# Motivation: To let the user be able to split the test and training sets in the desired configuration we implement a slider in streamlink. 
# Reason:
# Limitations: The slider only goes from 10-90%. The user can not chose to have a training set of 5% of the data and a test set of 95% of the data. To obtain reliable results we believe the test and train sets should be within 10-90%.
# Alternatives: