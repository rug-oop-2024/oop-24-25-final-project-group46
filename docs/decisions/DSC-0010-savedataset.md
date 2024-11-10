# DSC-0010: Save dataset
# Date: 2021-10-22
# Decision: The user is them prompted to give the dataset a name and a tag. The metadata id is genarated by a private _generate_id method which gives a random id to the metadata. The dataset is stored as an artifact object in the assets file.
# Status: Accepted
# Motivation: The users need to give the correct data to the file and with the generated id the dataset is stored correctly as an artifact. 
# Reason:
# Limitations:
# Alternatives: