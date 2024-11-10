# DSC-0002: Artifact class
# Date: 2024-10-22
# Decision: added method to make an id for the artifact
# Status: Accepted
# Motivation: Instantiate the attributes of the artifact class. make use of base64
# Reason: To instantiate artifact and to generate an id for the artifacts. 
# Limitations:
# Alternatives:

# DSC-0002: Artifact class
# Date: 2024-10-22
# Decision: artifact type is set as private 
# Status: Accepted
# Motivation: This is not necessary for the user to see.
# Reason: Added as attribute only for the working of the code
# Limitations: - 
# Alternatives: Could have been made public, but is not necessary.

# DSC-0002: Artifact class
# Date: 2024-10-22
# Decision: artifact storage is set as private 
# Status: Accepted
# Motivation: This is not necessary for the user to see.
# Reason: Added as attribute only for the working of the code.
# Limitations: - 
# Alternatives: Could have been made public, but is not necessary.

# DSC-0002: Artifact class
# Date: 2024-10-22
# Decision: The read, save, print and get_asset_id with are public method
# Status: Accepted
# Motivation: The read, save and print methods should be public since they are something the user interacts with. 
# Reason:
# Limitations: The get_asset_id could be private since this is more a internal method.
# Alternatives: 

# DSC-0002: Dataset class
# Date: 2024-10-22
# Decision: Added the tags and metadata in the constructor
# Status: Accepted
# Motivation: As these are things the user should be able to set and have for the storage of his dataset.
# Reason:
# Limitations: 
# Alternatives: 