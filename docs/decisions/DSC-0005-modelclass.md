# DSC-0005: Model class
# Date: 2024-10-25
# Decision: We made a model class. This is ABC that also inherits from from artifict. In the constuctor the parameters, model and model_type are initiated. Also the attributes of the Artifact class are in inherited.  
# Status: Accepted
# Motivation: The class acts as base model for the actual models. This includes the fit and preduct methods. Also for the parameters a setter is implemented for the wrapped models for classification and regression. Lastly, the class includes a to_artifact method which can be used to turn the model to a artifact. 
# Reason: The model class should be an ABC for the actual models. 
# Limitations:
# Alternatives:

# DSC-0005: Model class
# Date: 2024-10-25
# Decision: Inheritance of artifact. 
# Status: Accepted
# Motivation: when the model class inherits from artifact. we are able to turn the model in an artifact. This could be used to store the model. 
# Reason: In the inheritance the same arguments passed from artifact with their public and private configuration. 
# Limitations:
# Alternatives:

# DSC-0005: Model class
# Date: 2024-10-25
# Decision: Inheritance of artifact. 
# Status: Accepted
# Motivation: when the model class inherits from artifact. we are able to turn the model in an artifact. This could be used to store the model. 
# Reason: In the inheritance the same arguments passed from artifact with their public and private configuration. 
# Limitations:
# Alternatives:

# DSC-0005: Model class
# Date: 2024-10-25
# Decision: getters and setters for the attributes.
# Status: Accepted
# Motivation: Because the different artibutes of them model are being called we use the a getter for this. Also for when they are set by the use we use a setter
# Reason: In the inheritance the same arguments passed from artifact with their public and private configuration. 
# Limitations:
# Alternatives: