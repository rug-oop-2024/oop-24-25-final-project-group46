# DSC-0005: Model class
# Date: 2024-10-25
# Decision: We made a model class. This is ABC that also inherits from from artifict. In the constuctor the parameters, model and model_type are initiated. Also the attributes of the Artifact class are in inherited.  
# Status: Accepted
# Motivation: The class acts as base model for the actual models. This includes the fit and preduct methods. Also for the parameters a setter is implemented for the wrapped models for classification and regression. Lastly, the class includes a to_artifact method which can be used to turn the model to a artifact. 
# Reason: The model class should be an ABC for the actual models. 
# Limitations:
# Alternatives: