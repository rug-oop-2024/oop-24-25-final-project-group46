# DSC-0004: Metric class
# Date: 2021-10-24
# Decision: Make a metric class with 6 different metrics as extentions for the models. This includes ABC Metric which is the general form for the metrics. The 6 metrics are all inherited from this ABC. This includes the mean squared error, mean absolute error, root mean squared error for the evaluation of the regression models. For the classification models we added the metrics accuracy, precision and recall. 
# Status: Accepted
# Motivation: The metrics are needed to evaluate the model in the pipeline. 
# Reason: The metric class has a abstract metric class and six metric classen that inherit from the ABC.
# Limitations: The name of the metric name is not printed correctly in streamlit. This could be due to the implementation of the __str__ magic method. 
# Alternatives:

# DSC-0004: Metric class
# Date: 2021-10-24
# Decision: All the mettric classes inherit from the metric class.
# Status: Accepted
# Motivation:
# Reason: This way the structure is similar and we can use inheritance. They also take the same arguments ground truth and predictions this way
# Limitations: 
# Alternatives:

# DSC-0004: Metric class
# Date: 2021-10-24
# Decision: Numpy package is used for all the math inside the metric functions.
# Status: Accepted
# Motivation:
# Reason: To make make the logic inside the specific metric classes work we made use of the  numy package and use it in our operations. 
# Limitations: 
# Alternatives: