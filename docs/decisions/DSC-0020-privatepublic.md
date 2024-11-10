# DSC-0020; Private public.
# Date: 2024-11-10
# Decision: To ensure a clean interface, we use public methods for user interaction and private methods for internal processing.
# Status: Accepted
# Motivation: Using public methods for user interaction provides a accesable streamlit application and the that key functions are clearly exposed. Private methods encapsulate internal logic, such as data checks, preprocessing, and validations in the codebase. This can be seen in the autoop files which is the internal structure and the app files which are the streamlit app. 
# Reason: Public methods expose core functionalities, while private methods safeguard internal operations.
# Limitations: 
# Alternatives: 