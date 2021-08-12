"""
This script is used to get the training data for learning a retriver that can get back the most forgettable examples given a batch of error cases to fix.

Input:
    - The training streams. ---> get the error cases.
    - model.

Output:
    - The pairs between error cases and associated forgettable examples.


Key logic:
    
    - Use the simple_CL method and put it work on the training streams (can be randomly sampled.)
    - For each episode, before and after the error-fixing (continual fine-tuning) step, we record the forgetted the examples.
    

"""


