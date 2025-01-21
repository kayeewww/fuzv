# FUZV (Verified Zero-shot Federated Unlearning)
## About The Project
FUZV allows a federated client to unlearn a class from the Federated Learning system and eliminate the influences of target data on the global model trained by the standard Federated Learning. 

## Presented Unlearning Methods
The parameters of the client model saved by users during the standard FL process are utilized as the step size for server-side pseudodata unlearning. Using this step size, the pseudodata generator refines the target category forgetting through KL divergence amplification, while retaining non-target categories using W2 distance minimization. The refined generator updates the server model, which becomes the initialization for the next round of training. A dynamic adjustment mechanism adjusts rhot to optimize KL temperature and knowledge distillation, balancing forgetting and retention. The updated model is validated locally by comparing W2 distances and attack metrics without client-side pseudodata generation.

Besides, this code also provides the function of membership inference attacks and backdoor attack, to evaluate whether the unlearned data has been unlearned by the model. 

The main function is contained in main.py. 
