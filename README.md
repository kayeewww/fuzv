# A Zero-Shot Federated Unlearning Framework with Stability Verification
## About The Project
Our method allows a federated client to unlearn a class for zero-shot Federated Unlearning, eliminating the influences of target data on the global model trained. 

## Presented Unlearning Methods
We propose a novel federated unlearning framework that eliminates class-specific knowledge without requiring access to original client data. The method leverages a pseudo-data generator to simulate both target and retained categories under Wasserstein-constrained geometry-aware regularization. To achieve effective and verifiable forgetting, we introduce a distillation-based unlearning process that simultaneously maximizes the attention discrepancy for the target class and minimizes distributional divergence for retained knowledge. Furthermore, a dynamic information bottleneck module adaptively adjusts the distillation temperature and attention weights, ensuring a balance between unlearning completeness and model utility preservation. Our framework supports zero-shot unlearning, requires no client-side participation, and is resilient to non-IID distributions.

Besides, this code also provides the function of membership inference attacks and backdoor attack, to evaluate whether the unlearned data has been unlearned by the model. 

The main function is contained in main.py. 
