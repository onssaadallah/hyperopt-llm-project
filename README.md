![Python Budget](https://img.shields.io)
# hyperopt-llm-project
LLM-AutoOpt is a hybrid HPO framework that combines BO with LLM-based contextual reasoning. The framework encodes dataset meta-features, model descriptions, historical optimization outcomes, and target objectives as structured meta-knowledge within LLM prompts, using BO to initialize the search and mitigate cold-start effects. This design enables context-aware and stable hyperparameter refinement while exposing the reasoning behind optimization decisions. Experiments on a multivariate time series forecasting benchmark demonstrate that LLM-AutoOpt achieves improved predictive performance and more interpretable optimization behavior compared to BO and LLM baselines without meta-knowledge.

