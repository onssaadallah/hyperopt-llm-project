# My Project

<!-- Technology & Tools Badges -->
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
[![Open In Colab](https://img.shields.io/badge/Open%20in-Colab-red?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/USERNAME/REPO/blob/main/NOTEBOOK.ipynb)
![LangChain](https://img.shields.io/badge/LangChain-Enabled-brightgreen?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Supported-lightgrey?logo=ollama&logoColor=white)

---

## Description
# hyperopt-llm-project
LLM-AutoOpt is a hybrid HPO framework that combines BO with LLM-based contextual reasoning. The framework encodes dataset meta-features, model descriptions, historical optimization outcomes, and target objectives as structured meta-knowledge within LLM prompts, using BO to initialize the search and mitigate cold-start effects. This design enables context-aware and stable hyperparameter refinement while exposing the reasoning behind optimization decisions. Experiments on a multivariate time series forecasting benchmark demonstrate that LLM-AutoOpt achieves improved predictive performance and more interpretable optimization behavior compared to BO and LLM baselines without meta-knowledge.

![Python](images/diagram.png)


