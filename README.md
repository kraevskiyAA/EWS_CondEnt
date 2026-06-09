# Early Warning System for Concept Drift Detection

**Authors:** Artem Kraevskiy, Evgeniy Sokolovskiy, Artem Prokhorov

## Overview

This repository contains code and examples for an Early Warning System (EWS) designed for online concept drift detection in multivariate financial time series.

**Abstract:** Financial markets of emerging economies are characterized by unusually high volatility, non-linear dependence, and strong correlation in extreme values, making standard tools for early detection of financial crises inapplicable. In this paper we develop and apply a new early warning system (EWS) for online concept drift identification that permits unrestricted dependence patterns and heavy tails. The core of the EWS is an effective detection of jumps in the conditional entropy of financial indicators, rooted in change-point detection theory. We focus on finding significant information shifts between interdependent time series under the curse of dimensionality.

- We develop an approach that analyzes the stability of linear relations between target and explanatory variables.
- We propose extensions that capture non-linear interactions and impose no restrictions on the variables' distribution — particularly relevant for emerging financial markets.
- We show the consistency of EWS results on synthetic data and apply the approach to financial data from an emerging markets (Russia and South Africa).

## Repository Structure

```
EWS_condent/
├── ChangePoint/
│   └── ShiryaevRoberts_CPD.py     # Shiryaev–Roberts change-point detection statistic
├── CondEnt/
│   ├── CondEnt_RFCDE.py           # Conditional entropy estimation (linear & LLF models)
│   └── CondEnt_RFCDE_correct_ranks.py  # Variant with corrected rank transformations
├── Data/
│   ├── df_Rus_Fin_Markets.csv     # Russian financial markets data
│   └── df_cond_ent.csv            # Uzbekistan market data
├── Examples/
│   ├── Synthetic_data.ipynb       # Experiments from the paper on synthetic data
│   └── New_ranks_experimets.ipynb # Rank correction experiments
├── utils.py                       # Plotting utilities
└── __init__.py
```

## Installation

Install the package in editable mode along with required dependencies:

```bash
pip install -e .
```

**Key dependencies:** `numpy`, `pandas`, `matplotlib`, `seaborn`, `statsmodels`, `scikit-learn`, `rfcde`, `scipy`

## Examples

Reproduce the experiments from the paper using the notebooks in the `Examples/` directory:

- **`Examples/Synthetic_data.ipynb`** — demonstrates both information-transfer termination and inversion scenarios on synthetic data, comparing linear and local linear forest (LLF) models with and without rank transformations.
- **`Examples/Synthetic_HeavyTailed.ipynb`** — applies developed approach towards synthetic heavy-tailde data with tail-dependence.
