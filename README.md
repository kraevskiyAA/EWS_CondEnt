# Early warning system for concept drift detection

Artem Kraevskiy, Artem Prokhorov, Evgeniy Sokolovskiy.

This repository contains code and examples of EWS for online concept drift detection.

**Abstract**: Financial markets of emerging economies are characterized by unusually high volatility, non-linear dependence and strong correlation in extreme values, making standard tools for early detection of financial crises inapplicable. In this paper we develop and apply a new early warning system (EWS) for what is known as online concept drift identification, permitting unrestricted dependence patterns and heavy tails. The key of the EWS is an effective detection of jumps in the conditional entropy of the financial indicators, rooted in change-point detection theory. We focus on finding significant information shifts between the interdependent time series under the curse of dimensionality.  First, we develop an approach that analyze the stability of linear relations between target and explanatory variables. 
Second, we propose extensions that capture non-linear interactions and do not impose any restriction on the variables’ distribution that are particularly the case for the emerging financial markets. 
Finally, we show the consistency of EWS’ results on synthetic data and apply the developed approach towards financial data of the emerging markets (Uzbekistan).



## Installing

Run to install local package EWS_CondEnt and other required packages:

```
pip install -e .
```

## Examples on synthetic data

One can find the examples from the original paper in "Examples/Synthetic_data.ipynb".


## Application towards Usbekistan financial market

The anticipated results for Uzbekistan financial markets are located in "Examples/Usbekistan_FinMarkets.ipynb". The original data is located in "Data/df_cond_ent.csv".


## Contacts

If you have any questions about this repository, please, contact authors at akraevskiy@hse.ru
