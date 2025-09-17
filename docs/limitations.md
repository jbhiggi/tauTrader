# Limitations & Next Steps

This project is presented as a **proof of concept** rather than a production-ready trading system. Several limitations and opportunities for further development are worth highlighting.

---

## 1. Data Constraints
- The dataset is relatively small (~2,000 daily candles).  
- This limited sample size makes results sensitive to potential **regime shifts** between training and test periods.  
- Future work should explore **larger datasets** and **finer time resolutions** (e.g., hourly or minute bars) to improve statistical power and robustness.  
- Additional tests such as **distributional checks** (e.g., Z-tests for normalized features) could confirm whether training and test sets are drawn from comparable distributions.  

---

## 2. Model Training Choices
- Hyperparameters were selected using sensible defaults rather than systematic optimization.  
- A dedicated **Hyperparameter Optimization Pipeline (HOP)** is under development and is expected to yield significant performance improvements.  
- The model was trained for a limited number of epochs; extended training with early stopping criteria should be explored.  

---

## 3. Labeling Strategy
- Current labels are based on simple threshold definitions of price movement.  
- Alternative labeling schemes may provide more meaningful signals for practical strategies (e.g., incorporating volatility bands or event-driven triggers).  

---

## 4. Integration with Trading Signals
- At present, the model outputs class probabilities but is not connected to a live **buy/sell signal generator**.  
- Future integration could allow this model to act as a standalone strategy or as an **alpha stream** within a larger portfolio management framework.  

---

## 5. Evaluation Metrics
- Cross-validation was used to optimize precision at a chosen recall level.  
- While effective for classification tasks, **finance-specific metrics** such as **Sharpe ratio** or **expected value** would be more appropriate for deployment.  
- Extending the evaluation framework to optimize directly over these metrics would better align model outputs with real-world trading objectives.  

---

## Summary
This project demonstrates the feasibility of using sequence models with careful threshold calibration for stock movement prediction. However, scaling up the dataset, refining hyperparameters, improving label definitions, and aligning evaluation with finance-specific metrics are necessary steps before deployment in a live trading environment.
