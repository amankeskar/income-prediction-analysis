# Income Prediction Analysis

## ML Project for Data Science Course

This is my final project for my machine learning course where I built an income prediction model using the Adult Census dataset. I learned a lot about different aspects of ML beyond just training models.

## What I Built

I created an XGBoost model that predicts whether someone makes more than $50K per year based on census data. The model gets about 87.5% accuracy which seems pretty good for this dataset.

### Main Features
- XGBoost classifier for income prediction
- Some analysis on whether the model is fair to different groups
- Basic monitoring to track how the model performs
- Simple dashboard to show business value
- Jupyter notebook walking through everything

## How to Run

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Open the main notebook:
```bash
jupyter notebook 02_enterprise_platform.ipynb
```

3. Run through the cells to see the analysis

## Project Structure

```
├── src/                          # Python modules I wrote
│   ├── fairness_analysis.py     # Checking for bias in the model
│   ├── model_monitoring.py      # Basic performance tracking
│   └── executive_dashboard.py   # Business impact stuff
├── data/                         # Dataset and processed files
├── notebooks/                    # Main analysis notebook
├── reports/                      # Generated charts and reports
└── requirements.txt              # Python packages needed
```

## What I Learned

Working on this project taught me about:

1. **Model Performance**: How to properly evaluate ML models beyond just accuracy
2. **Fairness**: Why it's important to check if models treat different groups fairly
3. **Monitoring**: How to track model performance over time (simulation)
4. **Business Value**: Ways to explain technical work to non-technical people
5. **Documentation**: How to organize and document an ML project

## Main Results

- **Model Performance**: 87.5% accuracy on test set
- **Fairness Check**: Found some differences between groups that might need attention
- **Business Impact**: Estimated the model could help with decision making

## Files Overview

- `02_enterprise_platform.ipynb` - Main notebook with all the analysis
- `src/fairness_analysis.py` - Code for checking bias and fairness
- `src/model_monitoring.py` - Code for tracking model performance
- `src/executive_dashboard.py` - Code for business impact analysis

## Technologies Used

- Python (pandas, scikit-learn, xgboost)
- Jupyter notebooks
- Matplotlib/seaborn for visualizations
- Some custom modules I wrote for different parts

## Next Steps

If I continue this project, I'd like to:
- Try other algorithms and compare results
- Add more fairness metrics
- Test on different datasets
- Maybe build a simple web interface

---

This was a good learning experience for understanding ML beyond just training models. The monitoring and fairness parts were especially interesting since I hadn't thought about those aspects much before.
