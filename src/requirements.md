# Required Dependencies

This document lists all the required Python packages to run the code in this project.

## Core Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- lightgbm
- matplotlib
- lime
- shap
- IPython

## Installation

You can install all required packages using pip:

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib lime shap IPython
```

## Version Information

The code has been tested with the following versions:
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- lightgbm >= 3.3.0
- matplotlib >= 3.4.0
- lime >= 0.2.0
- shap >= 0.40.0
- IPython >= 7.0.0

## Additional Notes

1. The code uses custom modules:
   - `bl` (BareLogic)
   - `neural_net`
   - `relieff`
   - `stats`

2. Make sure these custom modules are in your Python path or in the same directory as the main script.

3. For Windows users:
   - Some packages might require Microsoft Visual C++ Build Tools
   - LightGBM might require additional setup for Windows

4. For optimal performance:
   - Consider using a virtual environment
   - Install packages in the order listed above
   - Some packages might require additional system-level dependencies depending on your OS 