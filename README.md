# Feature Importance Visualizer

![Streamlit](https://img.shields.io/badge/Streamlit-App-orange?logo=streamlit)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Overview

Feature Importance Visualizer is an interactive web application built with [Streamlit](https://streamlit.io/) that allows users to train a Random Forest model on various datasets and visualize the feature importance. This tool is designed for data scientists, machine learning enthusiasts, and students who wish to explore and understand how different features in a dataset contribute to model predictions.

## Features

- **Interactive Dataset Selection:** Choose from multiple publicly available datasets or upload your own CSV file.
- **Random Forest Training:** Train a Random Forest model with customizable hyperparameters.
- **Feature Importance Visualization:** Instantly view feature importance scores as bar charts and tables.
- **Download Results:** Export visualizations and importance scores for further analysis.
- **User-friendly UI:** Simple, intuitive interface powered by Streamlit.


## Getting Started

### Prerequisites

- Python 3.7+
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Coding-with-Akrash/feature-importance-visualizer.git
   cd feature-importance-visualizer
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Running the App

```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Select a dataset from the sidebar or upload your own.
2. Configure Random Forest hyperparameters (number of estimators, max depth, etc.).
3. Click 'Train Model' to fit the model and display feature importance.
4. View the results as a chart or table. Optionally, download the outputs.

## Supported Datasets

- Iris
- Wine
- Breast Cancer
- Custom CSV upload

## Project Structure

```
feature-importance-visualizer/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── README.md               # Project Readme

```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.
See the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)
- Open source datasets

---

*Created by [Coding-with-Akrash](https://github.com/Coding-with-Akrash)*
