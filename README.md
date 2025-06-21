# AutoML-ify: Your Simplified MLOps Pipeline

**AutoML-ify** is a powerful and intuitive tool that automates the entire machine learning workflow, from data upload to model deployment and monitoring. It's designed for speed and efficiency, allowing you to focus on deriving insights rather than on repetitive coding tasks.

---

## Key MLOps Features

*   **Guided Workflow**: A new wizard-style navigation guides you step-by-step through the pipeline, from uploading data to interpreting your model's results.
*   **Experiment Tracking**: Automatically logs every modelling run, including timestamps, model performance, and hyperparameters, so you can easily compare experiments.
*   **Model Monitoring**: Continuously monitor your model's performance on new data. Detect data drift by visually comparing feature distributions over time.
*   **Model Interpretation**: Gain insights into your model's predictions using SHAP (SHapley Additive exPlanations) for feature importance and summary plots.
*   **Automated Data Cleaning & Preprocessing**: Handles missing values, outliers, and categorical encoding automatically.
*   **Download and Go**: Download any trained model with a single click to use in your own applications.

---

## How It Works

1.  **Get Started**: Follow the guided "Next" and "Skip" buttons to navigate the pipeline.
2.  **Upload & Analyze**: Upload your CSV dataset and generate an in-depth Exploratory Data Analysis (EDA) report.
3.  **Clean & Engineer**: Apply automated data cleaning and create new features on the fly.
4.  **Train & Track**: Run a streamlined set of models. The best model is automatically selected, and the experiment is logged.
5.  **Monitor & Interpret**: Upload new data to check for performance degradation or data drift. Use SHAP plots to understand your model's behavior.
6.  **Download**: Select any experiment from the log and download the trained model.

---

## Get Started

1.  Clone the repository:
    ```bash
    git clone https://github.com/NevinSelby/AutoML-ify.git
    ```
2.  Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Streamlit app:
    ```bash
    streamlit run automl.py
    ``` 