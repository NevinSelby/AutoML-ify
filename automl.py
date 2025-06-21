import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
import pickle
import shap
import matplotlib.pyplot as plt
import os
from datetime import datetime

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, RANSACRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import classification_report, confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="AutoML-ify", page_icon="✨")
PAGES = ["Home", "Upload", "EDA", "Data Cleaning", "Feature Engineering", "Modelling", "Experiment Tracking", "Monitoring", "Model Interpretation", "Download"]
EXPERIMENT_LOG_FILE = "experiment_log.csv"
MODEL_DIR = "models"

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --- Session State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'df' not in st.session_state:
    st.session_state.df = None
if 'best_model_path' not in st.session_state:
    st.session_state.best_model_path = None

# --- Helper Functions ---
def go_to(page: str):
    """Callback to switch pages."""
    st.session_state.page = page

# --- Sidebar ---
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AutoML-ify")

    # The radio button can be used for quick navigation, but the primary flow is via page buttons.
    page_index = PAGES.index(st.session_state.page) if st.session_state.page in PAGES else 0
    st.radio("Pipeline Steps", PAGES, index=page_index, key="navigation_radio", on_change=lambda: go_to(st.session_state.navigation_radio))
    
    st.info("Your Simplified ML Pipeline Automated.")

# --- Page Routing ---

if st.session_state.page == "Home":
    st.title("AutoML-ify: Your Simplified ML Pipeline Automated")
    st.write("""
    AutoML-ify is an end-to-end automated machine learning pipeline designed to simplify the entire ML workflow. With just a dataset as input, AutoML-ify handles data cleaning, preprocessing, exploratory data analysis (EDA), model selection, training, and visualization – all with minimal user intervention. Whether you're a data scientist, business analyst, or ML enthusiast, AutoML-ify makes machine learning accessible, efficient, and powerful.
    """)
    st.header("How It Works")
    st.write("""
    1. **Upload Your Dataset:** Provide your dataset in CSV format.
    2. **Analyze and Clean:** Explore your data with EDA and apply cleaning steps.
    3. **Train Models:** Automatically select and train the best model for your task.
    4. **Track & Monitor:** Track experiment history and monitor model performance over time.
    """)
    if st.button("Get Started"):
        go_to("Upload")

elif st.session_state.page == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset (CSV only)")
    if file: 
        df = pd.read_csv(file, index_col=None)
        st.session_state.df = df
        st.dataframe(df)
        st.success("Dataset uploaded successfully!")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Home", on_click=go_to, args=("Home",))
    with col2:
        if st.session_state.df is not None:
            st.button("Next: EDA", on_click=go_to, args=("EDA",))

elif st.session_state.page == "EDA":
    st.title("Exploratory Data Analysis")
    if st.session_state.df is not None:
        st.info("This may take some time to load for large datasets.")
        profile_df = ProfileReport(st.session_state.df, 
                                 title="Exploratory Data Analysis Report",
                                 explorative=True)
        st.components.v1.html(profile_df.to_html(), width=900, height=1200, scrolling=True)
    else:
        st.warning("Please upload a dataset first in the 'Upload' section.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Back to Upload", on_click=go_to, args=("Upload",))
    with col2:
        st.button("Skip to Modelling", on_click=go_to, args=("Modelling",))
    with col3:
        st.button("Next: Data Cleaning", on_click=go_to, args=("Data Cleaning",))

elif st.session_state.page == 'Data Cleaning':
    st.title("Data Cleaning")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        with st.expander("Automated Cleaning Steps"):
            st.write("1. **Drop columns with high missing values:** Columns with more than 50% missing values are dropped.")
            st.write("2. **One-hot encode categorical features:** Non-numeric columns are converted to numerical format.")
            st.write("3. **Impute missing values:** Remaining missing values are filled using linear interpolation.")

        if st.button("Clean Data"):
            with st.spinner("Cleaning data..."):
                threshold_cols = int(df.shape[0] * 0.5)
                df.dropna(axis=1, thresh=threshold_cols, inplace=True)
                for col in df.select_dtypes(include=['object', 'category']).columns:
                    df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
                    df.drop(col, axis=1, inplace=True)
                df.interpolate(method='linear', limit_direction='both', inplace=True)
                st.session_state.df = df
                st.dataframe(df)
                st.success("Data cleaned successfully!")
    else:
        st.warning("Please upload a dataset first.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Back to EDA", on_click=go_to, args=("EDA",))
    with col2:
        st.button("Skip to Modelling", on_click=go_to, args=("Modelling",))
    with col3:
        st.button("Next: Feature Engineering", on_click=go_to, args=("Feature Engineering",))

elif st.session_state.page == 'Feature Engineering':
    st.title("Feature Engineering")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        st.info("You can create new features from existing ones. Use 'df' to refer to the dataframe.")
        new_feature_string = st.text_area("Enter new feature expression", "df['new_col'] = df['col1'] + df['col2']")
        
        if st.button("Add Feature"):
            try:
                with st.spinner("Adding feature..."):
                    exec(new_feature_string, {'df': df})
                    st.session_state.df = df
                    st.dataframe(df)
                    st.success("Feature added successfully!")
            except Exception as e:
                st.error(f"Error adding feature: {e}")
    else:
        st.warning("Please upload a dataset first.")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Data Cleaning", on_click=go_to, args=("Data Cleaning",))
    with col2:
        st.button("Next: Modelling", on_click=go_to, args=("Modelling",))

elif st.session_state.page == "Modelling":
    st.title("Modelling")
    if st.session_state.df is not None:
        df = st.session_state.df.copy()
        classoreg = st.radio("Choose the type of problem", ["Regression", "Classification"])
        chosen_target = st.selectbox('Choose the Target Column', df.columns)
        
        if chosen_target:
            X = df.drop(chosen_target, axis=1)
            y = df[chosen_target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
            
            if st.button('Run Modelling'):
                models = []
                param_grid = {}
                if classoreg == "Regression":
                    st.info("Running a streamlined set of regression models for speed.")
                    models = [
                        ('Multiple Linear Regression', LinearRegression()),
                        ('Decision Tree', DecisionTreeRegressor(random_state=1)),
                        ('Random Forest', RandomForestRegressor(random_state=1)),
                    ]
                    param_grid = {
                        'Decision Tree': {'max_depth': [5, 10], 'min_samples_split': [2, 5]},
                        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
                    }
                else: # Classification
                    st.info("Running a streamlined set of classification models for speed.")
                    models = [
                        ('Logistic Regression', LogisticRegression(random_state=1)),
                        ('Random Forest', RandomForestClassifier(random_state=1)),
                        ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=1))
                    ]
                    param_grid = {
                        'Logistic Regression': {'C': [0.1, 1.0], 'solver': ['liblinear']},
                        'Random Forest': {'n_estimators': [50, 100], 'max_depth': [5, 10]},
                        'XGBoost': {'learning_rate': [0.1], 'max_depth': [3, 5], 'n_estimators': [100]}
                    }

                with st.spinner("Finding the best model... This might take a moment."):
                    best_model_obj = None
                    best_score = -np.inf
                    results = []
                    
                    for model_name, model in models:
                        try:
                            # Use fewer iterations for a faster search
                            grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid.get(model_name, {}), cv=3, n_jobs=-1, n_iter=3, random_state=1)
                            grid.fit(X, y)
                            results.append([model_name, grid.best_score_, grid.best_params_])
                            if grid.best_score_ > best_score:
                                best_score = grid.best_score_
                                best_model_obj = grid.best_estimator_
                        except Exception:
                            continue
                    
                    st.write("Modelling Results:")
                    results_df = pd.DataFrame(results, columns=['Model', 'Best Score', 'Best Parameters'])
                    st.dataframe(results_df)

                    if best_model_obj is not None:
                        # MLOps: Log the experiment
                        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_path = os.path.join(MODEL_DIR, f"model_{run_id}.pkl")
                        data_path = os.path.join(MODEL_DIR, f"data_{run_id}.csv")
                        st.session_state.best_model_path = model_path
                        
                        pickle.dump(best_model_obj, open(model_path, 'wb'))
                        df.to_csv(data_path, index=False)

                        log_entry = {
                            "timestamp": run_id,
                            "model_type": type(best_model_obj).__name__,
                            "problem_type": classoreg,
                            "target_column": chosen_target,
                            "score": f"{best_score:.4f}",
                            "parameters": str(best_model_obj.get_params()),
                            "model_path": model_path,
                            "data_path": data_path
                        }
                        log_df = pd.DataFrame([log_entry])
                        if os.path.exists(EXPERIMENT_LOG_FILE):
                            log_df.to_csv(EXPERIMENT_LOG_FILE, mode='a', header=False, index=False)
                        else:
                            log_df.to_csv(EXPERIMENT_LOG_FILE, index=False)

                        st.success(f'The best model is **{type(best_model_obj).__name__}** with a score of **{best_score:.4f}**.')
                        st.info(f"This experiment run ({run_id}) has been logged. You can view it in 'Experiment Tracking'.")
                    else:
                        st.error("Could not determine the best model.")
    else:
        st.warning("Please upload a dataset first.")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back", on_click=go_to, args=("Feature Engineering",))
    with col2:
        if st.session_state.best_model_path:
            st.button("Next: Track & Monitor", on_click=go_to, args=("Experiment Tracking",))

elif st.session_state.page == 'Experiment Tracking':
    st.title("Experiment Tracking")
    st.info("Here is a log of all your modelling runs.")
    if os.path.exists(EXPERIMENT_LOG_FILE):
        log_df = pd.read_csv(EXPERIMENT_LOG_FILE)
        st.dataframe(log_df)
        if st.button("Clear Experiment Log"):
            os.remove(EXPERIMENT_LOG_FILE)
            st.rerun()
    else:
        st.info("No experiments have been logged yet.")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Modelling", on_click=go_to, args=("Modelling",))
    with col2:
        st.button("Next: Monitor a Model", on_click=go_to, args=("Monitoring",))

elif st.session_state.page == 'Monitoring':
    st.title("Model Monitoring")
    st.info("Check a trained model's performance on new data and monitor for data drift.")

    if not os.path.exists(EXPERIMENT_LOG_FILE):
        st.warning("No experiments found. Please run the modelling step first.")
    else:
        log_df = pd.read_csv(EXPERIMENT_LOG_FILE)
        run_to_monitor = st.selectbox("Select an experiment run to monitor", log_df['timestamp'])
        
        if run_to_monitor:
            run_details = log_df[log_df['timestamp'] == run_to_monitor].iloc[0]
            st.write("### Monitoring Run:", run_details['timestamp'])
            
            new_data_file = st.file_uploader("Upload new data for monitoring", type=['csv'])
            if new_data_file:
                new_df = pd.read_csv(new_data_file)
                st.write("New Data Preview:")
                st.dataframe(new_df.head())

                try:
                    # Load original data and model
                    model = pickle.load(open(run_details['model_path'], 'rb'))
                    original_df = pd.read_csv(run_details['data_path'])
                    target_column = run_details['target_column']
                    
                    X_new = new_df.drop(target_column, axis=1, errors='ignore')
                    y_new = new_df[target_column]

                    # Check for column consistency
                    X_original = original_df.drop(target_column, axis=1)
                    missing_cols = set(X_original.columns) - set(X_new.columns)
                    extra_cols = set(X_new.columns) - set(X_original.columns)

                    if missing_cols: st.warning(f"Missing columns in new data: {missing_cols}")
                    if extra_cols: st.warning(f"Extra columns in new data: {extra_cols}")
                    
                    X_new = X_new[X_original.columns].fillna(0) # Align columns and fill NaNs

                    # Evaluate performance on new data
                    st.subheader("Performance on New Data")
                    predictions = model.predict(X_new)
                    if run_details['problem_type'] == 'Regression':
                        score = r2_score(y_new, predictions)
                        st.metric("R² Score", f"{score:.4f}")
                    else:
                        score = accuracy_score(y_new, predictions)
                        st.metric("Accuracy Score", f"{score:.4f}")
                        st.text("Classification Report:")
                        st.text(classification_report(y_new, predictions))

                    # Data Drift Visualization
                    st.subheader("Data Drift Analysis")
                    st.info("Comparing feature distributions between original training data and new data.")
                    
                    numerical_cols = X_original.select_dtypes(include=np.number).columns
                    feature_to_plot = st.selectbox("Select a feature to analyze drift", numerical_cols)
                    
                    if feature_to_plot:
                        fig, ax = plt.subplots()
                        ax.hist(original_df[feature_to_plot], bins=20, alpha=0.7, label='Original Data')
                        ax.hist(new_df[feature_to_plot], bins=20, alpha=0.7, label='New Data')
                        ax.legend()
                        ax.set_title(f"Distribution for {feature_to_plot}")
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"An error occurred during monitoring: {e}")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Tracking", on_click=go_to, args=("Experiment Tracking",))
    with col2:
        st.button("Next: Interpret a Model", on_click=go_to, args=("Model Interpretation",))


elif st.session_state.page == 'Model Interpretation':
    st.title("Model Interpretation")
    if not st.session_state.best_model_path:
        st.warning("No model has been trained in the current session. Please run the modelling step first.")
    elif st.session_state.df is None:
        st.warning("No data found. Please upload a dataset in the 'Upload' step.")
    else:
        try:
            model = pickle.load(open(st.session_state.best_model_path, 'rb'))
            df = st.session_state.df.copy()
            st.info("Interpreting the model from the most recent run. Choose the same target column used for modelling.")
            
            # This needs to be more robust, maybe save target to state
            target_col_options = list(df.columns)
            chosen_target = st.selectbox('Choose the Target Column', target_col_options)

            if chosen_target:
                X = df.drop(chosen_target, axis=1).copy()
                explainer = shap.Explainer(model, X)
                shap_values = explainer(X)

                st.subheader("SHAP Feature Importance")
                fig1, ax1 = plt.subplots()
                shap.summary_plot(shap_values, X, plot_type="bar", show=False)
                st.pyplot(fig1)
                
                st.subheader("SHAP Summary Plot (Beeswarm)")
                fig2, ax2 = plt.subplots()
                shap.summary_plot(shap_values, X, show=False)
                st.pyplot(fig2)

        except Exception as e:
            st.error(f"Error in Model Interpretation: {e}")

    col1, col2 = st.columns(2)
    with col1:
        st.button("Back to Monitoring", on_click=go_to, args=("Monitoring",))
    with col2:
        st.button("Next: Download Model", on_click=go_to, args=("Download",))

elif st.session_state.page == "Download":
    st.title("Download a Model")
    st.info("Select an experiment run to download the corresponding trained model.")

    if not os.path.exists(EXPERIMENT_LOG_FILE):
        st.warning("No experiments found to download models from.")
    else:
        log_df = pd.read_csv(EXPERIMENT_LOG_FILE)
        run_to_download = st.selectbox("Select an experiment run", log_df['timestamp'])
        
        if run_to_download:
            run_details = log_df[log_df['timestamp'] == run_to_download].iloc[0]
            model_path = run_details['model_path']
            
            with open(model_path, 'rb') as f:
                st.download_button('Download Model', f, file_name=os.path.basename(model_path))

    st.button("Back to Interpretation", on_click=go_to, args=("Model Interpretation",))
