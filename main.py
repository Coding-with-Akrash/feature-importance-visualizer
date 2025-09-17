import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris, load_wine, load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Feature Importance Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Available datasets
DATASETS = {
    "California Housing": fetch_california_housing,
    "Iris": load_iris,
    "Wine": load_wine,
    "Breast Cancer": load_breast_cancer
}

def load_data(dataset_name):
    """Load the selected dataset"""
    data_loader = DATASETS[dataset_name]()
    if hasattr(data_loader, 'data'):
        X = pd.DataFrame(data_loader.data, columns=data_loader.feature_names)
        y = data_loader.target
    else:
        X = pd.DataFrame(data_loader['data'], columns=data_loader['feature_names'])
        y = data_loader['target']
    return X, y, data_loader.feature_names if hasattr(data_loader, 'feature_names') else data_loader['feature_names']

def train_model(X, y, problem_type, n_estimators=100, max_depth=None):
    """Train Random Forest model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if problem_type == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, y_pred, scaler

def plot_feature_importance(feature_importance, feature_names, dataset_name):
    """Create feature importance plot"""
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(sorted_importance)), sorted_importance, align='center', 
                  color=plt.cm.viridis(np.linspace(0, 1, len(sorted_importance))),
                  edgecolor='black', alpha=0.7)
    
    # Add value labels
    for i, (value, bar) in enumerate(zip(sorted_importance, bars)):
        ax.text(value + 0.001, i, f'{value:.3f}', va='center', fontweight='bold', fontsize=10)
    
    # Customize plot
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=12)
    ax.set_xlabel('Feature Importance Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Features', fontsize=14, fontweight='bold')
    ax.set_title(f'Feature Importance - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Remove spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Feature Importance Visualizer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Dataset selection
        dataset_name = st.selectbox(
            "Select Dataset",
            list(DATASETS.keys()),
            index=0
        )
        
        # Model parameters
        st.subheader("Model Parameters")
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
        max_depth = st.slider("Max Depth", 2, 20, None)
        if max_depth == 20:  # Treat 20 as None (no limit)
            max_depth = None
        
        # Additional options
        st.subheader("Display Options")
        show_data = st.checkbox("Show Dataset Info", value=True)
        show_metrics = st.checkbox("Show Model Metrics", value=True)
    
    # Load data
    X, y, feature_names = load_data(dataset_name)
    
    # Determine problem type
    problem_type = "classification" if dataset_name != "California Housing" else "regression"
    
    # Train model
    with st.spinner("Training Random Forest model..."):
        model, X_train, X_test, y_train, y_test, y_pred, scaler = train_model(
            X, y, problem_type, n_estimators, max_depth
        )
    
    # Get feature importance
    feature_importance = model.feature_importances_
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Plot feature importance
        fig = plot_feature_importance(feature_importance, feature_names, dataset_name)
        st.pyplot(fig)
        
        # Additional visualization
        st.subheader("📈 Feature Importance Distribution")
        col1_1, col1_2 = st.columns(2)
        
        with col1_1:
            # Pie chart
            fig_pie, ax = plt.subplots(figsize=(8, 6))
            sorted_idx = np.argsort(feature_importance)[::-1]
            sorted_importance = feature_importance[sorted_idx][:6]  # Top 6 features
            sorted_features = [feature_names[i] for i in sorted_idx][:6]
            
            if len(feature_importance) > 6:
                sorted_importance = np.append(sorted_importance, np.sum(feature_importance[sorted_idx][6:]))
                sorted_features = sorted_features + ["Others"]
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_importance)))
            wedges, texts, autotexts = ax.pie(sorted_importance, labels=sorted_features, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
            plt.setp(autotexts, size=10, weight="bold")
            ax.set_title('Top Features Distribution (%)')
            st.pyplot(fig_pie)
        
        with col1_2:
            # Horizontal bar chart for percentages
            fig_bar, ax = plt.subplots(figsize=(8, 6))
            total_importance = np.sum(feature_importance)
            percentages = (feature_importance / total_importance) * 100
            sorted_idx = np.argsort(percentages)[::-1]
            
            # Show top 10 features
            top_n = min(10, len(feature_names))
            ax.barh(range(top_n), percentages[sorted_idx][:top_n][::-1], 
                   color=plt.cm.coolwarm(np.linspace(0, 1, top_n)))
            ax.set_yticks(range(top_n))
            ax.set_yticklabels([feature_names[i] for i in sorted_idx][:top_n][::-1])
            ax.set_xlabel('Importance Percentage (%)')
            ax.set_title('Top Features by Percentage')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig_bar)
    
    with col2:
        # Dataset info
        if show_data:
            st.subheader("📁 Dataset Information")
            st.metric("Number of Samples", X.shape[0])
            st.metric("Number of Features", X.shape[1])
            st.metric("Problem Type", "Classification" if problem_type == "classification" else "Regression")
            
            if problem_type == "classification":
                st.metric("Number of Classes", len(np.unique(y)))
        
        # Model metrics
        if show_metrics:
            st.subheader("📊 Model Performance")
            
            if problem_type == "classification":
                accuracy = accuracy_score(y_test, y_pred)
                st.metric("Accuracy", f"{accuracy:.4f}")
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.metric("Mean Squared Error", f"{mse:.4f}")
                st.metric("R² Score", f"{r2:.4f}")
            
            st.metric("Number of Trees", n_estimators)
            if max_depth:
                st.metric("Max Depth", max_depth)
            else:
                st.metric("Max Depth", "No limit")
        
        # Feature importance table
        st.subheader("📋 Feature Importance Scores")
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance,
            'Percentage': (feature_importance / np.sum(feature_importance)) * 100
        }).sort_values('Importance', ascending=False)
        
        st.dataframe(
            importance_df.head(10).style.format({
                'Importance': '{:.4f}',
                'Percentage': '{:.1f}%'
            }).background_gradient(cmap='Blues', subset=['Importance']),
            height=400
        )
        
        # Download button
        csv = importance_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Feature Importance CSV",
            data=csv,
            file_name="feature_importance.csv",
            mime="text/csv"
        )
    
    # Additional information
    with st.expander("ℹ️ About this App"):
        st.write("""
        This app visualizes feature importance from a Random Forest model using various datasets.
        
        **Features:**
        - Interactive dataset selection
        - Adjustable model parameters
        - Multiple visualization types
        - Performance metrics
        - Downloadable results
        
        **How to interpret:**
        - Higher importance scores indicate more influential features
        - Features with very low importance might be candidates for removal
        - The model uses Scikit-learn's Random Forest implementation
        """)

if __name__ == "__main__":
    main()
