"""
CONFIRM Data Converter - Main Application
Transforms raw client data into properly formatted contingency tables
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, Any, List
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data_import import DataImporter
from schema_detector import SchemaDetector
from variable_selector import VariableSelector
from contingency_generator import ContingencyTableGenerator
from recommendation_engine import RecommendationEngine

# Page configuration
st.set_page_config(
    page_title="CONFIRM Data Converter",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .info-box {
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    h1 {
        color: #2c3e50;
        border-bottom: 3px solid #4CAF50;
        padding-bottom: 10px;
    }
    h2 {
        color: #34495e;
        margin-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class ConfirmDataConverter:
    """Main application class for CONFIRM Data Converter"""
    
    def __init__(self):
        """Initialize the application"""
        self.initialize_session_state()
        self.data_importer = DataImporter()
        self.schema_detector = SchemaDetector()
        self.variable_selector = VariableSelector()
        self.contingency_generator = ContingencyTableGenerator()
        self.recommendation_engine = RecommendationEngine()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'schema' not in st.session_state:
            st.session_state.schema = None
        if 'selected_row_var' not in st.session_state:
            st.session_state.selected_row_var = None
        if 'selected_col_var' not in st.session_state:
            st.session_state.selected_col_var = None
        if 'contingency_table' not in st.session_state:
            st.session_state.contingency_table = None
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = []
    
    def render_header(self):
        """Render application header"""
        st.title("🔄 CONFIRM Data Converter")
        st.markdown("""
        <div class="info-box">
        <strong>Welcome to CONFIRM Data Converter!</strong><br>
        Transform your raw data into properly formatted contingency tables for statistical validation.
        This tool supports multiple input formats and provides intelligent recommendations for variable selection.
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with navigation and info"""
        with st.sidebar:
            st.header("📋 Navigation")
            
            # Progress indicator
            progress = 0
            if st.session_state.data is not None:
                progress += 25
            if st.session_state.schema is not None:
                progress += 25
            if st.session_state.selected_row_var and st.session_state.selected_col_var:
                progress += 25
            if st.session_state.contingency_table is not None:
                progress += 25
            
            st.progress(progress / 100)
            st.caption(f"Progress: {progress}%")
            
            st.markdown("---")
            
            # Quick stats
            if st.session_state.data is not None:
                st.subheader("📊 Data Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rows", f"{len(st.session_state.data):,}")
                with col2:
                    st.metric("Columns", f"{len(st.session_state.data.columns):,}")
                
                # Memory usage
                memory_usage = st.session_state.data.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory", f"{memory_usage:.2f} MB")
            
            st.markdown("---")
            
            # Help section
            with st.expander("ℹ️ How to Use"):
                st.markdown("""
                1. **Import Data**: Upload your file or connect to database
                2. **Review Schema**: Check detected data types
                3. **Select Variables**: Choose row and column variables
                4. **Generate Table**: Create contingency table
                5. **Export Results**: Download formatted table
                """)
            
            # About section
            with st.expander("📖 About CONFIRM"):
                st.markdown("""
                CONFIRM (Contingency Table Formatter) is designed to:
                - Support multiple data formats
                - Automatically detect data types
                - Provide intelligent recommendations
                - Generate validated contingency tables
                - Export results in various formats
                """)
    
    def render_data_import(self):
        """Render data import section"""
        st.header("1️⃣ Data Import")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            import_method = st.selectbox(
                "Select Import Method",
                ["Upload File", "Database Connection", "Sample Data"],
                help="Choose how to import your data"
            )
        
        with col2:
            if import_method == "Upload File":
                file_type = st.selectbox(
                    "File Type",
                    ["CSV", "Excel", "JSON"],
                    help="Select the file format"
                )
        
        if import_method == "Upload File":
            uploaded_file = st.file_uploader(
                f"Choose a {file_type} file",
                type={"CSV": ["csv"], "Excel": ["xlsx", "xls"], "JSON": ["json"]}[file_type],
                help="Upload your data file"
            )
            
            if uploaded_file is not None:
                with st.spinner("Loading data..."):
                    try:
                        if file_type == "CSV":
                            st.session_state.data = self.data_importer.import_csv(uploaded_file)
                        elif file_type == "Excel":
                            st.session_state.data = self.data_importer.import_excel(uploaded_file)
                        elif file_type == "JSON":
                            st.session_state.data = self.data_importer.import_json(uploaded_file)
                        
                        st.success(f"✅ Successfully loaded {len(st.session_state.data):,} rows")
                        
                        # Auto-detect schema
                        st.session_state.schema = self.schema_detector.detect_schema(st.session_state.data)
                        
                    except Exception as e:
                        st.error(f"❌ Error loading file: {str(e)}")
        
        elif import_method == "Database Connection":
            self.render_database_connection()
        
        elif import_method == "Sample Data":
            self.load_sample_data()
    
    def render_database_connection(self):
        """Render database connection interface"""
        db_type = st.selectbox(
            "Database Type",
            ["PostgreSQL", "MySQL", "SQLite"],
            help="Select your database type"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if db_type != "SQLite":
                host = st.text_input("Host", value="localhost")
                port = st.number_input("Port", value=5432 if db_type == "PostgreSQL" else 3306)
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
            database = st.text_input("Database Name" if db_type != "SQLite" else "Database Path")
        
        with col2:
            query = st.text_area(
                "SQL Query",
                value="SELECT * FROM your_table LIMIT 1000",
                height=150,
                help="Enter your SQL query to fetch data"
            )
        
        if st.button("Connect and Load", type="primary"):
            with st.spinner("Connecting to database..."):
                try:
                    if db_type == "SQLite":
                        connection_string = f"sqlite:///{database}"
                    elif db_type == "PostgreSQL":
                        connection_string = f"postgresql://{username}:{password}@{host}:{port}/{database}"
                    else:  # MySQL
                        connection_string = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
                    
                    st.session_state.data = self.data_importer.import_from_database(
                        connection_string, query
                    )
                    st.success(f"✅ Successfully loaded {len(st.session_state.data):,} rows")
                    
                    # Auto-detect schema
                    st.session_state.schema = self.schema_detector.detect_schema(st.session_state.data)
                    
                except Exception as e:
                    st.error(f"❌ Database connection error: {str(e)}")
    
    def load_sample_data(self):
        """Load sample dataset for demonstration"""
        sample_datasets = {
            "Sales Data": self.generate_sample_sales_data,
            "Survey Results": self.generate_sample_survey_data,
            "Medical Records": self.generate_sample_medical_data
        }
        
        selected_sample = st.selectbox(
            "Select Sample Dataset",
            list(sample_datasets.keys()),
            help="Choose a sample dataset to explore the tool"
        )
        
        if st.button("Load Sample Data", type="primary"):
            with st.spinner("Generating sample data..."):
                st.session_state.data = sample_datasets[selected_sample]()
                st.success(f"✅ Loaded {selected_sample} with {len(st.session_state.data):,} rows")
                
                # Auto-detect schema
                st.session_state.schema = self.schema_detector.detect_schema(st.session_state.data)
    
    def generate_sample_sales_data(self) -> pd.DataFrame:
        """Generate sample sales data"""
        np.random.seed(42)
        n_records = 1000
        
        data = {
            'customer_id': range(1, n_records + 1),
            'region': np.random.choice(['North', 'South', 'East', 'West'], n_records),
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], n_records),
            'customer_type': np.random.choice(['New', 'Returning', 'VIP'], n_records, p=[0.3, 0.5, 0.2]),
            'sales_amount': np.random.lognormal(5, 1.5, n_records),
            'quantity': np.random.poisson(3, n_records),
            'satisfaction_score': np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied'], 
                                                 n_records, p=[0.3, 0.4, 0.2, 0.1]),
            'purchase_month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], n_records)
        }
        
        return pd.DataFrame(data)
    
    def generate_sample_survey_data(self) -> pd.DataFrame:
        """Generate sample survey data"""
        np.random.seed(42)
        n_records = 800
        
        data = {
            'respondent_id': range(1, n_records + 1),
            'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_records),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_records, p=[0.3, 0.4, 0.25, 0.05]),
            'employment': np.random.choice(['Employed', 'Self-employed', 'Unemployed', 'Student', 'Retired'], n_records),
            'income_bracket': np.random.choice(['<30k', '30-60k', '60-100k', '>100k'], n_records),
            'preference': np.random.choice(['Option A', 'Option B', 'Option C'], n_records),
            'satisfaction': np.random.choice(['1', '2', '3', '4', '5'], n_records),
            'recommend': np.random.choice(['Yes', 'No', 'Maybe'], n_records, p=[0.6, 0.2, 0.2])
        }
        
        return pd.DataFrame(data)
    
    def generate_sample_medical_data(self) -> pd.DataFrame:
        """Generate sample medical data"""
        np.random.seed(42)
        n_records = 500
        
        data = {
            'patient_id': range(1, n_records + 1),
            'age_category': np.random.choice(['Child', 'Adult', 'Senior'], n_records, p=[0.2, 0.5, 0.3]),
            'gender': np.random.choice(['Male', 'Female'], n_records),
            'diagnosis': np.random.choice(['Type A', 'Type B', 'Type C', 'None'], n_records, p=[0.25, 0.25, 0.25, 0.25]),
            'treatment': np.random.choice(['Treatment 1', 'Treatment 2', 'Treatment 3', 'No Treatment'], n_records),
            'outcome': np.random.choice(['Improved', 'Stable', 'Declined'], n_records, p=[0.6, 0.3, 0.1]),
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], n_records, p=[0.5, 0.35, 0.15]),
            'follow_up': np.random.choice(['Yes', 'No'], n_records, p=[0.7, 0.3])
        }
        
        return pd.DataFrame(data)
    
    def render_data_preview(self):
        """Render data preview section"""
        if st.session_state.data is not None:
            st.header("2️⃣ Data Preview & Schema")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📊 Schema", "📈 Statistics", "🔍 Data Quality"])
            
            with tab1:
                st.subheader("First 100 Rows")
                # Display options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_index = st.checkbox("Show Index", value=False)
                with col2:
                    n_rows = st.slider("Number of rows", 10, 100, 50)
                with col3:
                    search_term = st.text_input("Search in data", placeholder="Type to filter...")
                
                # Apply search filter if provided
                display_data = st.session_state.data.head(100)
                if search_term:
                    mask = display_data.apply(lambda x: x.astype(str).str.contains(search_term, case=False)).any(axis=1)
                    display_data = display_data[mask]
                
                st.dataframe(
                    display_data.head(n_rows),
                    use_container_width=True,
                    hide_index=not show_index,
                    height=400
                )
            
            with tab2:
                st.subheader("Data Schema")
                schema_df = pd.DataFrame(st.session_state.schema).T
                schema_df = schema_df.reset_index()
                schema_df.columns = ['Column', 'Data Type', 'Nullable', 'Unique Values', 'Sample Values']
                
                # Add color coding for data types
                def color_data_type(val):
                    colors = {
                        'categorical': 'background-color: #90EE90',
                        'numeric': 'background-color: #87CEEB',
                        'datetime': 'background-color: #FFB6C1',
                        'text': 'background-color: #F0E68C'
                    }
                    return colors.get(val, '')
                
                styled_schema = schema_df.style.applymap(color_data_type, subset=['Data Type'])
                st.dataframe(styled_schema, use_container_width=True, hide_index=True)
                
                # Schema summary
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    categorical_cols = sum(1 for v in st.session_state.schema.values() if v['data_type'] == 'categorical')
                    st.metric("Categorical", categorical_cols)
                with col2:
                    numeric_cols = sum(1 for v in st.session_state.schema.values() if v['data_type'] == 'numeric')
                    st.metric("Numeric", numeric_cols)
                with col3:
                    datetime_cols = sum(1 for v in st.session_state.schema.values() if v['data_type'] == 'datetime')
                    st.metric("DateTime", datetime_cols)
                with col4:
                    text_cols = sum(1 for v in st.session_state.schema.values() if v['data_type'] == 'text')
                    st.metric("Text", text_cols)
            
            with tab3:
                st.subheader("Column Statistics")
                selected_col = st.selectbox("Select column for detailed statistics", st.session_state.data.columns)
                
                col_data = st.session_state.data[selected_col]
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Basic Statistics**")
                    stats_dict = {
                        "Count": len(col_data),
                        "Unique": col_data.nunique(),
                        "Missing": col_data.isna().sum(),
                        "Missing %": f"{col_data.isna().sum() / len(col_data) * 100:.2f}%"
                    }
                    
                    if pd.api.types.is_numeric_dtype(col_data):
                        stats_dict.update({
                            "Mean": f"{col_data.mean():.2f}",
                            "Std": f"{col_data.std():.2f}",
                            "Min": f"{col_data.min():.2f}",
                            "Max": f"{col_data.max():.2f}"
                        })
                    
                    for key, value in stats_dict.items():
                        st.metric(key, value)
                
                with col2:
                    st.markdown("**Distribution Visualization**")
                    if pd.api.types.is_numeric_dtype(col_data):
                        fig = px.histogram(col_data.dropna(), nbins=30, title=f"Distribution of {selected_col}")
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        value_counts = col_data.value_counts().head(10)
                        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                                   title=f"Top 10 Values in {selected_col}")
                        fig.update_layout(xaxis_title="Value", yaxis_title="Count", height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Data Quality Report")
                
                # Calculate data quality metrics
                quality_metrics = []
                for col in st.session_state.data.columns:
                    missing_pct = (st.session_state.data[col].isna().sum() / len(st.session_state.data)) * 100
                    unique_pct = (st.session_state.data[col].nunique() / len(st.session_state.data)) * 100
                    
                    quality_metrics.append({
                        'Column': col,
                        'Completeness': f"{100 - missing_pct:.1f}%",
                        'Uniqueness': f"{unique_pct:.1f}%",
                        'Data Type': st.session_state.schema[col]['data_type'],
                        'Quality Score': self.calculate_quality_score(col)
                    })
                
                quality_df = pd.DataFrame(quality_metrics)
                
                # Color code quality scores
                def color_quality_score(val):
                    if isinstance(val, str) and val.endswith('%'):
                        score = float(val[:-1])
                        if score >= 90:
                            return 'background-color: #90EE90'
                        elif score >= 70:
                            return 'background-color: #FFD700'
                        else:
                            return 'background-color: #FFB6C1'
                    return ''
                
                styled_quality = quality_df.style.applymap(color_quality_score, subset=['Quality Score'])
                st.dataframe(styled_quality, use_container_width=True, hide_index=True)
                
                # Overall data quality score
                avg_quality = np.mean([float(m['Quality Score'][:-1]) for m in quality_metrics])
                st.metric("Overall Data Quality Score", f"{avg_quality:.1f}%")
    
    def calculate_quality_score(self, column: str) -> str:
        """Calculate quality score for a column"""
        col_data = st.session_state.data[column]
        
        # Factors for quality score
        completeness = (1 - col_data.isna().sum() / len(col_data)) * 40
        consistency = 30  # Simplified for now
        
        # Check for appropriate cardinality
        uniqueness = col_data.nunique() / len(col_data)
        if uniqueness < 0.5:  # Good for categorical
            cardinality_score = 30
        elif uniqueness > 0.95:  # Might be ID column
            cardinality_score = 10
        else:
            cardinality_score = 20
        
        total_score = completeness + consistency + cardinality_score
        return f"{total_score:.1f}%"
    
    def render_variable_selection(self):
        """Render variable selection interface"""
        if st.session_state.data is not None:
            st.header("3️⃣ Variable Selection")
            
            # Get recommendations
            if not st.session_state.recommendations:
                st.session_state.recommendations = self.recommendation_engine.get_recommendations(
                    st.session_state.data, st.session_state.schema
                )
            
            # Display recommendations if available
            if st.session_state.recommendations:
                st.markdown("""
                <div class="info-box">
                <strong>💡 Recommendations:</strong><br>
                Based on your data structure, we recommend the following variable pairs for contingency tables:
                </div>
                """, unsafe_allow_html=True)
                
                for i, rec in enumerate(st.session_state.recommendations[:3], 1):
                    col1, col2, col3 = st.columns([3, 3, 2])
                    with col1:
                        st.write(f"**Option {i}:** Row: `{rec['row_var']}`")
                    with col2:
                        st.write(f"Column: `{rec['col_var']}`")
                    with col3:
                        if st.button(f"Use Option {i}", key=f"rec_{i}"):
                            st.session_state.selected_row_var = rec['row_var']
                            st.session_state.selected_col_var = rec['col_var']
                            st.rerun()
            
            st.markdown("---")
            
            # Manual selection
            col1, col2 = st.columns(2)
            
            # Get categorical columns
            categorical_cols = [col for col, info in st.session_state.schema.items() 
                              if info['data_type'] == 'categorical']
            
            with col1:
                st.subheader("Row Variable (X-axis)")
                selected_row = st.selectbox(
                    "Select row variable",
                    categorical_cols,
                    index=categorical_cols.index(st.session_state.selected_row_var) 
                        if st.session_state.selected_row_var in categorical_cols else 0,
                    help="This variable will form the rows of your contingency table"
                )
                
                if selected_row:
                    st.session_state.selected_row_var = selected_row
                    self.display_variable_info(selected_row)
            
            with col2:
                st.subheader("Column Variable (Y-axis)")
                # Filter out the selected row variable
                available_cols = [col for col in categorical_cols if col != selected_row]
                
                selected_col = st.selectbox(
                    "Select column variable",
                    available_cols,
                    index=available_cols.index(st.session_state.selected_col_var) 
                        if st.session_state.selected_col_var in available_cols else 0,
                    help="This variable will form the columns of your contingency table"
                )
                
                if selected_col:
                    st.session_state.selected_col_var = selected_col
                    self.display_variable_info(selected_col)
            
            # Categorization options for continuous variables
            st.markdown("---")
            st.subheader("⚙️ Advanced Options")
            
            with st.expander("Continuous Variable Categorization"):
                numeric_cols = [col for col, info in st.session_state.schema.items() 
                               if info['data_type'] == 'numeric']
                
                if numeric_cols:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        cont_var = st.selectbox("Select continuous variable", numeric_cols)
                    
                    with col2:
                        method = st.selectbox("Categorization method", 
                                            ["Equal Width", "Equal Frequency", "Custom Breaks"])
                    
                    with col3:
                        n_bins = st.number_input("Number of categories", min_value=2, max_value=10, value=4)
                    
                    if st.button("Categorize Variable"):
                        self.categorize_continuous_variable(cont_var, method, n_bins)
                else:
                    st.info("No continuous variables found in the dataset")
    
    def display_variable_info(self, variable: str):
        """Display information about selected variable"""
        col_data = st.session_state.data[variable]
        value_counts = col_data.value_counts()
        
        # Create metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Values", value_counts.shape[0])
        with col2:
            st.metric("Missing", f"{col_data.isna().sum():,}")
        with col3:
            st.metric("Mode", value_counts.index[0] if len(value_counts) > 0 else "N/A")
        
        # Show value distribution
        if len(value_counts) <= 10:
            st.markdown("**Value Distribution:**")
            for val, count in value_counts.items():
                pct = (count / len(col_data)) * 100
                st.progress(pct / 100)
                st.caption(f"{val}: {count:,} ({pct:.1f}%)")
        else:
            st.info(f"Showing top 5 of {len(value_counts)} unique values")
            for val, count in value_counts.head(5).items():
                pct = (count / len(col_data)) * 100
                st.caption(f"{val}: {count:,} ({pct:.1f}%)")
    
    def categorize_continuous_variable(self, variable: str, method: str, n_bins: int):
        """Categorize a continuous variable"""
        try:
            col_data = st.session_state.data[variable].dropna()
            
            if method == "Equal Width":
                bins = pd.cut(col_data, bins=n_bins)
            elif method == "Equal Frequency":
                bins = pd.qcut(col_data, q=n_bins, duplicates='drop')
            else:  # Custom Breaks
                st.warning("Custom breaks not yet implemented")
                return
            
            # Create new categorical column
            new_col_name = f"{variable}_cat"
            st.session_state.data[new_col_name] = bins
            
            # Update schema
            st.session_state.schema[new_col_name] = {
                'data_type': 'categorical',
                'nullable': True,
                'unique_values': bins.nunique(),
                'sample_values': list(bins.value_counts().index[:3])
            }
            
            st.success(f"✅ Created categorical variable: {new_col_name}")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error categorizing variable: {str(e)}")
    
    def render_contingency_table(self):
        """Render contingency table generation"""
        if st.session_state.selected_row_var and st.session_state.selected_col_var:
            st.header("4️⃣ Contingency Table Generation")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                normalize = st.selectbox(
                    "Normalization",
                    ["None", "Row", "Column", "All"],
                    help="Choose how to normalize the contingency table"
                )
            
            with col2:
                show_margins = st.checkbox("Show Margins", value=True, 
                                          help="Include row and column totals")
            
            with col3:
                if st.button("Generate Table", type="primary"):
                    self.generate_contingency_table(normalize, show_margins)
            
            # Display contingency table if generated
            if st.session_state.contingency_table is not None:
                st.markdown("---")
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Table View", "📈 Visualization", 
                                                   "📉 Statistics", "💾 Export"])
                
                with tab1:
                    st.subheader("Contingency Table")
                    
                    # Display options
                    display_format = st.radio(
                        "Display Format",
                        ["Counts", "Percentages", "Both"],
                        horizontal=True
                    )
                    
                    if display_format == "Counts":
                        st.dataframe(
                            st.session_state.contingency_table['counts'],
                            use_container_width=True
                        )
                    elif display_format == "Percentages":
                        st.dataframe(
                            st.session_state.contingency_table['percentages'],
                            use_container_width=True
                        )
                    else:  # Both
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**Counts**")
                            st.dataframe(st.session_state.contingency_table['counts'])
                        with col2:
                            st.markdown("**Percentages**")
                            st.dataframe(st.session_state.contingency_table['percentages'])
                
                with tab2:
                    st.subheader("Visual Representation")
                    
                    viz_type = st.selectbox(
                        "Visualization Type",
                        ["Heatmap", "Stacked Bar", "Grouped Bar", "3D Surface"]
                    )
                    
                    if viz_type == "Heatmap":
                        fig = px.imshow(
                            st.session_state.contingency_table['counts'],
                            labels=dict(x=st.session_state.selected_col_var,
                                      y=st.session_state.selected_row_var,
                                      color="Count"),
                            title=f"Contingency Table: {st.session_state.selected_row_var} vs {st.session_state.selected_col_var}",
                            text_auto=True
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Stacked Bar":
                        df_plot = st.session_state.contingency_table['counts'].reset_index()
                        df_melted = df_plot.melt(id_vars=[st.session_state.selected_row_var],
                                                var_name=st.session_state.selected_col_var,
                                                value_name='Count')
                        
                        fig = px.bar(df_melted, 
                                   x=st.session_state.selected_row_var,
                                   y='Count',
                                   color=st.session_state.selected_col_var,
                                   title=f"Stacked Bar: {st.session_state.selected_row_var} by {st.session_state.selected_col_var}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    elif viz_type == "Grouped Bar":
                        df_plot = st.session_state.contingency_table['counts'].reset_index()
                        df_melted = df_plot.melt(id_vars=[st.session_state.selected_row_var],
                                                var_name=st.session_state.selected_col_var,
                                                value_name='Count')
                        
                        fig = px.bar(df_melted, 
                                   x=st.session_state.selected_row_var,
                                   y='Count',
                                   color=st.session_state.selected_col_var,
                                   barmode='group',
                                   title=f"Grouped Bar: {st.session_state.selected_row_var} by {st.session_state.selected_col_var}")
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:  # 3D Surface
                        z_data = st.session_state.contingency_table['counts'].values
                        x_data = list(st.session_state.contingency_table['counts'].columns)
                        y_data = list(st.session_state.contingency_table['counts'].index)
                        
                        fig = go.Figure(data=[go.Surface(z=z_data, x=x_data, y=y_data)])
                        fig.update_layout(
                            title=f"3D Surface: {st.session_state.selected_row_var} vs {st.session_state.selected_col_var}",
                            scene=dict(
                                xaxis_title=st.session_state.selected_col_var,
                                yaxis_title=st.session_state.selected_row_var,
                                zaxis_title='Count'
                            ),
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("Statistical Analysis")
                    
                    if 'statistics' in st.session_state.contingency_table:
                        stats = st.session_state.contingency_table['statistics']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Chi-Square Statistic", f"{stats['chi2']:.4f}")
                            st.metric("Degrees of Freedom", stats['dof'])
                        
                        with col2:
                            st.metric("P-value", f"{stats['p_value']:.6f}")
                            significance = "✅ Significant" if stats['p_value'] < 0.05 else "❌ Not Significant"
                            st.metric("Significance (α=0.05)", significance)
                        
                        with col3:
                            st.metric("Cramér's V", f"{stats['cramers_v']:.4f}")
                            st.metric("Expected Min Count", f"{stats['expected_freq'].min():.2f}")
                        
                        # Interpretation
                        st.markdown("---")
                        st.markdown("**Interpretation:**")
                        
                        if stats['p_value'] < 0.05:
                            st.success(f"""
                            The chi-square test indicates a **statistically significant** association between 
                            {st.session_state.selected_row_var} and {st.session_state.selected_col_var} 
                            (p = {stats['p_value']:.6f} < 0.05).
                            
                            The strength of association (Cramér's V = {stats['cramers_v']:.4f}) is 
                            {self.interpret_cramers_v(stats['cramers_v'])}.
                            """)
                        else:
                            st.info(f"""
                            The chi-square test does **not** indicate a statistically significant association between 
                            {st.session_state.selected_row_var} and {st.session_state.selected_col_var} 
                            (p = {stats['p_value']:.6f} > 0.05).
                            """)
                        
                        # Expected frequencies
                        with st.expander("View Expected Frequencies"):
                            st.dataframe(pd.DataFrame(stats['expected_freq'],
                                                     index=st.session_state.contingency_table['counts'].index,
                                                     columns=st.session_state.contingency_table['counts'].columns))
                
                with tab4:
                    st.subheader("Export Options")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        export_format = st.selectbox(
                            "Export Format",
                            ["CSV", "Excel", "JSON", "LaTeX", "Markdown"]
                        )
                    
                    with col2:
                        include_stats = st.checkbox("Include Statistics", value=True)
                    
                    if st.button("Prepare Export", type="primary"):
                        export_data = self.prepare_export(export_format, include_stats)
                        
                        if export_format == "CSV":
                            st.download_button(
                                label="Download CSV",
                                data=export_data,
                                file_name="contingency_table.csv",
                                mime="text/csv"
                            )
                        elif export_format == "Excel":
                            st.download_button(
                                label="Download Excel",
                                data=export_data,
                                file_name="contingency_table.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        elif export_format == "JSON":
                            st.download_button(
                                label="Download JSON",
                                data=export_data,
                                file_name="contingency_table.json",
                                mime="application/json"
                            )
                        elif export_format == "LaTeX":
                            st.code(export_data, language="latex")
                            st.download_button(
                                label="Download LaTeX",
                                data=export_data,
                                file_name="contingency_table.tex",
                                mime="text/plain"
                            )
                        elif export_format == "Markdown":
                            st.markdown(export_data)
                            st.download_button(
                                label="Download Markdown",
                                data=export_data,
                                file_name="contingency_table.md",
                                mime="text/plain"
                            )
    
    def generate_contingency_table(self, normalize: str, show_margins: bool):
        """Generate the contingency table"""
        try:
            result = self.contingency_generator.generate(
                st.session_state.data,
                st.session_state.selected_row_var,
                st.session_state.selected_col_var,
                normalize=normalize.lower() if normalize != "None" else None,
                margins=show_margins
            )
            
            st.session_state.contingency_table = result
            st.success("✅ Contingency table generated successfully!")
            
        except Exception as e:
            st.error(f"Error generating contingency table: {str(e)}")
    
    def interpret_cramers_v(self, v: float) -> str:
        """Interpret Cramér's V value"""
        if v < 0.1:
            return "negligible"
        elif v < 0.3:
            return "weak"
        elif v < 0.5:
            return "moderate"
        else:
            return "strong"
    
    def prepare_export(self, format: str, include_stats: bool):
        """Prepare data for export"""
        table = st.session_state.contingency_table['counts']
        
        if format == "CSV":
            buffer = StringIO()
            table.to_csv(buffer)
            if include_stats and 'statistics' in st.session_state.contingency_table:
                buffer.write("\n\nStatistics:\n")
                stats = st.session_state.contingency_table['statistics']
                buffer.write(f"Chi-square: {stats['chi2']}\n")
                buffer.write(f"P-value: {stats['p_value']}\n")
                buffer.write(f"Cramér's V: {stats['cramers_v']}\n")
            return buffer.getvalue()
        
        elif format == "Excel":
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                table.to_excel(writer, sheet_name='Contingency Table')
                if include_stats and 'statistics' in st.session_state.contingency_table:
                    stats_df = pd.DataFrame([st.session_state.contingency_table['statistics']])
                    stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            return buffer.getvalue()
        
        elif format == "JSON":
            export_dict = {
                'contingency_table': table.to_dict(),
                'row_variable': st.session_state.selected_row_var,
                'column_variable': st.session_state.selected_col_var
            }
            if include_stats and 'statistics' in st.session_state.contingency_table:
                export_dict['statistics'] = st.session_state.contingency_table['statistics']
            return json.dumps(export_dict, indent=2, default=str)
        
        elif format == "LaTeX":
            latex = table.to_latex()
            if include_stats and 'statistics' in st.session_state.contingency_table:
                stats = st.session_state.contingency_table['statistics']
                latex += f"\n\n% Statistics\n"
                latex += f"% Chi-square: {stats['chi2']:.4f}\n"
                latex += f"% P-value: {stats['p_value']:.6f}\n"
                latex += f"% Cramér's V: {stats['cramers_v']:.4f}\n"
            return latex
        
        elif format == "Markdown":
            markdown = table.to_markdown()
            if include_stats and 'statistics' in st.session_state.contingency_table:
                stats = st.session_state.contingency_table['statistics']
                markdown += f"\n\n### Statistics\n"
                markdown += f"- **Chi-square**: {stats['chi2']:.4f}\n"
                markdown += f"- **P-value**: {stats['p_value']:.6f}\n"
                markdown += f"- **Cramér's V**: {stats['cramers_v']:.4f}\n"
            return markdown
    
    def run(self):
        """Run the main application"""
        self.render_header()
        self.render_sidebar()
        
        # Main content area
        self.render_data_import()
        
        if st.session_state.data is not None:
            self.render_data_preview()
            self.render_variable_selection()
            self.render_contingency_table()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
        CONFIRM Data Converter v1.0 | Built with Streamlit
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main entry point"""
    app = ConfirmDataConverter()
    app.run()

if __name__ == "__main__":
    main()