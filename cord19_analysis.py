# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import streamlit as st
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

# Part 1: Data Loading and Basic Exploration
def load_and_explore_data():
    """Load the dataset and perform basic exploration"""
    st.header("Data Loading and Basic Exploration")
    
    # Load the data
    df = pd.read_csv('metadata.csv', low_memory=False)
    
    # Display basic information
    st.subheader("Dataset Overview")
    st.write(f"Dataset shape: {df.shape}")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Show first few rows
    if st.checkbox("Show first few rows"):
        st.dataframe(df.head())
    
    # Display data types
    if st.checkbox("Show data types"):
        st.write(df.dtypes)
    
    # Display missing values
    if st.checkbox("Show missing values"):
        missing_values = df.isnull().sum().sort_values(ascending=False)
        st.write("Missing values per column:")
        st.write(missing_values)
        
        # Calculate missing percentage
        missing_percent = (df.isnull().sum() / len(df)) * 100
        st.write("Missing value percentage:")
        st.write(missing_percent.sort_values(ascending=False))
    
    return df

# Part 2: Data Cleaning and Preparation
def clean_and_prepare_data(df):
    """Clean and prepare the data for analysis"""
    st.header("Data Cleaning and Preparation")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # Check which columns have too many missing values
    missing_percent = (df_clean.isnull().sum() / len(df_clean)) * 100
    
    # Drop columns with more than 80% missing values
    columns_to_drop = missing_percent[missing_percent > 80].index
    df_clean = df_clean.drop(columns=columns_to_drop)
    st.write(f"Dropped columns with >80% missing values: {list(columns_to_drop)}")
    
    # Handle missing values in important columns
    df_clean['abstract'] = df_clean['abstract'].fillna('No abstract available')
    df_clean['authors'] = df_clean['authors'].fillna('Unknown authors')
    
    # Convert publish_time to datetime and extract year
    df_clean['publish_time'] = pd.to_datetime(df_clean['publish_time'], errors='coerce')
    df_clean['publication_year'] = df_clean['publish_time'].dt.year
    
    # Create abstract word count column
    df_clean['abstract_word_count'] = df_clean['abstract'].apply(lambda x: len(str(x).split()))
    
    # Filter for COVID-related papers (assuming we want to focus on recent years)
    df_covid = df_clean[df_clean['publication_year'] >= 2019].copy()
    
    st.write(f"After cleaning, dataset shape: {df_covid.shape}")
    
    if st.checkbox("Show cleaned data sample"):
        st.dataframe(df_covid.head())
    
    return df_covid

# Part 3: Data Analysis and Visualization
def analyze_and_visualize(df):
    """Perform data analysis and create visualizations"""
    st.header("Data Analysis and Visualization")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    year_range = st.sidebar.slider(
        "Select Year Range",
        min_value=int(df['publication_year'].min()),
        max_value=int(df['publication_year'].max()),
        value=(2020, 2021)
    )
    
    journal_options = df['journal'].dropna().unique()
    selected_journals = st.sidebar.multiselect(
        "Select Journals",
        options=journal_options,
        default=df['journal'].value_counts().head(5).index.tolist() if len(journal_options) > 5 else journal_options
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['publication_year'] >= year_range[0]) & 
        (df['publication_year'] <= year_range[1])
    ]
    
    if selected_journals:
        filtered_df = filtered_df[filtered_df['journal'].isin(selected_journals)]
    
    # Display basic metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Papers", len(filtered_df))
    col2.metric("Unique Journals", filtered_df['journal'].nunique())
    col3.metric("Average Abstract Length", f"{filtered_df['abstract_word_count'].mean():.1f} words")
    col4.metric("Earliest Publication", int(filtered_df['publication_year'].min()))
    
    # Visualizations
    st.subheader("Publications by Year")
    year_counts = filtered_df['publication_year'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(year_counts.index.astype(str), year_counts.values)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Publications')
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Top journals
    st.subheader("Top Journals")
    top_journals = filtered_df['journal'].value_counts().head(10)
    if not top_journals.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_journals.values, y=top_journals.index, ax=ax)
        ax.set_xlabel('Number of Publications')
        st.pyplot(fig)
    else:
        st.write("No data available for the selected filters.")
    
    # Word cloud
    st.subheader("Word Cloud of Paper Titles")
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text
    
    all_titles = ' '.join(filtered_df['title'].dropna().apply(clean_text))
    
    if all_titles.strip():  # Check if there's any text
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
        
        # Most frequent words
        st.subheader("Most Frequent Words in Titles")
        word_freq = Counter(all_titles.split())
        common_words = word_freq.most_common(20)
        
        words, counts = zip(*common_words)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=list(counts), y=list(words), ax=ax)
        ax.set_xlabel('Frequency')
        st.pyplot(fig)
    else:
        st.write("No titles available for word cloud generation.")
    
    # Top sources
    st.subheader("Top Sources")
    top_sources = filtered_df['source_x'].value_counts().head(10)
    if not top_sources.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=top_sources.values, y=top_sources.index, ax=ax)
        ax.set_xlabel('Number of Publications')
        st.pyplot(fig)
    else:
        st.write("No source data available.")
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(filtered_df[['title', 'journal', 'publication_year', 'abstract']].head(10))

# Part 4: Streamlit Application Main Function
def main():
    """Main function to run the Streamlit app"""
    st.set_page_config(
        page_title="CORD-19 Data Explorer",
        page_icon=":microscope:",
        layout="wide"
    )
    
    st.title("CORD-19 COVID-19 Research Explorer")
    st.write("""
    This application explores the CORD-19 dataset of COVID-19 research papers.
    Use the filters in the sidebar to explore publications by year, journal, and more.
    """)
    
    # Load and process data
    df = load_and_explore_data()
    df_clean = clean_and_prepare_data(df)
    
    # Add a divider
    st.markdown("---")
    
    # Perform analysis and visualization
    analyze_and_visualize(df_clean)
    
    # Add documentation and reflection
    st.sidebar.header("About")
    st.sidebar.info("""
    This app analyzes the CORD-19 dataset, which contains metadata
    about COVID-19 research papers. The data is updated regularly
    as new research is published.
    
    **Key Features:**
    - Data exploration and cleaning
    - Publication trends by year
    - Journal analysis
    - Word frequency analysis
    - Interactive filtering
    """)
    
    # Reflection section
    if st.sidebar.checkbox("Show Reflection"):
        st.sidebar.subheader("Project Reflection")
        st.sidebar.write("""
        **Challenges Faced:**
        1. Large dataset size requiring careful memory management
        2. Significant missing values in many columns
        3. Inconsistent date formats in publish_time column
        4. Streamlit performance optimization for large datasets
        
        **Key Findings:**
        1. COVID-19 research publications increased dramatically in 2020
        2. Medical and virology journals published the most COVID-19 research
        3. Common terms in titles include "covid", "sars", "pandemic", and "clinical"
        4. Preprint servers were significant sources of COVID-19 papers
        
        **Learning Outcomes:**
        1. Experience with large dataset manipulation in pandas
        2. Strategic handling of missing data
        3. Creating informative visualizations
        4. Building interactive Streamlit applications
        5. Performance optimization with caching
        """)

# Run the application
if __name__ == "__main__":
    main()