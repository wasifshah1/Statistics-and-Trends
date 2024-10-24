# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import zipfile
import os

def load_and_extract_data(zip_file_path, extract_path):
    """
    Load and extract data from a zip file.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    books_data_path = os.path.join(extract_path, 'books_of_the_decade.csv')
    reviews_data_path = os.path.join(extract_path, 'user_reviews_dataset.csv')
    
    books_df = pd.read_csv(books_data_path)
    reviews_df = pd.read_csv(reviews_data_path)
    
    return books_df, reviews_df

def data_cleaning_and_preprocessing(books_df, reviews_df):
    """
    Perform data cleaning, outlier treatment, and normalization.
    """
    books_df['Rating'] = pd.to_numeric(books_df['Rating'], errors='coerce')
    books_df['Number of Votes'] = pd.to_numeric(books_df['Number of Votes'], errors='coerce')
    books_df.fillna(method='ffill', inplace=True)
    books_df['Rating'].fillna(books_df['Rating'].mean(), inplace=True)
    reviews_df.fillna(method='ffill', inplace=True)

    # Exclude 'Score' from normalization to preserve its original variance
    numerical_columns_books = books_df.select_dtypes(include=['float64', 'int64']).columns.drop('Score')
    numerical_columns_reviews = reviews_df.select_dtypes(include=['float64', 'int64']).columns

    for column in numerical_columns_books:
        lower_bound = books_df[column].quantile(0.01)
        upper_bound = books_df[column].quantile(0.99)
        books_df[column] = books_df[column].clip(lower=lower_bound, upper=upper_bound)

    for column in numerical_columns_reviews:
        lower_bound = reviews_df[column].quantile(0.01)
        upper_bound = reviews_df[column].quantile(0.99)
        reviews_df[column] = reviews_df[column].clip(lower=lower_bound, upper=upper_bound)

    scaler = MinMaxScaler()
    books_df[numerical_columns_books] = scaler.fit_transform(books_df[numerical_columns_books])
    reviews_df[numerical_columns_reviews] = scaler.fit_transform(reviews_df[numerical_columns_reviews])

    return books_df, reviews_df

def plot_top_books_by_votes(books_df):
    """
    Plot top 10 books by number of votes.
    """
    plt.figure(figsize=(12, 8))
    top_10_books_votes = books_df.nlargest(10, 'Number of Votes')
    sns.barplot(
        x='Number of Votes',
        y='Book Name',
        data=top_10_books_votes,
        palette='viridis'
    )
    plt.title('Top 10 Books by Number of Votes', fontsize=20, fontweight='bold')
    plt.xlabel('Number of Votes (Normalized)', fontsize=16, fontweight='bold')
    plt.ylabel('Book Name', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_author_popularity_pie_chart(books_df):
    """
    Plot pie chart of top 10 authors by popularity.
    """
    author_counts = books_df['Author'].value_counts().nlargest(10)
    plt.figure(figsize=(12, 12))
    plt.pie(
        author_counts.values,
        labels=author_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette('Set3'),
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )
    plt.title('Top 10 Authors by Popularity in Books Dataset', fontsize=18, fontweight='bold')
    plt.axis('equal')
    plt.legend(title='Author', loc='upper right', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_score_trends_line_chart(books_df):
    """
    Plot line chart showing score trends for top 5 books by score.
    """
    # Ensure 'Score' exists and is numeric
    if 'Score' not in books_df.columns or not pd.api.types.is_numeric_dtype(books_df['Score']):
        print("The 'Score' column is missing or not numeric.")
        return

    # Select the top 5 books based on their score
    top_5_books = books_df.nlargest(5, 'Score')

    # Check if there is enough variance in the scores
    if top_5_books['Score'].nunique() == 1:
        print("All top 5 books have the same score. Line chart might not show variation.")
    
    # Plotting a line chart for the top 5 books by score
    plt.figure(figsize=(14, 8))
    plt.plot(top_5_books['Book Name'], top_5_books['Score'], marker='o', linestyle='-', color='b', linewidth=2, markersize=10)

    # Adding details to the chart for better readability
    plt.xlabel('Book Title', fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=16, fontweight='bold')
    plt.title('Score Trends for Top 5 Books', fontsize=20, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show plot with tight layout
    plt.tight_layout()
    plt.show()

def plot_books_correlation_heatmap(books_df):
    """
    Plot heatmap of correlations in books dataset.
    """
    plt.figure(figsize=(10, 6))
    correlation_matrix_books = books_df.corr(numeric_only=True)
    sns.heatmap(correlation_matrix_books, annot=True, cmap='coolwarm', linewidths=0.5, annot_kws={"fontsize":12, "fontweight":"bold"})
    plt.title('Heatmap of Correlations - Books Dataset', fontsize=18, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def main():
    # Define paths
    zip_file_path = 'Best Book Of Decade 2020.zip'
    extract_path = 'best_book_decade/'

    # Load and extract data
    books_df, reviews_df = load_and_extract_data(zip_file_path, extract_path)

    # Initial Data Inspection
    print("Books Dataset Columns:", books_df.columns)
    print("Reviews Dataset Columns:", reviews_df.columns)
    print("Books Dataset Preview:\n", books_df.head())
    print("Reviews Dataset Preview:\n", reviews_df.head())

    # Data Cleaning and Preprocessing
    books_df, reviews_df = data_cleaning_and_preprocessing(books_df, reviews_df)

    # Statistical Analysis
    print("Summary statistics for Books Dataset:\n", books_df.describe())
    print("Correlation matrix for Books Dataset (numerical columns only):\n", books_df.corr(numeric_only=True))
    print("Summary statistics for Reviews Dataset:\n", reviews_df.describe())
    print("Correlation matrix for Reviews Dataset (numerical columns only):\n", reviews_df.corr(numeric_only=True))

    # Plotting
    plot_top_books_by_votes(books_df)
    plot_author_popularity_pie_chart(books_df)
    plot_books_correlation_heatmap(books_df)
    plot_score_trends_line_chart(books_df)

if __name__ == "__main__":
    main()