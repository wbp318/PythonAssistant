import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import io
from PIL import Image
import tabula
import PyPDF2

def read_file(file_path):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.pdf'):
        # Try to read tables from PDF
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        if tables:
            return pd.concat(tables, ignore_index=True)
        else:
            # If no tables found, try to extract text
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            return pd.DataFrame({'text': [text]})
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .pdf")

def create_infographic(file_path):
    # Read the file
    df = read_file(file_path)
    
    # Create a large figure for the infographic
    fig = plt.figure(figsize=(20, 30))
    
    # 1. Basic Statistics
    ax1 = fig.add_subplot(421)
    numeric_df = df.select_dtypes(include=[np.number])
    stats = numeric_df.describe().transpose()
    ax1.axis('off')
    ax1.text(0.1, 0.9, "Basic Statistics", fontsize=16, fontweight='bold')
    ax1.table(cellText=stats.values.round(2), 
              rowLabels=stats.index, 
              colLabels=stats.columns, 
              cellLoc='center', 
              loc='center')
    
    # 2. Correlation Heatmap
    ax2 = fig.add_subplot(422)
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), ax=ax2, cmap='coolwarm', annot=True, fmt='.2f')
        ax2.set_title("Correlation Heatmap")
    else:
        ax2.text(0.5, 0.5, "No numeric data for correlation", ha='center', va='center')
    
    # 3. Distribution of a key numeric variable
    ax3 = fig.add_subplot(423)
    if not numeric_df.empty:
        key_numeric = numeric_df.columns[0]  # Choose the first numeric column
        sns.histplot(df[key_numeric], kde=True, ax=ax3)
        ax3.set_title(f"Distribution of {key_numeric}")
    else:
        ax3.text(0.5, 0.5, "No numeric data for distribution", ha='center', va='center')
    
    # 4. Top 10 categories of a categorical variable
    ax4 = fig.add_subplot(424)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        key_categorical = categorical_columns[0]  # Choose the first categorical column
        df[key_categorical].value_counts().nlargest(10).plot(kind='bar', ax=ax4)
        ax4.set_title(f"Top 10 {key_categorical}")
        ax4.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
    else:
        ax4.text(0.5, 0.5, "No categorical data for top categories", ha='center', va='center')
    
    # 5. Time series plot (if date column exists)
    ax5 = fig.add_subplot(425)
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0 and not numeric_df.empty:
        date_col = date_columns[0]
        key_numeric = numeric_df.columns[0]
        df.set_index(date_col)[key_numeric].resample('M').sum().plot(ax=ax5)
        ax5.set_title(f"Monthly {key_numeric} Trend")
    else:
        ax5.text(0.5, 0.5, "No date column or numeric data found", ha='center', va='center')
    
    # 6. Scatter plot of two key variables
    ax6 = fig.add_subplot(426)
    if len(numeric_df.columns) >= 2:
        x_col, y_col = numeric_df.columns[:2]
        ax6.scatter(df[x_col], df[y_col])
        ax6.set_xlabel(x_col)
        ax6.set_ylabel(y_col)
        ax6.set_title(f"{x_col} vs {y_col}")
    else:
        ax6.text(0.5, 0.5, "Insufficient numeric data for scatter plot", ha='center', va='center')
    
    # 7. Pie chart of a categorical variable
    ax7 = fig.add_subplot(427)
    if len(categorical_columns) > 0:
        df[key_categorical].value_counts().plot(kind='pie', ax=ax7, autopct='%1.1f%%')
        ax7.set_title(f"Distribution of {key_categorical}")
    else:
        ax7.text(0.5, 0.5, "No categorical data for pie chart", ha='center', va='center')
    
    # 8. Word Cloud (if text data is available)
    ax8 = fig.add_subplot(428)
    if len(categorical_columns) > 0:
        text = ' '.join(df[key_categorical].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax8.imshow(wordcloud, interpolation='bilinear')
        ax8.axis('off')
        ax8.set_title(f"Word Cloud of {key_categorical}")
    elif 'text' in df.columns:
        text = ' '.join(df['text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax8.imshow(wordcloud, interpolation='bilinear')
        ax8.axis('off')
        ax8.set_title("Word Cloud of PDF Text")
    else:
        ax8.text(0.5, 0.5, "No text data for word cloud", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('infographic.png', dpi=300, bbox_inches='tight')
    print("Infographic saved as 'infographic.png'")

    # Display some basic information about the dataset
    print("\nDataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())

# Usage
file_path = 'FloridaBarBudget20242025.pdf'  # Replace with your file name (can be .xlsx or .pdf)
create_infographic(file_path)

if not numeric_df.empty and not numeric_df.isnull().all().all():
    sns.heatmap(numeric_df.corr(), ax=ax2, cmap='coolwarm', annot=True, fmt='.2f')
    ax2.set_title("Correlation Heatmap")
else:
    ax2.text(0.5, 0.5, "Insufficient data for correlation heatmap", ha='center', va='center')