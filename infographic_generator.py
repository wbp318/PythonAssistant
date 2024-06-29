import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import io
from PIL import Image

def create_infographic(excel_file):
    # Read the Excel file
    df = pd.read_excel(excel_file)
    
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
    sns.heatmap(numeric_df.corr(), ax=ax2, cmap='coolwarm', annot=True, fmt='.2f')
    ax2.set_title("Correlation Heatmap")
    
    # 3. Distribution of a key numeric variable
    ax3 = fig.add_subplot(423)
    key_numeric = numeric_df.columns[0]  # Choose the first numeric column
    sns.histplot(df[key_numeric], kde=True, ax=ax3)
    ax3.set_title(f"Distribution of {key_numeric}")
    
    # 4. Top 10 categories of a categorical variable
    ax4 = fig.add_subplot(424)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        key_categorical = categorical_columns[0]  # Choose the first categorical column
        df[key_categorical].value_counts().nlargest(10).plot(kind='bar', ax=ax4)
        ax4.set_title(f"Top 10 {key_categorical}")
        ax4.set_xlabel("")
        plt.xticks(rotation=45, ha='right')
    
    # 5. Time series plot (if date column exists)
    ax5 = fig.add_subplot(425)
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0:
        date_col = date_columns[0]
        df.set_index(date_col)[key_numeric].resample('M').sum().plot(ax=ax5)
        ax5.set_title(f"Monthly {key_numeric} Trend")
    else:
        ax5.text(0.5, 0.5, "No date column found", ha='center', va='center')
    
    # 6. Scatter plot of two key variables
    ax6 = fig.add_subplot(426)
    if len(numeric_df.columns) >= 2:
        x_col, y_col = numeric_df.columns[:2]
        ax6.scatter(df[x_col], df[y_col])
        ax6.set_xlabel(x_col)
        ax6.set_ylabel(y_col)
        ax6.set_title(f"{x_col} vs {y_col}")
    
    # 7. Pie chart of a categorical variable
    ax7 = fig.add_subplot(427)
    if len(categorical_columns) > 0:
        df[key_categorical].value_counts().plot(kind='pie', ax=ax7, autopct='%1.1f%%')
        ax7.set_title(f"Distribution of {key_categorical}")
    
    # 8. Word Cloud (if text data is available)
    ax8 = fig.add_subplot(428)
    if len(categorical_columns) > 0:
        text = ' '.join(df[key_categorical].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax8.imshow(wordcloud, interpolation='bilinear')
        ax8.axis('off')
        ax8.set_title(f"Word Cloud of {key_categorical}")
    
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
excel_file = 'fsi2023.xlsx'  # Replace with your Excel file name
create_infographic(excel_file)