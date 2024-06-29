import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import io
from PIL import Image
import tabula
import PyPDF2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def read_file(file_path):
    if file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    elif file_path.endswith('.pdf'):
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        if tables:
            return pd.concat(tables, ignore_index=True)
        else:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            return pd.DataFrame({'text': [text]})
    else:
        raise ValueError("Unsupported file format. Please use .xlsx or .pdf")

def create_detailed_infographic(file_path):
    df = read_file(file_path)
    
    fig = plt.figure(figsize=(30, 40))
    
    # 1. Basic Statistics
    ax1 = fig.add_subplot(441)
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
    ax2 = fig.add_subplot(442)
    if not numeric_df.empty and not numeric_df.isnull().all().all():
        sns.heatmap(numeric_df.corr(), ax=ax2, cmap='coolwarm', annot=True, fmt='.2f', cbar=False)
        ax2.set_title("Correlation Heatmap")
    else:
        ax2.text(0.5, 0.5, "Insufficient data for correlation heatmap", ha='center', va='center')
    
    # 3. Distribution of key numeric variables
    for i, column in enumerate(numeric_df.columns[:6]):  # Plot for up to 6 numeric columns
        ax = fig.add_subplot(4, 4, i+3)
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel("")
    
    # 4. Top categories of categorical variables
    categorical_columns = df.select_dtypes(include=['object']).columns
    for i, column in enumerate(categorical_columns[:3]):  # Plot for up to 3 categorical columns
        ax = fig.add_subplot(4, 4, i+9)
        df[column].value_counts().nlargest(10).plot(kind='bar', ax=ax)
        ax.set_title(f"Top 10 {column}")
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # 5. Time series plot
    ax5 = fig.add_subplot(4, 4, 12)
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if len(date_columns) > 0 and not numeric_df.empty:
        date_col = date_columns[0]
        key_numeric = numeric_df.columns[0]
        df.set_index(date_col)[key_numeric].resample('M').sum().plot(ax=ax5)
        ax5.set_title(f"Monthly {key_numeric} Trend")
    else:
        ax5.text(0.5, 0.5, "No date column or numeric data found", ha='center', va='center')
    
    # 6. Scatter plot matrix
    ax6 = fig.add_subplot(443, projection='3d')
    if len(numeric_df.columns) >= 3:
        x, y, z = numeric_df.columns[:3]
        ax6.scatter(df[x], df[y], df[z])
        ax6.set_xlabel(x)
        ax6.set_ylabel(y)
        ax6.set_zlabel(z)
        ax6.set_title("3D Scatter Plot")
    else:
        ax6.text(0.5, 0.5, 0.5, "Insufficient numeric data for 3D scatter plot", ha='center', va='center')
    
    # 7. Pie chart of a categorical variable
    ax7 = fig.add_subplot(444)
    if len(categorical_columns) > 0:
        df[categorical_columns[0]].value_counts().plot(kind='pie', ax=ax7, autopct='%1.1f%%')
        ax7.set_title(f"Distribution of {categorical_columns[0]}")
    else:
        ax7.text(0.5, 0.5, "No categorical data for pie chart", ha='center', va='center')
    
    # 8. Word Cloud
    ax8 = fig.add_subplot(414)
    if len(categorical_columns) > 0:
        text = ' '.join(df[categorical_columns].astype(str).values.ravel('K'))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax8.imshow(wordcloud, interpolation='bilinear')
        ax8.axis('off')
        ax8.set_title("Word Cloud of Categorical Data")
    elif 'text' in df.columns:
        text = ' '.join(df['text'].astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        ax8.imshow(wordcloud, interpolation='bilinear')
        ax8.axis('off')
        ax8.set_title("Word Cloud of PDF Text")
    else:
        ax8.text(0.5, 0.5, "No text data for word cloud", ha='center', va='center')
    
    # 9. PCA plot
    ax9 = fig.add_subplot(447)
    if not numeric_df.empty:
        scaler = StandardScaler()
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaler.fit_transform(numeric_df.fillna(numeric_df.mean())))
        ax9.scatter(pca_result[:, 0], pca_result[:, 1])
        ax9.set_title("PCA of Numeric Data")
        ax9.set_xlabel("First Principal Component")
        ax9.set_ylabel("Second Principal Component")
    else:
        ax9.text(0.5, 0.5, "Insufficient numeric data for PCA", ha='center', va='center')
    
    # 10. Box plot of numeric variables
    ax10 = fig.add_subplot(448)
    if not numeric_df.empty:
        sns.boxplot(data=numeric_df, ax=ax10)
        ax10.set_title("Box Plot of Numeric Variables")
        plt.setp(ax10.get_xticklabels(), rotation=45, ha='right')
    else:
        ax10.text(0.5, 0.5, "No numeric data for box plot", ha='center', va='center')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('detailed_infographic.png', dpi=300, bbox_inches='tight')
    print("Detailed infographic saved as 'detailed_infographic.png'")

    # Display some basic information about the dataset
    print("\nDataset Information:")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())

# Usage
#file_path = 'EmployeeSampleData.xlsx'  # Replace with your file name (can be .xlsx or .pdf)
#create_detailed_infographic(file_path)