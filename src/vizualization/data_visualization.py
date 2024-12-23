import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import plotly.express as px

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data = pd.read_csv('processed_data/cleaned_data.csv')

viz_dir = "visualization_files"

if os.path.exists(viz_dir):
    print(f"Directory {viz_dir} already exists at {os.getcwd()}")
else:
    os.mkdir(viz_dir)
    print(f"Directory Created with name {viz_dir} at {os.getcwd()}")

def save_plot(figure, file_name, viz_dir):
    output_file = os.path.join(viz_dir, file_name)
    figure.savefig(output_file)
    print(f"Visualization saved to {output_file}")
    
plt.figure(figsize=(10, 6))
sns.histplot(data['Units_Sold'], kde=True, color='g', bins=12)
plt.grid(True, alpha=0.12, color='red')
plt.title('Distribution of Units Sold')
save_plot(plt, "Distribution_of_Units_Sold.png", viz_dir)
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Discount'], kde=True, color='r', bins=12)
plt.title('Distribution of Discount')
save_plot(plt, "Distribution_of_Discount.png", viz_dir)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='Product_Category', data=data, palette='Set1')
plt.title('Count of Product Categories Sold')
plt.xlabel('Product Category')
plt.ylabel('Frequency')
save_plot(plt, "Count_of_Product_Categories_Sold.png", viz_dir)
plt.show()

sales_wise_cat = data.groupby('Product_Category')['Units_Sold'].sum().reset_index().sort_values(by='Units_Sold', ascending=False)
plt.figure(figsize=(10, 6))
plt.pie(sales_wise_cat['Units_Sold'], labels=sales_wise_cat['Product_Category'], autopct="%1.2f%%", shadow=True)
plt.title('Total Sales of Products by Category')
save_plot(plt, "Total_Sales_of_Products_by_Category.png", viz_dir)
plt.show()

numerical_df = data.select_dtypes('number')
filtered_data = numerical_df.drop(columns=['Month', 'year', 'Day'])
correlation = filtered_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation of Numerical Values')
save_plot(plt, "Correlation_of_Numerical_Values.png", viz_dir)
plt.show()

year_wise_cat_sales = data.groupby(['year', 'Product_Category'])['Units_Sold'].sum().reset_index()
plt.figure(figsize=(15, 6))
sns.barplot(x='year', y='Units_Sold', hue='Product_Category', data=year_wise_cat_sales, palette='Set2')
plt.title('Yearly Sales by Product Category')
save_plot(plt, "Yearly_Sales_by_Product_Category.png", viz_dir)
plt.show()

plt.figure(figsize=(14, 8))
sns.boxenplot(x='Product_Category', y='Price', hue='Customer_Segment', data=data, palette='coolwarm')
plt.title('Price Distribution by Product Category and Customer Segment')
plt.xlabel('Product Category')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(plt, "Price_Distribution_by_Product_Category_and_Customer_Segment.png", viz_dir)
plt.show()

month_wise_sales = data.groupby(['year', 'Month'])['Units_Sold'].sum().reset_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x='Month', y='Units_Sold', hue='year', data=month_wise_sales, palette='Set1')
plt.title('Monthly Sales from 2023 to 2025')
plt.grid(True, alpha=0.3)
plt.xticks(month_wise_sales['Month'].unique())
save_plot(plt, "Monthly_Sales_from_2023_to_2025.png", viz_dir)
plt.show()

category_discount = data.groupby('Product_Category', as_index=False)['Discount'].mean()
plt.figure(figsize=(8, 6))
sns.heatmap(category_discount.set_index('Product_Category').sort_values('Discount', ascending=False), annot=True, cmap='coolwarm', fmt=".2f", linewidths=2)
plt.title('Average Discounts by Product Category')
plt.xlabel('Average Discount')
plt.ylabel('Product Category')
save_plot(plt, "Average_Discounts_by_Product_Category.png", viz_dir)
plt.show()

category_marketing = data.groupby('Product_Category')['Marketing_Spend'].sum().reset_index()
plt.figure(figsize=(10, 6))
squarify.plot(sizes=category_marketing['Marketing_Spend'], label=category_marketing['Product_Category'], color=sns.color_palette('Set3'), alpha=0.8)
plt.title('Marketing Spend Distribution Across Categories')
save_plot(plt, "Marketing_Spend_Distribution.png", viz_dir)
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Product_Category', y='Price', data=data, palette='Set1')
plt.title('Comparison of Unit Price Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Unit Price')
save_plot(plt, "Comparison of Unit Price Across Product Categories.png", viz_dir)
plt.show()

product_metrics=data.groupby('Product_Category').agg({'Units_Sold':'sum','Price':'mean','Discount':'mean',
                                                      'Marketing_Spend': 'sum','Revenue': 'sum','ROI': 'mean'}).reset_index()
product_metrics=product_metrics.sort_values('ROI',ascending=True)
plt.figure(figsize=(12,6))
plt.hlines(y=product_metrics['Product_Category'], xmin=0, xmax=product_metrics['ROI'], color='green',)
plt.plot(product_metrics['ROI'], product_metrics['Product_Category'], "*", markersize=10, color='red')
plt.title('Product Category ROI Analysis', fontsize=15, pad=20)
plt.xlabel('Return on Investment (ROI)', fontsize=12)
plt.ylabel('Product Category', fontsize=12)
for i, v in enumerate(product_metrics['ROI']):
    plt.text(v, i, f' {v:1.1f}', va='center', fontsize=14)
plt.grid(True,alpha=0.1,color='navy')
plt.tight_layout()
save_plot(plt, "Product Category ROI Analysis.png", viz_dir)
plt.show()

fig = px.sunburst(data,path=['Product_Category', 'Customer_Segment'],
                  values='Revenue',title='Revenue Distribution by Product Category and Customer Segment')
save_plot(plt, "Revenue Distribution by Product Category and Customer Segment.png", viz_dir)
fig.show()

