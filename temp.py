import pandas as pd
emp = pd.read_csv("/Users/kaushaldamania/llms/kitana-agentic-wrapper/src/multi-hopp-data/employee_performance.csv")
phone = pd.read_csv("/Users/kaushaldamania/llms/kitana-agentic-wrapper/data/datalake/employee_phone_numeric.csv")
review = pd.read_csv("/Users/kaushaldamania/llms/kitana-agentic-wrapper/data/datalake/phone_reviews_numeric.csv")

# Merge the DataFrames on 'phone_no' and 'employee_id'
merged_df = pd.merge(phone, review, on='phone_no', how='inner')
merged_df = pd.merge(merged_df, emp, on='employee_id', how='inner')
merged_df["performance"]

review['reviews'] = merged_df['performance']

review.to_csv("/Users/kaushaldamania/llms/kitana-agentic-wrapper/data/datalake/phone_reviews_numeric.csv", index=False)

print(review.head())
