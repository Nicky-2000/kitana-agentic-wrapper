import pandas as pd
import random
# Faker is no longer needed
# from faker import Faker
# time module is no longer needed
# import time

def generate_emp_data():
    # --- Configuration ---
    num_employees = 1000 # Number of employees to generate
    employee_id_start = 101
    email_id_start = 5001 # Starting number for unique email IDs
    phone_min = 2000000000 # Smallest 10-digit number (avoiding 0/1 start)
    phone_max = 9999999999 # Largest 10-digit number
    # Use a fixed seed for reproducibility if needed
    # random.seed(42)

    # --- Define Performance/Review Correlation (Numeric Reviews 0-5) ---
    correlation_map = {
        "Excellent": {
            "score_range": (90, 100),
            "review_range": (4.5, 5.0)  # Numeric range for reviews
        },
        "Good": {
            "score_range": (75, 89),
            "review_range": (3.5, 4.4)
        },
        "Average": {
            "score_range": (60, 74),
            "review_range": (2.5, 3.4)
        },
        "Needs Improvement": {
            "score_range": (40, 59),
            "review_range": (1.0, 2.4)
        }
    }

    # --- Data Generation ---
    employee_data = []
    generated_phones = set() # To ensure phone number uniqueness
    # generated_emails set is no longer needed

    # Email domain variables are no longer needed
    # fixed_year = 2025 # Based on previous context if needed elsewhere
    # email_domain = f'example-{fixed_year}.com'

    print(f"Generating {num_employees} employee records...")

    for i in range(num_employees):
        employee_id = employee_id_start + i
        email_id = email_id_start + i # Simple sequential email ID

        # Generate unique *numeric* phone number (10 digits)
        while True:
            # Generate a random 10-digit integer
            phone_no = random.randint(phone_min, phone_max)
            if phone_no not in generated_phones:
                 generated_phones.add(phone_no)
                 break
            # Safeguard for very high numbers of employees (collision check)
            if len(generated_phones) >= num_employees: # Stop if we generated as many unique phones as needed
                 if phone_no in generated_phones: # Check if the last generated one is the problem
                     continue # try generating another one
                 else: # Should not happen logically, but as a safeguard
                     generated_phones.add(phone_no)
                     break
            # Add a more robust warning/check if generating many more numbers than employees
            if len(generated_phones) > num_employees * 1.5 and i < num_employees -1 :
                 print(f"Warning: High collision rate generating unique phone numbers at record {i+1}.")
                 # Consider breaking or raising error if it becomes impossible

        # --- Email generation block removed ---

        # Select performance category
        # Example weights: weights = [0.2, 0.4, 0.3, 0.1] # Sum must be 1
        # category_name = random.choices(list(correlation_map.keys()), weights=weights, k=1)[0]
        category_name = random.choice(list(correlation_map.keys())) # Using simple choice
        category_data = correlation_map[category_name]

        # Generate performance score within the category range
        min_score, max_score = category_data["score_range"]
        performance_score = random.randint(min_score, max_score)

        # Generate *numeric* review score corresponding to the category
        min_review, max_review = category_data["review_range"]
        review_score = round(random.uniform(min_review, max_review), 1)

        # Store data for this employee
        employee_data.append({
            "employee_id": employee_id,
            "performance": performance_score,
            "phone_no": phone_no,          # Integer phone number
            "email_id": email_id,          # Integer email ID
            "review_score": review_score   # Numeric review score (float)
        })

    print(f"Data generation complete. Generated {len(employee_data)} records.")

    # --- Create Pandas DataFrames ---
    if employee_data:
        master_df = pd.DataFrame(employee_data)

        # Table 1: employee_performance
        df_performance = master_df[['employee_id', 'performance']].copy()
        df_performance['employee_id'] = df_performance['employee_id'].astype(int)
        df_performance['performance'] = df_performance['performance'].astype(int)


        # Table 2: employee_phone
        df_phone = master_df[['employee_id', 'phone_no']].copy()
        # Ensure types are integer
        df_phone['employee_id'] = df_phone['employee_id'].astype(int)
        df_phone['phone_no'] = df_phone['phone_no'].astype(int)


        # Table 3: employee_email_id (Changed name and content)
        df_email_id = master_df[['employee_id', 'email_id']].copy()
         # Ensure types are integer
        df_email_id['employee_id'] = df_email_id['employee_id'].astype(int)
        df_email_id['email_id'] = df_email_id['email_id'].astype(int)

        # Table 4: phone_reviews
        df_reviews = master_df[['phone_no', 'review_score']].copy()
        df_reviews.rename(columns={'review_score': 'reviews'}, inplace=True)
        # Ensure phone_no is integer and reviews is float
        df_reviews['phone_no'] = df_reviews['phone_no'].astype(int)
        df_reviews['reviews'] = df_reviews['reviews'].astype(float)


        # --- Display Results (Optional) ---
        # Limit display if num_employees is large
        display_limit = 20
        print(f"\n--- Generated DataFrames (Displaying first {display_limit} rows) ---")

        print("\nTable 1: employee_performance")
        print(df_performance.head(display_limit).to_string(index=False))

        print("\nTable 2: employee_phone")
        print(df_phone.head(display_limit).to_string(index=False)) # phone_no is now numeric

        print("\nTable 3: employee_email_id") # Updated table name
        print(df_email_id.head(display_limit).to_string(index=False)) # Shows email_id (numeric)

        print("\nTable 4: phone_reviews")
        print(df_reviews.head(display_limit).to_string(index=False)) # phone_no is numeric, reviews are numeric


        # --- Create directory if it doesn't exist ---
        # Requires 'import os' at the top if you uncomment this
        # import os
        # output_dir = 'src/multi-hopp-data'
        # os.makedirs(output_dir, exist_ok=True)

        # --- Save to CSV (Using specified paths) ---
        # It's safer to join paths like this:
        # performance_path = os.path.join(output_dir, 'employee_performance.csv')
        # But using the direct string as provided:
        df_performance.to_csv('src/multi-hopp-data/employee_performance.csv', index=False)
        df_phone.to_csv('src/multi-hopp-data/employee_phone_numeric.csv', index=False) # Indicate numeric phone
        df_email_id.to_csv('src/multi-hopp-data/employee_email_id.csv', index=False)   # New filename
        df_reviews.to_csv('src/multi-hopp-data/phone_reviews_numeric.csv', index=False) # Indicate numeric reviews/phone
        print(f"\nDataFrames saved to CSV files in 'src/multi-hopp-data/' directory.")

    else:
        print("\nNo data was generated.")

def copy_to_datalake():
    import shutil
    import os

    # Define the source and destination directories
    source_dir = 'src/multi-hopp-data'
    destination_dir = 'data/datalake'

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Copy files from source to destination
    for filename in os.listdir(source_dir):
        full_file_name = os.path.join(source_dir, filename)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, destination_dir)
            print(f"Copied {filename} to {destination_dir}")

# --- Execute the function ---
generate_emp_data()
copy_to_datalake()