import pandas as pd
import os
from num2words import num2words

def perform_eda(file_path):
    """
    Perform exploratory data analysis on the given file.
    Includes preprocessing and conversion of numerical columns to words.
    """
    try:
        # Load the data from file
        df = pd.read_csv(file_path)
        print(f"File '{file_path}' loaded successfully.")

        # Preprocess the data
        df = preprocess_data(df)

        # Convert numerical columns to words
        df = convert_numerical_to_words(df)

        return df
    except Exception as e:
        print(f"Error during EDA: {e}")
        return None

def preprocess_data(df):
    """
    Handle missing values in the DataFrame.
    """
    print("Preprocessing data: Replacing NaN values with 0.")
    df.fillna(0, inplace=True)  # Replace NaN with 0
    return df

def convert_numerical_to_words(df):
    """
    Convert numerical columns in the DataFrame to their word equivalents.
    """
    print("Converting numerical columns to words...")
    for col in df.select_dtypes(include=['float', 'int']).columns:
        try:
            # Convert column values to numeric type, handling errors
            numeric_values = pd.to_numeric(df[col], errors='coerce')
            # Apply num2words to valid numeric values
            df[col] = numeric_values.apply(lambda x: num2words(x) if pd.notnull(x) else x)
        except Exception as e:
            print(f"Error converting column '{col}' to words: {e}")
    return df

def save_to_file(df, output_path):
    """
    Save the processed DataFrame to a text file.
    """
    try:
        # Save as a CSV file as well for better readability
        csv_output_path = output_path.replace(".txt", ".csv")
        df.to_csv(csv_output_path, index=False)
        print(f"Processed data saved to CSV: {csv_output_path}")
        
        # Convert the dataframe to string and save as text file
        df_str = df.to_string(index=False)
        with open(output_path, 'w') as f:
            f.write(df_str)
        print(f"Processed data saved to text file: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")

# Main execution
def main():
    upload_folder = "uploads"
    output_file = "llmdata.txt"

    # Ensure the uploads folder exists
    if not os.path.exists(upload_folder):
        print(f"Uploads folder '{upload_folder}' does not exist. Please create it and add files.")
        return

    # Get list of CSV files in the folder
    files = [f for f in os.listdir(upload_folder) if f.endswith('.csv')]

    if len(files) > 0:
        file_path = os.path.join(upload_folder, files[0])  # Process the first CSV file
        print(f"Processing file: {file_path}")

        # Perform EDA
        eda_result = perform_eda(file_path)
        if eda_result is not None:
            # Save the result to a file
            save_to_file(eda_result, output_file)
    else:
        print("No CSV files found in the 'uploads' folder.")

if __name__ == "__main__":  # Fixed: Changed _name_ to __name__
    main()
