import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    original_excel = "C:/Users/andre/PyCharmMiscProject/dataset_raw/SCUT-FBP5500_v2/All_Ratings.xlsx"
    original_img_dir = "C:/Users/andre/PyCharmMiscProject/dataset_raw/SCUT-FBP5500_v2/Images"
    base_out_dir = "C:/Users/andre/PyCharmMiscProject/dataset"

    # 1. Load Data
    print("Loading Excel file (this might take a minute for 330k rows)...")
    df = pd.read_excel(original_excel)
    print(f"Total rows loaded: {len(df)}")

    # 2. Clean Data
    # Drop rows with empty Filename or Rating (removes trailing empty cells)
    df = df.dropna(subset=['Filename', 'Rating'])

    # 3. Aggregate Data (CRITICAL ML STEP)
    # Group by Filename and calculate the mean (average) of the Rating column.
    # This prevents Data Leakage and reduces 330k rows to ~5500 unique images.
    print("Averaging multiple ratings per image...")
    # Keep only needed columns before grouping
    df = df[['Filename', 'Rating']]
    df = df.groupby('Filename', as_index=False)['Rating'].mean()
    print(f"Unique images after grouping: {len(df)}")

    # 4. Split Data (80% Train, 20% Val)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # 5. Create Target Directories
    for split in ['train', 'val']:
        os.makedirs(os.path.join(base_out_dir, split), exist_ok=True)

    def process_split(split_df, split_name):
        out_dir = os.path.join(base_out_dir, split_name)

        # Save aggregated CSV
        csv_path = os.path.join(base_out_dir, f"{split_name}.csv")
        split_df.to_csv(csv_path, index=False)

        # Copy image files
        print(f"Processing {split_name} subset...")
        for img_name in tqdm(split_df['Filename'], desc=f"Copying {split_name}", leave=False):
            src = os.path.join(original_img_dir, str(img_name))
            dst = os.path.join(out_dir, str(img_name))

            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"Warning: File missing -> {src}")

    process_split(train_df, 'train')
    process_split(val_df, 'val')
    print("\nData splitting and copying operations complete!")


if __name__ == "__main__":
    main()