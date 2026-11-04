import configparser
import os
import re
from pathlib import Path

import pandas as pd

# Translation mapping for location names
TRANSLATIONS = {
    # Longest matches first to avoid partial replacements
    '第十五~十六片': '15-16', '第九~十片': '9-10', '第六~七片': '6-7', '第三~四片': '3-4',
    '下相': 'Low', '上相': 'Up',
    '小号侧': 'S', '大号侧': 'B',
    '串': 'Str',
    '玉中线': 'YZ',
    '片': 'P',
    '左边': 'L', '右边': 'R',
    '_': '-',
}

def translate_location(loc):
    """Translate Chinese location to English abbreviation."""
    result = loc
    for cn, en in TRANSLATIONS.items():
        result = result.replace(cn, en)
    # Clean up: remove redundant dashes and fix format
    result = re.sub(r'(\d+)([A-Z])', r'\1-\2', result)  # Add dash between number and letter
    result = re.sub(r'-+', '-', result)  # Collapse multiple dashes
    result = result.strip('-')
    return result

# Parse config/config.ini to get device locations
config = configparser.ConfigParser()
project_root = Path(__file__).resolve().parents[1]
config.read(project_root / 'config' / 'config.ini', encoding='utf-8')

device_locations = {}
if 'DeviceLocations' in config:
    for device_id, location in config['DeviceLocations'].items():
        device_locations[device_id.strip()] = translate_location(location.strip())

print(f"Loaded {len(device_locations)} device locations from config/config.ini")
for did, loc in list(device_locations.items())[:3]:
    print(f"  {did} -> {loc}")

# Process both Excel files
all_data = []

excel_files = [
    project_root / 'data' / 'raw' / 'extracted_data220.xlsx',
    project_root / 'data' / 'raw' / 'extracted_data110.xlsx',
]

for excel_file in excel_files:
    if not os.path.exists(excel_file):
        print(f"Warning: {excel_file} not found, skipping...")
        continue

    print(f"\nProcessing {excel_file}...")

    # Read all sheets
    xlsx = pd.ExcelFile(excel_file)
    for sheet_name in xlsx.sheet_names:
        device_id = sheet_name.strip()

        # Read the sheet
        df = pd.read_excel(xlsx, sheet_name=sheet_name)

        # Add location column based on sheet name (device ID)
        location = device_locations.get(device_id, 'Unknown')
        df['Location'] = location
        all_data.append(df)
        print(f"  Sheet {sheet_name}: {len(df)} rows, Location: {location}")

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

# Keep only desired columns: IMEI, Location, then original columns (excluding added ones)
keep_cols = ['IMEI', 'Location', 'Timestamp', 'Battery', 'Measure', 'Temp', 'Hum', 'Version']
combined_df = combined_df[keep_cols]

# Add humidity flag column (1 if Hum > 55, else 0)
combined_df['Hum_High'] = (combined_df['Hum'] > 55).astype(int)

# Save to CSV
output_file = project_root / 'data' / 'raw' / 'combined_data.csv'
combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\nDone. Combined {} rows into {}".format(len(combined_df), output_file))
print(f"Columns: {list(combined_df.columns)}")
