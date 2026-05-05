import os
import json
import csv
from pathlib import Path
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Fields to extract from JSON
REQUIRED_FIELDS = [
    "dc_identifier",
    "dc_title",
    "dc_description",
    "dc_source",
    "dc_date_year",
    "dc_externalIdentifier",
    "foi_meetingYear",
    "foi_meetingDate",
    "foi_meetingNumber",
    "foi_meetingItemNumber",
    "foi_handelingType",
    "foi_startPage",
    "foi_endPage",
    "foi_meetingItemNumberRaw"
]

def get_overheid_category(external_id):
    """
    Fetch OVERHEID.category from metadata.xml using the external identifier
    Extracts the content attribute from metadata elements with name="OVERHEID.category"
    """
    try:
        url = f"https://zoek.officielebekendmakingen.nl/{external_id}/metadata.xml"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            
            # Find all metadata elements with name="OVERHEID.category" and extract content attribute
            categories = []
            for metadata in root.findall('.//metadata'):
                name = metadata.get('name')
                if name == 'OVERHEID.category':
                    content = metadata.get('content')
                    if content:
                        categories.append(content)
            
            # Return all categories joined by semicolon, or empty string if none found
            return '; '.join(categories) if categories else ""
        
        return ""
    
    except Exception as e:
        print(f"Error fetching metadata for {external_id}: {str(e)}")
        return ""

def extract_fields_from_json(json_data):
    """
    Extract required fields from JSON data
    """
    extracted = {}
    for field in REQUIRED_FIELDS:
        extracted[field] = json_data.get(field, "")
    return extracted

def process_all_files(data_dir, output_file):
    """
    Process all JSON files in the data directory across all years
    """
    all_data = []
    file_count = 0
    
    # Check if data directory exists
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Loop through all year directories
    year_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if not year_dirs:
        print(f"No year directories found in {data_dir}")
        return
    
    print(f"Found {len(year_dirs)} year directories")
    
    # First pass: Extract JSON fields from all files
    print("\n[Phase 1] Extracting JSON fields...")
    for year_dir in year_dirs:
        print(f"  Processing year: {year_dir.name}")
        
        # Get all JSON files in the year directory
        json_files = sorted(year_dir.glob("*.json"))
        print(f"    Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Extract required fields
                extracted_data = extract_fields_from_json(json_data)
                all_data.append(extracted_data)
                file_count += 1
                
                if file_count % 100 == 0:
                    print(f"    Processed {file_count} files...")
            
            except Exception as e:
                print(f"    Error processing {json_file.name}: {str(e)}")
                continue
    
    print(f"\n✓ Extracted {file_count} JSON files")
    
    # Second pass: Fetch metadata concurrently
    print("\n[Phase 2] Fetching metadata concurrently...")
    
    # Prepare tasks with index to track which data entry to update
    metadata_tasks = {}
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for idx, extracted_data in enumerate(all_data):
            external_id = extracted_data.get("dc_externalIdentifier", "")
            if external_id:
                future = executor.submit(get_overheid_category, external_id)
                metadata_tasks[future] = (idx, external_id)
        
        completed = 0
        for future in as_completed(metadata_tasks):
            idx, external_id = metadata_tasks[future]
            try:
                overheid_category = future.result()
                all_data[idx]["OVERHEID.category"] = overheid_category
            except Exception as e:
                print(f"  Error fetching metadata for {external_id}: {str(e)}")
                all_data[idx]["OVERHEID.category"] = ""
            
            completed += 1
            if completed % 100 == 0:
                print(f"  Fetched metadata for {completed}/{file_count} files...")
    
    print(f"✓ Fetched metadata for {completed} files")
    
    # Write to CSV
    if all_data:
        csv_fields = REQUIRED_FIELDS + ["OVERHEID.category"]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"\n✓ Successfully extracted {file_count} files")
        print(f"✓ Output saved to: {output_file}")
    else:
        print("No files were processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract JSON files to CSV with OVERHEID metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_to_csv.py
  python extract_to_csv.py --input ./data --output ./results/output.csv
  python extract_to_csv.py -i /path/to/data -o /path/to/output.csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data',
        help='Input folder containing year directories with JSON files (default: data)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file path (default: extracted_data_TIMESTAMP.csv in current directory)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    data_dir = Path(args.input)
    
    # Use provided output path or generate timestamped default
    if args.output:
        output_file = Path(args.output)
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_file = Path(f"extracted_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    print("Starting extraction of JSON files to CSV...")
    print(f"Data directory: {data_dir}")
    print(f"Output file: {output_file}")
    process_all_files(data_dir, output_file)
