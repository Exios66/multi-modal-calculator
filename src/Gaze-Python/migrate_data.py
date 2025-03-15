#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Migration Script

This script migrates existing processed data to the new date-formatted folder structure.
It organizes files based on their timestamps into folders with the format "MM#DD#YYYY".
"""

import os
import shutil
import re
from datetime import datetime

def migrate_data(source_dir='processed_data', target_dir='processed_data'):
    """
    Migrate existing data to the new folder structure.
    
    Parameters:
    -----------
    source_dir : str
        Source directory containing the data to migrate
    target_dir : str
        Target directory where the data will be migrated to
    """
    print(f"Starting data migration from {source_dir}...")
    
    # Regular expression to extract date from filenames (format: YYYYMMDD_HHMMSS)
    date_pattern = re.compile(r'(\d{8})_\d{6}')
    
    # Get all files in the source directory
    files = []
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            # Skip files in already date-formatted directories
            if any(part for part in root.split(os.sep) if re.match(r'\d{2}#\d{2}#\d{4}', part)):
                continue
            
            # Only process files directly in the source directory or its immediate subdirectories
            if root == source_dir or os.path.dirname(root) == source_dir:
                files.append(os.path.join(root, filename))
    
    # Process each file
    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extract date from filename
        match = date_pattern.search(filename)
        if match:
            date_str = match.group(1)  # YYYYMMDD
            try:
                # Parse the date
                file_date = datetime.strptime(date_str, "%Y%m%d")
                
                # Create the date-formatted directory path (MM#DD#YYYY)
                date_folder = file_date.strftime("%m#%d#%Y")
                date_folder_path = os.path.join(target_dir, date_folder)
                
                # Determine the appropriate subdirectory based on file extension
                if filename.endswith('.csv'):
                    subdir = 'csv'
                elif filename.endswith(('.png', '.jpg', '.jpeg')):
                    subdir = 'graphs'
                elif filename.endswith(('.npz', '.npy')):
                    subdir = 'data'
                else:
                    subdir = 'other'
                
                # Create the target directory structure
                target_subdir = os.path.join(date_folder_path, subdir)
                os.makedirs(target_subdir, exist_ok=True)
                
                # Construct the target file path
                target_file_path = os.path.join(target_subdir, filename)
                
                # Copy the file to the new location
                print(f"Moving {file_path} to {target_file_path}")
                shutil.copy2(file_path, target_file_path)
                
            except ValueError:
                print(f"Could not parse date from filename: {filename}")
        else:
            print(f"No date pattern found in filename: {filename}")
    
    print("Data migration completed.")

def clean_up_old_files(source_dir='processed_data', confirm=True):
    """
    Remove old files after migration.
    
    Parameters:
    -----------
    source_dir : str
        Source directory containing the data to clean up
    confirm : bool
        Whether to ask for confirmation before deleting files
    """
    # Regular expression to extract date from filenames (format: YYYYMMDD_HHMMSS)
    date_pattern = re.compile(r'(\d{8})_\d{6}')
    
    # Get all files in the source directory
    files = []
    for root, _, filenames in os.walk(source_dir):
        for filename in filenames:
            # Skip files in already date-formatted directories
            if any(part for part in root.split(os.sep) if re.match(r'\d{2}#\d{2}#\d{4}', part)):
                continue
            
            # Only process files directly in the source directory or its immediate subdirectories
            if root == source_dir or os.path.dirname(root) == source_dir:
                files.append(os.path.join(root, filename))
    
    if not files:
        print("No files to clean up.")
        return
    
    print(f"Found {len(files)} files to clean up.")
    
    if confirm:
        response = input("Are you sure you want to delete these files? (y/n): ")
        if response.lower() != 'y':
            print("Clean-up cancelled.")
            return
    
    # Delete each file
    for file_path in files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    
    print("Clean-up completed.")

if __name__ == "__main__":
    # Migrate existing data
    migrate_data()
    
    # Ask user if they want to clean up old files
    response = input("Do you want to clean up old files after migration? (y/n): ")
    if response.lower() == 'y':
        clean_up_old_files()
    else:
        print("Skipping clean-up. Old files are preserved.") 
