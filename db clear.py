#!/usr/bin/env python3
"""
Script to clear all prediction data from the Roman Numeral Classifier database
"""

import sqlite3
import os
from datetime import datetime

def clear_predictions_database(db_path="predictions.db"):
    """
    Clear all prediction data from the database
    
    Args:
        db_path (str): Path to the database file
    """
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            print(f"❌ Database file '{db_path}' not found!")
            print("ℹ️  This might mean no predictions have been made yet.")
            return False
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current record count before clearing
        cursor.execute('SELECT COUNT(*) FROM predictions')
        record_count = cursor.fetchone()[0]
        
        if record_count == 0:
            print("ℹ️  Database is already empty - no records to clear.")
            conn.close()
            return True
        
        print(f"📊 Found {record_count} prediction records in database")
        
        # Confirm deletion
        response = input(f"❓ Are you sure you want to delete all {record_count} records? (y/N): ")
        
        if response.lower() not in ['y', 'yes']:
            print("❌ Operation cancelled by user")
            conn.close()
            return False
        
        # Clear all records
        cursor.execute('DELETE FROM predictions')
        deleted_count = cursor.rowcount
        
        # Reset the auto-increment counter (optional)
        cursor.execute('DELETE FROM sqlite_sequence WHERE name="predictions"')
        
        # Commit changes
        conn.commit()
        conn.close()
        
        print(f"✅ Successfully deleted {deleted_count} prediction records")
        print("🔄 Database has been reset - prediction IDs will start from 1 again")
        
        return True
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def backup_database_before_clear(db_path="predictions.db"):
    """
    Create a backup of the database before clearing it
    
    Args:
        db_path (str): Path to the database file
    """
    try:
        if not os.path.exists(db_path):
            print("ℹ️  No database to backup")
            return True
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"predictions_backup_{timestamp}.db"
        
        # Simple file copy for SQLite database
        import shutil
        shutil.copy2(db_path, backup_path)
        
        print(f"💾 Database backed up to: {backup_path}")
        return True
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def verify_database_empty(db_path="predictions.db"):
    """
    Verify that the database is empty after clearing
    
    Args:
        db_path (str): Path to the database file
    """
    try:
        if not os.path.exists(db_path):
            print("ℹ️  Database file doesn't exist - considered empty")
            return True
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        count = cursor.fetchone()[0]
        
        conn.close()
        
        if count == 0:
            print("✅ Verification passed - database is empty")
            return True
        else:
            print(f"❌ Verification failed - {count} records still exist")
            return False
            
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False

def main():
    """Main function to clear predictions database"""
    print("🏛️ Roman Numeral Classifier - Database Clear Utility")
    print("=" * 60)
    
    db_path = "predictions.db"
    
    # Option 1: Clear with backup
    print("📋 Options:")
    print("1. Clear database with backup")
    print("2. Clear database without backup")
    print("3. Just create backup (don't clear)")
    print("4. Check current database status")
    
    choice = input("\n❓ Choose an option (1-4): ").strip()
    
    if choice == "1":
        print("\n💾 Creating backup before clearing...")
        if backup_database_before_clear(db_path):
            print("\n🗑️ Clearing database...")
            if clear_predictions_database(db_path):
                verify_database_empty(db_path)
        
    elif choice == "2":
        print("\n🗑️ Clearing database without backup...")
        if clear_predictions_database(db_path):
            verify_database_empty(db_path)
            
    elif choice == "3":
        print("\n💾 Creating backup only...")
        backup_database_before_clear(db_path)
        
    elif choice == "4":
        print("\n📊 Checking database status...")
        try:
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT COUNT(*) FROM predictions')
                count = cursor.fetchone()[0]
                
                cursor.execute('SELECT MIN(timestamp), MAX(timestamp) FROM predictions')
                date_range = cursor.fetchone()
                
                conn.close()
                
                print(f"📊 Database contains {count} prediction records")
                if count > 0 and date_range[0]:
                    print(f"📅 Date range: {date_range[0]} to {date_range[1]}")
            else:
                print("ℹ️  Database file doesn't exist yet")
                
        except Exception as e:
            print(f"❌ Error checking database: {e}")
    
    else:
        print("❌ Invalid choice")
    
    print("\n" + "=" * 60)
    print("✅ Operation completed")

if __name__ == "__main__":
    main()