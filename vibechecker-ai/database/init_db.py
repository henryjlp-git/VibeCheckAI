"""
VibeChecker AI — Database Initialization
Run this once to create the vibechecker.db file and all tables.

Usage:
    python init_db.py
"""

import os
from models import Base, engine

def init_database():
    # Create the storage directory for images
    os.makedirs("storage/images", exist_ok=True)

    # Create all tables defined in models.py
    Base.metadata.create_all(engine)
    print("Database created successfully: vibechecker.db")
    print("Image storage directory created: storage/images/")
    print()
    print("Tables created:")
    for table_name in Base.metadata.tables:
        print(f"  - {table_name}")

if __name__ == "__main__":
    init_database()
