"""Create vibechecker.db and the storage/ directory. Run once."""

import os
from models import Base, engine


def init_database():
    os.makedirs("storage/images", exist_ok=True)
    Base.metadata.create_all(engine)
    print("Database created: vibechecker.db")
    print("Image storage:    storage/images/")
    print("Tables:")
    for table_name in Base.metadata.tables:
        print(f"  - {table_name}")


if __name__ == "__main__":
    init_database()
