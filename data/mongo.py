from pymongo import MongoClient
from datetime import datetime

# MongoDB client
client = MongoClient("mongodb://localhost:27017/")

# Create the database
db = client["Tillotoma"]

# Create the collection for waste disposal data
waste_disposal = db["waste_disposal"]

# Insert sample data
waste_disposal.insert_many([
    {
        "image_id": 1,
        "type_of_waste": 1,  # E.g., 1 for Organic, 2 for Recyclable, etc.
        "collection": 0.75,  # Volume in cubic meters
        "destination_zone": 3.5  # Zone in geographic coordinates or similar
    },
    {
        "image_id": 2,
        "type_of_waste": 2,
        "collection": 0.50,
        "destination_zone": 2.0
    },
    {
        "image_id": 3,
        "type_of_waste": 1,
        "collection": 1.25,
        "destination_zone": 1.0
    },
    {
        "image_id": 4,
        "type_of_waste": 3,
        "collection": 0.85,
        "destination_zone": 4.0
    }
])

print("Database and collection for waste disposal data created and populated successfully!")