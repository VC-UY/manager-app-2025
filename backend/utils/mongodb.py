# backend/utils/mongodb.py
from pymongo import MongoClient
from django.conf import settings
from datetime import datetime
from bson.objectid import ObjectId

def get_db_handle():
    client = MongoClient(settings.MONGODB_URI)
    db = client[settings.MONGODB_NAME]
    return db, client

def get_collection(collection_name):
    db, client = get_db_handle()
    return db[collection_name], client