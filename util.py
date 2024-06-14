import os
from pymongo import MongoClient


def crateMongoConnection(mongodb_uri: str = None):
    return MongoClient(mongodb_uri)
