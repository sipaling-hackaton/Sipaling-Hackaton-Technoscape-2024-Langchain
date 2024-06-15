import os
from pymongo import MongoClient
from google.cloud import storage
from google.oauth2 import service_account
from langchain.document_loaders import PyPDFLoader
import logging
import requests

gcp_credentials = service_account.Credentials.from_service_account_file("creds.json")
logger = logging.getLogger("uvicorn")


def crateMongoConnection(mongodb_uri: str = None):
    return MongoClient(mongodb_uri)


def download_pdf_from_gcs(
    bucket_name, source_blob_name, destination_file_name, credentials
):
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)


def load_pdf_from_gcs(url):
    os.makedirs(os.path.dirname("./temp/"), exist_ok=True)

    bucket_name = os.getenv("GCS_BUCKET_NAME")
    source_blob_name = f"{url}"

    destination_file_name = f"./temp/{url}.pdf"
    download_pdf_from_gcs(
        bucket_name, source_blob_name, destination_file_name, gcp_credentials
    )
    return PyPDFLoader(destination_file_name)
