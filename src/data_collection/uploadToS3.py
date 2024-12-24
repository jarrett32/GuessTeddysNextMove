import json
import os
import shutil
from pathlib import Path

import boto3


def get_bucket_name():
    terraform_state_path = Path('terraform/terraform.tfstate')
    try:
        with open(terraform_state_path) as f:
            tf_state = json.load(f)
            return tf_state['outputs']['data_collection_bucket_name']['value']
    except Exception as e:
        raise Exception("Failed to read bucket name from Terraform state") from e

def get_s3_client():
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_REGION')
    return boto3.client('s3', aws_access_key_id=access_key,
                        aws_secret_access_key=secret_key, region_name=region)

def upload_files_to_s3(s3_client):
    # Set up paths
    data_dir = Path('data')
    saved_dir = data_dir / 'saved'

    saved_dir.mkdir(exist_ok=True)

    BUCKET_NAME = get_bucket_name()

    for file_path in data_dir.glob('*'):
        if file_path.is_dir() or file_path.name == 'saved':
            continue

        try:
            try:
                s3_client.head_object(Bucket=BUCKET_NAME, Key=file_path.name)
                continue
            except s3_client.exceptions.ClientError:
                print(f"Uploading {file_path.name} to S3...")
                s3_client.upload_file(
                    str(file_path),
                    BUCKET_NAME,
                    file_path.name
                )

                shutil.move(str(file_path), str(saved_dir / file_path.name))
                print(f"Moved {file_path.name} to saved directory")

        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    s3_client = get_s3_client()
    upload_files_to_s3(s3_client)
