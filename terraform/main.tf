# How to run:
# terraform init
# terraform plan
# terraform apply

provider "aws" {
    region = var.aws_region
}

# Figure a cheap way to backup this videos in case i delete them 
resource "aws_s3_bucket" "data_collection_bucket" {
    bucket = var.data_collection_bucket_name
}