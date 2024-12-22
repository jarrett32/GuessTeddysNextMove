provider "aws" {
    region = var.aws_region
}

# Figure a cheap way to backup this videos in case i delete them 
resource "aws_s3_bucket" "teddy_videos" {
    bucket = var.data_collection_bucket_name
}