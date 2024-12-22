output "data_collection_bucket_name" {
    value = aws_s3_bucket.data_collection_bucket.id
}

output "data_collection_bucket_arn" {
    value = aws_s3_bucket.data_collection_bucket.arn
}