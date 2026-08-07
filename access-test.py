import boto3
s3 = boto3.client("s3")
print(s3.list_objects_v2(Bucket="aeronation-rag-test-log-and-chat-history", Prefix="persist/"))