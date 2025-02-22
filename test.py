import boto3

dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('stockdat')  # Replace with your table name

print(table.table_status)