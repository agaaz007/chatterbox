import sagemaker
from sagemaker.serve.builder.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve import Mode
from app.inference import ChatterboxTTSInferenceSpec
import boto3
import json

# --- Configuration ---
role = "arn:aws:iam::534437858001:role/service-role/AmazonSageMaker-ExecutionRole-20250226T082807"

account_id = boto3.client('sts').get_caller_identity().get('Account')
region = boto3.Session().region_name
image_uri = f'{account_id}.dkr.ecr.{region}.amazonaws.com/chatterbox-streaming-sagemaker:latest'

instance_type = 'ml.g4dn.xlarge'  # GPU instance is recommended
endpoint_name = 'chatterbox-streaming-endpoint'

# --- ModelBuilder Deployment ---
sagemaker_session = sagemaker.Session()

sample_input = {"text": "Hello, how are you?"}
sample_output = {"audio_base64": "base64_encoded_audio_string"}

model_builder = ModelBuilder(
    mode=Mode.SAGEMAKER_ENDPOINT,
    model_server=None,  # Using custom container
    schema_builder=SchemaBuilder(sample_input, sample_output),
    inference_spec=ChatterboxTTSInferenceSpec(),
    role_arn=role,
    image_uri=image_uri
)
model = model_builder.build()

print("Deploying model to a SageMaker endpoint...")
predictor = model.deploy(
    initial_instance_count=1,
    instance_type=instance_type,
    endpoint_name=endpoint_name
)

print(f"\nEndpoint '{predictor.endpoint_name}' deployed successfully.")
print("You can now invoke the endpoint.")

# --- Example Invocation ---
print("\n--- Testing the endpoint ---")
payload = {
    "text": "Hi, I'm a customer support agent from Concentrix, and I'm extremely happy to help you today.",
}

response = predictor.predict(payload)
print("Response from endpoint:", response)

# --- To delete the endpoint when you're done ---
# predictor.delete_endpoint()
# print(f"\nEndpoint '{endpoint_name}' has been deleted.")