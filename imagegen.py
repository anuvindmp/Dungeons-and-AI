import sys
import os
from huggingface_hub import InferenceClient

os.makedirs("app/generated", exist_ok=True)
prompt = sys.argv[1] if len(sys.argv) > 1 else "A mysterious fantasy scene in a glowing forest"

client = InferenceClient(
    provider="nebius",
    api_key="eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwOTI1ODQzMDAwMjY4MDI1OTczOSIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwODY2MTYyNiwidXVpZCI6ImQ2YzM2NDBiLWRhMjEtNDgzMi05M2M2LWYxOWQ2MmYyZGM1MSIsIm5hbWUiOiJkbmQiLCJleHBpcmVzX2F0IjoiMjAzMC0wNi0yNVQyMzo0NzowNiswMDAwIn0.J8cpXatXwp74bdZcZSWDVSNeBQFawGC751moyt64aIw"
)

image = client.text_to_image(
    prompt=prompt,
    model="black-forest-labs/FLUX.1-dev",
)

# Generate a unique file name based on prompt hash
filename = f"image_{abs(hash(prompt)) % 100000}.png"
save_path = os.path.join("app", "generated", filename)

image.save(save_path)
print(f'{{"imageUrl": "/generated/{filename}"}}')
