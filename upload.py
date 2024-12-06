from datasets import load_from_disk
from huggingface_hub import create_repo, login

repo_name='GigaCaps_val'
datadir = "/data/lmorove1/hwang258/SSR-Speech/gigacaps-val/"
# Step 1: Authenticate
login(token="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")  # Replace with your token

# Step 2: Create the repository
create_repo(repo_id=f"OpenSound/{repo_name}", repo_type="dataset", private=True)

# Step 3: Load the dataset

dataset = load_from_disk(datadir)

# Step 4: Push the dataset to the Hugging Face Hub
dataset.push_to_hub(repo_id=f"OpenSound/{repo_name}", max_shard_size="100MB", private=True)
