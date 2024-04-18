import pandas as pd
import os

# Define the CSV file path
csv_file_path = "data/nvidia/NvidiaDocumentationQandApairs.csv"

# Your existing code...
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Create the directories if they don't exist
os.makedirs("data/nvidia/questions", exist_ok=True)
os.makedirs("data/nvidia/answers", exist_ok=True)

# Save the answers into text files
for i, question in enumerate(df["question"]):
    with open(f"data/nvidia/questions/question_{i}.txt", "w") as f:
        f.write(str(question))

for i, answer in enumerate(df["answer"]):
    with open(f"data/nvidia/answers/answer_{i}.txt", "w") as f:
        f.write(str(answer))
