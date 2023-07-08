import sys
import os
import pandas as pd
from fastai.vision.all import *
from huggingface_hub import from_pretrained_fastai

def label_func(f): return f.name[:2]

country_codes = {
    'AT': 'Austria',
    'ES': 'Spain',
    'GR': 'Greece',
    'HR': 'Croatia',
    'HU': 'Hungary',
    'MD': 'Moldova',
    'ME': 'Montenegro',
    'PL': 'Poland',
    'PT': 'Portugal',
    'RO': 'Romania',
    'RS': 'Serbia',
    'SI': 'Slovenia',
    'TR': 'Turkey',
}

# Function to convert country codes to full country names
def convert_to_country_name(code):
    return country_codes.get(code, "Unknown country code")

def predict_image(image_path):
    learn = from_pretrained_fastai("smaciu/bee-wings-classifier")
    img = PILImage.create(image_path)
    bee,_,probs = learn.predict(img)
    return bee, max(probs)

def process_directory(directory_path):
    predictions = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            cat, prob = predict_image(image_path)
            country = convert_to_country_name(cat)
            predictions.append({"filename": filename, "country": country, "probability": prob.item()})

    return pd.DataFrame(predictions)

if __name__ == "__main__":
    directory_path = sys.argv[1]
    df = process_directory(directory_path)
    df.to_csv("predictions.csv", index=False)
    print(f"Predictions saved to predictions.csv")
