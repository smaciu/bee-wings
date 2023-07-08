import sys
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

if __name__ == "__main__":
    image_path = sys.argv[1]
    cat, prob = predict_image(image_path)
    country = convert_to_country_name(cat)
    print(f"Honey bee from: {country}. {100*prob.item():.2f}%")
    
