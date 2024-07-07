import torch
from pysrt import SubRipFile
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer
model_path = "./model"  # Make sure this points to your model's directory
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

def batch_translate_text(texts, model, tokenizer, batch_size=10):
    # Process texts in batches for efficiency
    translations = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # Move inputs to the same device as model
        with torch.no_grad():  # Ensure no gradients are computed to save memory and computations
            translated_tokens = model.generate(**inputs)
        batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translations.extend(batch_translations)
    return translations

def translate_srt_file(input_file_path, output_file_path, model, tokenizer, batch_size=10):
    subs = SubRipFile.open(input_file_path)
    texts = [sub.text for sub in subs]
    translated_texts = batch_translate_text(texts, model, tokenizer, batch_size=batch_size)
    
    # Update subtitles with their translations
    for sub, translation in zip(subs, translated_texts):
        sub.text = translation
    
    # Save the translated subtitles to a new file
    subs.save(output_file_path, encoding='utf-8')

# Example usage
input_srt_file = "english2.srt"  # Adjust to your input file path
output_srt_file = "arabic-patch2.srt"  # Adjust to your desired output file path
translate_srt_file(input_srt_file, output_srt_file, model, tokenizer, batch_size=10)  # Adjust batch_size based on your model and hardware
