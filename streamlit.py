import streamlit as st
import torch
import datetime
from pysrt import SubRipFile, SubRipItem, SubRipTime
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tempfile import NamedTemporaryFile
import time

# Assuming the model and tokenizer are initialized and loaded outside of a function
model_path = "./model"  # Adjust this to the path where your model is stored
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

def estimate_total_time(num_texts, batch_size=10):
    # Simple estimation function; adjust based on your model's performance
    time_per_batch = 6  # Assuming each batch takes 5 seconds as a placeholder
    total_batches = -(-num_texts // batch_size)  # Ceiling division
    return total_batches * time_per_batch

def batch_translate_text(texts, model, tokenizer, batch_size=10, progress_bar=None, timer_placeholder=None, start_time=None, total_estimate_time=None):
    translations = []
    for i in range(0, len(texts), batch_size):
        current_time = time.time()
        elapsed_time = current_time - start_time
        remaining_time = max(total_estimate_time - elapsed_time, 0)
        progress = min(elapsed_time / total_estimate_time, 1.0)
        
        if progress_bar is not None and timer_placeholder is not None:
            progress_bar.progress(progress)
            timer_placeholder.text(f"Estimated time remaining: {int(remaining_time)} seconds")
        
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            translated_tokens = model.generate(**inputs)
        batch_translations = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
        translations.extend(batch_translations)
        
        # Optional: simulate or wait for real processing time
        # time.sleep(1) # Remove or adjust based on actual processing time
        
    return translations
def translate_srt_content(srt_content, model, tokenizer, batch_size=10, 
                          opening_sentence="| ترجمة بالذكاء الإصطناعي علي طريقة إسلام الجيز!وي |", 
                          closing_sentence="| فكرة وتصميم وتنفيذ عبدالله عفيفي |"):
    subs = SubRipFile.from_string(srt_content)
    texts = [sub.text for sub in subs]
    
    total_estimate_time = estimate_total_time(len(texts), batch_size=batch_size)
    progress_bar = st.progress(0)
    timer_placeholder = st.empty()
    start_time = time.time()
    
    translated_texts = batch_translate_text(texts, model, tokenizer, batch_size=batch_size, progress_bar=progress_bar, timer_placeholder=timer_placeholder, start_time=start_time, total_estimate_time=total_estimate_time)
    
    progress_bar.empty()
    timer_placeholder.empty()
    
    # Update the subtitles with the translated texts
    for sub, translation in zip(subs, translated_texts):
        sub.text = translation

    # Create opening and closing subtitle items
    opening_sub = SubRipItem(index=0,
                             start=SubRipTime(0, 0, 0, 0),
                             end=SubRipTime(0, 0, 5, 0),  # 5 seconds duration
                             text=opening_sentence)
    
    closing_sub = SubRipItem(index=len(subs)+1,
                             start=subs[-1].end + SubRipTime(0, 0, 5, 0),  # Start 5 seconds after the last subtitle
                             end=subs[-1].end + SubRipTime(0, 0, 10, 0),  # 5 seconds duration
                             text=closing_sentence)

    # Insert the opening and closing subtitles
    subs.insert(0, opening_sub)
    subs.append(closing_sub)

    # Convert the updated SubRipFile back to a string
    translated_srt_content = ''
    for sub in subs:
        translated_srt_content += str(sub) + '\n\n'

    return translated_srt_content

# Streamlit UI code remains the same
st.title("SRT Translator: English to Arabic")

uploaded_file = st.file_uploader("Choose an SRT file", type=['srt'])
if uploaded_file is not None:
    srt_content = uploaded_file.read().decode("utf-8")
    if st.button('Translate'):
        with st.spinner('Translating...'):
            translated_srt_content = translate_srt_content(srt_content, model, tokenizer, batch_size=10)
            st.success('Translation complete!')
            # Create a link to download the translated content
            tmp_file = NamedTemporaryFile(delete=False, suffix='.srt')
            with open(tmp_file.name, 'w', encoding='utf-8') as f:
                f.write(translated_srt_content)

            st.download_button(label="Download Translated SRT",
                               data=translated_srt_content,
                               file_name="translated.srt",
                               mime='text/srt')
