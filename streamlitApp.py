import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import keras
import numpy as np
import tensorflow as tf

emotion_model = keras.models.load_model("emotion_model.h5")
# Load pre-trained model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
def main():
# Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Streamlit app
    st.title("Image Captioning & Emotion Classfication")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Process uploaded image
        image = Image.open(uploaded_file)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Generate caption
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Display image and caption
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Caption:", caption)
        
        # Preprocess the image for emotion classification
        image_emotion = image.resize((48, 48)).convert("L")
        image_array_emotion = np.array(image_emotion) / 255.0
        image_tensor_emotion = tf.convert_to_tensor(image_array_emotion)  # Convert to TensorFlow tensor

        # Perform emotion classification
        emotion_prediction = emotion_model.predict(image_tensor_emotion[tf.newaxis, ..., tf.newaxis])  # Add an extra dimension
        emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_index = np.argmax(emotion_prediction)
        predicted_emotion = emotions[emotion_index]
        st.write(f"Predicted Emotion: {predicted_emotion}")
    
if __name__ == "__main__":
    main()
