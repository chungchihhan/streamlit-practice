import streamlit as st
import base64
from PIL import Image
import io

st.set_page_config(page_title="base64 converter", layout="wide")

# Function to encode image to Base64
def encode_image(image_file):
    """Encode image to Base64."""
    image_data = image_file.getvalue()
    base64_encoded = base64.b64encode(image_data)
    return base64_encoded.decode("utf-8")


# Function to decode Base64 encoded image to file
def decode_image(base64_encoded):
    """Decode Base64 encoded image."""
    image_data = base64.b64decode(base64_encoded.encode("utf-8"))
    return Image.open(io.BytesIO(image_data))


# Streamlit interface
st.title("Base64 Image Encoder/Decoder")
tab1, tab2 = st.tabs(["Image to Base64", "Base64 to Image"])

# Image upload section
with tab1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", width=250)
        # Encode the image
        encoded_image = encode_image(uploaded_file)
        # st.text_area("Base64 Encoded Image", encoded_image, height=250)
        st.subheader("Copy your Base64 down below")
        st.code(encoded_image)

# Base64 decode section
with tab2:
    base64_string = st.text_area("Enter Base64 Image String to Decode", height=250)
    if st.button("Decode Image"):
        if base64_string:
            # Decode the image
            decoded_image = decode_image(base64_string)
            # Display the decoded image
            st.image(decoded_image, caption="Decoded Image", use_column_width=True)
        else:
            st.error("Please enter a valid Base64 string.")
