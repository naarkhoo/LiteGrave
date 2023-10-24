import streamlit as st
import os
import base64
from pathlib import Path

LOGO_IMAGE = "src/logo.png"
LOGO_WIDTH = 200

def load_css(css_path):
    with open(css_path, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path, width=None, proportion=None):
    # Base64 encoding the image
    img_data = f"<img src='data:image/png;base64,{img_to_bytes(img_path)}' class='img-fluid'"

    # If width is specified
    if width:
        img_data += f" width='{width}'"

    # If proportion is specified
    if proportion:
        # Calculate new width based on the original image dimensions
        original_width = Image.open(img_path).width
        new_width = int(original_width * proportion)
        img_data += f" width='{new_width}'"

    img_data += ">"
    return img_data


def main():
    # Set the current working directory (for debugging purposes).
    print(f"Current working directory: {os.getcwd()}")

    # Load the external CSS.
    load_css('src/style.css')

    logo_html = img_to_html(LOGO_IMAGE, width = LOGO_WIDTH)
    st.markdown(f'<div class="centered">{logo_html}</div>', unsafe_allow_html=True)

    # Centered search bar in the main area.
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    search_query = st.text_input('Search', key="search")
    st.markdown('</div>', unsafe_allow_html=True)

    # If a search query is entered, perform the search (assuming you've defined a search_google function).
    if search_query:
        results = search_google(search_query)

        # Display the search results.
        st.write('Search results:')
        for result in results:
            st.markdown(f'<a href="{result[1]}">{result[0]}</a>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
