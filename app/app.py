import streamlit as st
import numpy as np
from PIL import Image
from time import sleep

# HTML colour settings: progress bar = blue
st.markdown("""
<style>
.stProgress .st-bo {
    background-color: blue;
}
</style>
""", unsafe_allow_html=True)

def white_space(rows: int):
    for i in range(rows):
        st.text('')

def show_uploaded_image(uploaded_image):
    ''' This displays an uploaded file
    '''
    image = Image.open(uploaded_image)
    st.image(image, use_column_width=True)

def show_loading_bars(loading_bar_dict):
    ''' This shows a loading bar for each key in a dictionary provided,
        for the number of seconds included in their keys
    '''
    # Delay
    sleep(0.5)

    # Extract data from loading_bar_dict
    text_prompts = list(loading_bar_dict.keys())
    time_delay = list(loading_bar_dict.values())

    # Create a loading bar for each dictionary key
    for i in range(len(text_prompts)):
        loading_bar(text_prompts[i], time_delay[i])

    # Print success message and delete after 2 seconds
    success = st.success('✅ Classifcation complete!')
    sleep(1)
    success.empty()

def loading_bar(text: str, loading_time: int):
    ''' This creates a loading bar that displays the inputted text
        and lasts for the loading_time length
    '''
    progress_bar = st.progress(0)
    st.text(text)
    for i in range(loading_time * 10):
        progress = (i + 1) / (loading_time * 10)
        progress_bar.progress(progress)
        sleep(0.1)

def show_results():
    return 1

def main():

    st.title('X-ray classifier')

    # Store an uploaded image
    uploaded_image = st.file_uploader('', type=["jpg", "jpeg", "png"])

    white_space(2)

    # Display an uploaded image and loading bars
    left_column, right_column = st.columns(2)
    if uploaded_image is not None:
        # Here is where will call a function to actually start running the model
        # while the image is displayed / loading bar is shown
        with left_column:
            show_uploaded_image(uploaded_image)
        with right_column:
            if start:
                loading_bar_dict = {'Upoading image ...': 2,
                                'Classifying image ...': 2,
                                'Downloading results ...': 2

                }
                show_loading_bars(loading_bar_dict)

        white_space(2)

        # Display the model results
        st.text('Healthy lung')

if __name__ == "__main__":
    main()
