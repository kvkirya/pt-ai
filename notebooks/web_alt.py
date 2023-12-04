import streamlit as st
import numpy as np
from PIL import Image
from time import sleep
import cv2




def webcam():
    cap = cv2.VideoCapture(0)


    frame_placeholder=st.empty()
    stop_button_pressed = st.button('Stop')

    while cap.isOpened() and not stop_button_pressed:
        ret, frame = cap.read()

        if not ret:
            st.write('Video Capture has terminated')
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels='RGB')

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break


    cv2.imshow('Body Detector', frame)


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

def home():
    st.subheader('''''')

def about_the_team():
    st.title('About the Team')
    st.write("Welcome to the PT AI Team page!")
    st.write("Our team is dedicated to creating innovative solutions for physical training and health.")

def squat_tutorial():
    st.title('Squat Tutorial')
    st.markdown("![Alt Text](https://seven.app/media/images/image4.gif)")
    st.subheader('Proper Foot Placement:')
    st.write('''Stand with your feet shoulder-width apart. Your toes should be slightly turned out, around 5-20 degrees, depending on your comfort and mobility.
Keep your weight evenly distributed across your feet.''')
    st.subheader('''Posture:''')
    st.write('''Maintain a neutral spine with your chest up and shoulders back.
Engage your core by pulling your belly button toward your spine. This will help stabilize your spine during the squat.''')
    st.subheader('''Squat Descent:''')
    st.write('''Begin the squat by pushing your hips back, as if you are sitting into a chair. Imagine that your hips are moving backward and downward simultaneously.
Keep your knees in line with your toes and make sure they don't cave inward.
Lower yourself gradually and maintain control. Go as low as your mobility and comfort allow, ideally until your thighs are parallel to the ground or even lower (known as a deep squat).
Keep your weight on your heels or mid-foot, not on your toes.''')

    st.subheader('''Depth and Range of Motion:''')
    st.write('''Aim for a full range of motion if your mobility allows it. Going deeper can activate more muscle fibers.
Ensure your knees do not go past your toes when you squat.
''')

    st.subheader('''Squat Ascent:''')
    st.write('''Push through your heels to stand back up. Keep your core engaged throughout the movement.
Straighten your hips and knees simultaneously as you return to the starting position.''')

def pushup_tutorial():
    st.title('Push-Up Tutorial')
    st.markdown("![Proper Push-Up](https://example.com/your_pushup_gif.gif)")
    st.subheader('Proper Push-Up Form:')
    st.write('''1. **Starting Position:** Begin with your hands placed slightly wider than shoulder-width apart, and your palms flat on the floor.
    2. **Body Alignment:** Keep your body in a straight line from head to heels. Engage your core muscles to maintain this alignment throughout the exercise.
    3. **Elbows Tucked:** Lower your chest toward the ground by bending your elbows. Keep your elbows close to your body at a 45-degree angle.
    4. **Full Range of Motion:** Lower your body until your chest is about an inch from the floor, or as low as your mobility allows.
    5. **Push Back Up:** Push through your palms and extend your arms to return to the starting position.
    6. **Breathing:** Inhale as you lower your body, and exhale as you push back up.
    7. **Repetitions and Sets:** Aim for 3 sets of 10-15 repetitions, or adjust based on your fitness level.
    8. **Variations:** You can modify push-ups by doing them on your knees or elevating your hands on a stable surface if needed.
    9. **Rest:** Make sure to take adequate rest between sets to maintain proper form.''')

def lunge_tutorial():
    st.title('Lunge Tutorial')
    st.markdown("![Proper Lunge](https://example.com/your_lunge_gif.gif)")
    st.subheader('Proper Lunge Form:')
    st.write('''1. **Starting Position:** Stand up straight with your feet hip-width apart.
    2. **Step Forward:** Take a step forward with one leg, bending both knees to create two 90-degree angles.
    3. **Front Knee Alignment:** Ensure that your front knee is directly above your front ankle.
    4. **Back Knee:** Your back knee should hover just above the ground without touching it.
    5. **Upright Posture:** Keep your upper body straight with your chest up and shoulders back.
    6. **Core Engagement:** Engage your core muscles to maintain balance and stability.
    7. **Step Back:** Push off with your front foot to return to the starting position.
    8. **Switch Legs:** Alternate between your left and right legs for each lunge.
    9. **Breathing:** Inhale as you step forward and lower into the lunge, and exhale as you push back up.
    10. **Repetitions and Sets:** Aim for 3 sets of 10-12 lunges per leg or adjust based on your fitness level.
    11. **Variations:** You can modify lunges by doing reverse lunges, walking lunges, or adding weights for extra resistance.
    12. **Rest:** Take adequate rest between sets to maintain proper form.''')


def main():
    st.set_page_config(layout='wide')

     # Create a horizontal layout for the title and logo
    title_column, logo_column = st.columns([5, 1])


    with title_column:
        st.title('PT-AI')
        st.sidebar.title("Navigation")

    with logo_column:
        st.image("https://t4.ftcdn.net/jpg/02/49/85/41/360_F_249854185_WiRZhGX2B81qEtXcYVCcNiyBVDfeFWIb.jpg", use_column_width=True)



    # Add navigation buttons with custom styling
    if st.sidebar.button("Home"):
        home()
    if st.sidebar.button("About the Team"):
        about_the_team()
    if st.sidebar.button("Squat Tutorial"):
        squat_tutorial()
    if st.sidebar.button("Pushup Tutorial"):
        pushup_tutorial()
    if st.sidebar.button("Lunge Tutorial"):
        lunge_tutorial()

    left_column, right_column = st.columns(2)

    with left_column:
        start = st.button('Start')
        if start:
            loading_bar_dict = {
                'Loading Movenet ...': 2
            }
            show_loading_bars(loading_bar_dict)
            webcam()

    with right_column:
        st.subheader('Number of Reps')
        slider_count = st.slider('', min_value=3, max_value=20, step=2, format=None, key=None, label_visibility="visible")

if __name__ == "__main__":
    main()
