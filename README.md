# PT.AI - Pose Tracking with AI

PT.AI is a project that leverages two APIs to provide real-time feedback on exercise poses. Whether you're doing push-ups, squats, left lunges, or right lunges, PT.AI has got your form covered.

## How it Works

1. **Input an Image**
   - 📷 Provide an image to the first API.

2. **Apply Movenet**
   - 🤖 Movenet is applied to the image, calculating joint angles.

3. **Pose Classification**
   - 🏋️‍♂️ The second API uses XGBoost, an ML algorithm, to classify the pose based on calculated joint angles.

4. **Feedback on Form**
   - 🚀 PT.AI compares the angles with ideal angles for the classified movement.
   - 💡 Receive feedback on your exercise form and know where you can improve.

## How to Use

### API Endpoints

- **Endpoint 1 (Movenet):**
  - Input: Image
  - Output: Joint angles

- **Endpoint 2 (Pose Classification):**
  - Input: Angles from Movenet
  - Output: Classified pose and form feedback
