import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="ASL Detector",
    page_icon="🤟",
    layout="wide"
)

# -----------------------------
# Session State
# -----------------------------
if "camera_on" not in st.session_state:
    st.session_state["camera_on"] = False

if "detected_sign" not in st.session_state:
    st.session_state["detected_sign"] = "-"

if "confidence" not in st.session_state:
    st.session_state["confidence"] = 0

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    with open("model/asl_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, encoder

model, encoder = load_model()

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)
# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background:#1F1736;
}

.title{
    text-align:center;
    color:white;
    font-size:42px;
    font-weight:bold;
    text-decoration:underline;
    margin-bottom:30px;
}

.camera-box{
    background:#D4D4D8;
    height:350px;
    border-radius:12px;
    display:flex;
    justify-content:center;
    align-items:center;
    color:#666;
    font-size:18px;
}

div[data-testid="stButton"] button{
    height:50px;
    border-radius:12px;
    font-weight:bold;
}

div[data-testid="stButton"] button[kind="primary"]{
    background:#22C55E !important;
    color:white !important;
}

div[data-testid="stButton"] button:not([kind="primary"]){
    background:#EF4444 !important;
    color:white !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown(
    '<div class="title">American Sign Language Detector</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Start / Stop Buttons
# -----------------------------
btn1, btn2, spacer = st.columns([1,1,4])

with btn1:
    if st.button(
        "▶ Start",
        use_container_width=True,
        type="primary"
    ):
        st.session_state["camera_on"] = True

with btn2:
    if st.button(
        "■ Stop",
        use_container_width=True
    ):
        st.session_state["camera_on"] = False
# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([5,1])

with left_col:

    camera_placeholder = st.empty()

    if st.session_state["camera_on"]:

        cap = cv2.VideoCapture(0)

        success, frame = cap.read()

        if success:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            results = hands.process(rgb)

            if results.multi_hand_landmarks:

                hand = results.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                landmarks = []

                for lm in hand.landmark:
                    landmarks.extend(
                        [lm.x, lm.y, lm.z]
                    )

                if len(landmarks) == 63:

                    X = np.array(
                        landmarks
                    ).reshape(1, -1)

                    prediction = model.predict(X)

                    label = encoder.inverse_transform(
                        prediction
                    )[0]

                    st.session_state["detected_sign"] = label

            camera_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

        cap.release()

    else:

        camera_placeholder.markdown("""
        <div class="camera-box">
            Camera feed will appear here — press Start
        </div>
        """, unsafe_allow_html=True)

with right_col:

    st.markdown("""
    <div style="
        width:60px;
        height:60px;
        background:#EAB308;
        border-radius:50%;
        display:flex;
        justify-content:center;
        align-items:center;
        font-size:30px;
        margin-top:50px;
        margin-bottom:20px;
    ">
        ?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="
        width:60px;
        height:60px;
        background:#EAB308;
        border-radius:50%;
        display:flex;
        justify-content:center;
        align-items:center;
    ">
        <img src="https://img.icons8.com/ios/50/rules.png" width="28">
    </div>
    """, unsafe_allow_html=True)
# -----------------------------
# Detected Sign
# -----------------------------
st.write("")

st.markdown("""
<h3 style="
color:white;
margin-top:20px;
">
Detected Sign
</h3>
""", unsafe_allow_html=True)

st.markdown(
    f"""
    <div style="
        background:#2A1F4A;
        border:2px solid #2DD4BF;
        border-radius:12px;
        padding:20px;
        text-align:center;
        color:#2DD4BF;
        font-size:48px;
        font-weight:bold;
    ">
        {st.session_state["detected_sign"]}
    </div>
    """,
    unsafe_allow_html=True
)