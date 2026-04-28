import streamlit as st
import cv2
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="Video Recorder", layout="centered")

st.title("🎥 Streamlit Video Recorder")
st.write("Click **Start** to begin recording and **Stop** to save and view result.")

# Initialize session state
if "frames" not in st.session_state:
    st.session_state.frames = []
if "recording" not in st.session_state:
    st.session_state.recording = False

# WebRTC configuration (IMPORTANT for cloud)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video processor class
class VideoRecorder(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Save frames only when recording is ON
        if st.session_state.recording:
            st.session_state.frames.append(img)

        return img

# Start button
if st.button("▶️ Start Recording"):
    st.session_state.recording = True
    st.session_state.frames = []
    st.success("Recording started...")

# Webcam stream
webrtc_ctx = webrtc_streamer(
    key="video-recorder",
    video_transformer_factory=VideoRecorder,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# Stop button
if st.button("⏹ Stop Recording"):
    st.session_state.recording = False
    st.success("Recording stopped!")

    if len(st.session_state.frames) > 0:
        # Create temp file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        height, width, _ = st.session_state.frames[0].shape

        # Video writer
        out = cv2.VideoWriter(
            temp_video.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (width, height),
        )

        # Write frames
        for frame in st.session_state.frames:
            out.write(frame)

        out.release()

        st.subheader("🎬 Recorded Video")
        st.video(temp_video.name)

        # Download button
        with open(temp_video.name, "rb") as f:
            st.download_button(
                label="⬇️ Download Video",
                data=f,
                file_name="recorded_video.mp4",
                mime="video/mp4",
            )

        # Clear frames after saving
        st.session_state.frames = []

    else:
        st.warning("No video recorded. Please click Start first.")
