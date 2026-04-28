import streamlit as st
import cv2
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="Video Recorder", layout="centered")

st.title("🎥 Streamlit Video Recorder")
st.write("1️⃣ Click START below to enable camera\n2️⃣ Then click Record")

# Session state
if "frames" not in st.session_state:
    st.session_state.frames = []
if "recording" not in st.session_state:
    st.session_state.recording = False

# WebRTC config (IMPORTANT)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Video processor
class VideoRecorder(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Only record if BOTH conditions true
        if st.session_state.recording:
            st.session_state.frames.append(img)

        return img

# WebRTC streamer (camera)
webrtc_ctx = webrtc_streamer(
    key="video",
    video_transformer_factory=VideoRecorder,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)

# RECORD button (only works if camera running)
if webrtc_ctx.state.playing:
    if st.button("🔴 Start Recording"):
        st.session_state.recording = True
        st.session_state.frames = []
        st.success("Recording started...")

    if st.button("⏹ Stop Recording"):
        st.session_state.recording = False
        st.success("Recording stopped!")

        if len(st.session_state.frames) > 0:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

            height, width, _ = st.session_state.frames[0].shape

            out = cv2.VideoWriter(
                temp_video.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                20,
                (width, height),
            )

            for frame in st.session_state.frames:
                out.write(frame)

            out.release()

            st.subheader("🎬 Recorded Video")
            st.video(temp_video.name)

            with open(temp_video.name, "rb") as f:
                st.download_button(
                    "⬇️ Download Video",
                    f,
                    "recorded_video.mp4",
                    "video/mp4",
                )

            st.session_state.frames = []

        else:
            st.warning("No frames captured. Try recording again.")

else:
    st.warning("⚠️ Please click START on the camera first")
