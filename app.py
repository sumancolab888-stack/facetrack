"""
FaceTrack - project report website (Streamlit)
Left side: index of sections.  Right side: report content + live demo.
Run:  streamlit run app.py
"""
import inspect
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="FaceTrack - Project Report", page_icon="◉", layout="wide",
                   initial_sidebar_state="expanded")

# =============================================================================
# EDIT THESE DETAILS
# =============================================================================
AUTHORS = "Suman Samanta, Pranav Kumar Singh, Rupak Raj"
FACULTY_ADVISOR = "Prof. Prithwijit Guha"
INTERNSHIP_NOTE = "Developed during an internship at EICT IIT Guwahati."
GITHUB_URL = "https://github.com/pranav-cyber1/video_analytics_project"
REPORT_PDF_URL = "https://drive.google.com/file/d/1To6arLcfNqxslLGbkvB88eXt53k9zY1L/view?usp=drive_link"
DEMO_VIDEO_URL = "https://www.youtube.com/watch?v=Lif1jCNxze0"   # project walkthrough video

# =============================================================================
# CORE LOGIC  (detection -> histogram -> Mean Shift -> IoU association -> draw)
# =============================================================================
def get_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("This OpenCV build has no CascadeClassifier. Install opencv-python-headless<5 (OpenCV 4.x).")
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not path.is_file():
        raise RuntimeError("Haar Cascade XML file is missing. Reinstall opencv-python-headless<5.")
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError("OpenCV could not load the Haar Cascade file.")
    return cascade


@st.cache_resource
def load_cascade():
    return get_cascade()


def detect_faces(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28))
    return [tuple(map(int, box)) for box in boxes]


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def histogram(frame, box):
    x, y, w, h = box
    roi = frame[max(0, y):max(0, y) + h, max(0, x):max(0, x) + w]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((0, 25, 25)), np.array((180, 255, 255)))
    if cv2.countNonZero(mask) == 0:
        mask = None
    hist = cv2.calcHist([hsv], [0], mask, [32], [0, 180])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def make_track(frame, box, track_id):
    hist = histogram(frame, box)
    if hist is None:
        return None
    return {"id": track_id, "box": box, "hist": hist, "missed": 0, "trail": []}


def mean_shift_tracking(frame, tracks, width, height):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for track in tracks.values():
        x, y, w, h = track["box"]
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))
        back_projection = cv2.calcBackProject([hsv], [0], track["hist"], [0, 180], 1)
        _, window = cv2.meanShift(back_projection, (x, y, w, h),
                                  (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
        nx, ny, nw, nh = map(int, window)
        if abs(nx - x) + abs(ny - y) < 90:
            track["box"] = (nx, ny, nw, nh)
            track["missed"] = 0
        else:
            track["missed"] += 1


def associate_tracks(frame, detections, tracks, next_id):
    pairs = sorted(((iou(t["box"], b), tid, di)
                    for tid, t in tracks.items() for di, b in enumerate(detections)), reverse=True)
    used_tracks, used_dets = set(), set()
    for score, tid, di in pairs:
        if score < 0.12 or tid in used_tracks or di in used_dets:
            continue
        hist = histogram(frame, detections[di])
        if hist is not None:
            tracks[tid]["box"] = detections[di]
            tracks[tid]["hist"] = hist
            tracks[tid]["missed"] = 0
            used_tracks.add(tid)
            used_dets.add(di)
    for di, box in enumerate(detections):
        if di not in used_dets:
            track = make_track(frame, box, next_id)
            if track is not None:
                tracks[next_id] = track
                next_id += 1
    return tracks, next_id


def draw_tracks(frame, tracks):
    rendered = frame.copy()
    for tid, tr in tracks.items():
        if tr["missed"]:
            continue
        x, y, w, h = tr["box"]
        color = ((53 * tid + 75) % 150 + 70, (89 * tid + 55) % 150 + 70, (37 * tid + 110) % 150 + 70)
        cv2.rectangle(rendered, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
        ly = max(20, y - 7)
        cv2.rectangle(rendered, (x, ly - 23), (x + 98, ly + 2), color, -1)
        cv2.putText(rendered, f"FACE ID {tid}", (x + 5, ly - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (9, 18, 27), 1, cv2.LINE_AA)
        tr["trail"].append((x + w // 2, y + h // 2))
        tr["trail"] = tr["trail"][-16:]
        for a, b in zip(tr["trail"][:-1], tr["trail"][1:]):
            cv2.line(rendered, a, b, (104, 230, 210), 2, cv2.LINE_AA)
    return rendered


def process_video(input_path, output_path, max_frames=500, on_frame=None):
    cascade = load_cascade()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("The video could not be opened. Try a standard MP4, MOV or AVI file.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width < 1 or height < 1:
        cap.release()
        raise RuntimeError("The video has invalid dimensions.")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create the output video.")

    tracks, next_id, idx, total_det = {}, 1, 0, 0
    try:
        while idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            mean_shift_tracking(frame, tracks, width, height)          # every frame
            detected = idx == 1 or idx % 8 == 0
            if detected:                                               # periodic detection
                dets = detect_faces(frame, cascade)
                total_det += len(dets)
                tracks, next_id = associate_tracks(frame, dets, tracks, next_id)
            tracks = {k: v for k, v in tracks.items() if v["missed"] <= 16}
            out = draw_tracks(frame, tracks)
            writer.write(out)
            if on_frame:
                on_frame(idx, min(total, max_frames) if total else 0, out,
                         sum(t["missed"] == 0 for t in tracks.values()), detected)
    finally:
        cap.release()
        writer.release()
    if idx == 0:
        raise RuntimeError("No readable frames were found in this video.")
    return {"frames": idx, "tracks": next_id - 1, "detections": total_det, "width": width, "height": height}


def to_browser_mp4(path):
    """Re-encode mp4v -> H.264 so browsers can play it (needs ffmpeg; otherwise returns the file as is)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return path
    out = path[:-4] + "_h264.mp4"
    try:
        subprocess.run([exe, "-y", "-i", path, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", out], check=True, capture_output=True, timeout=300)
        return out
    except Exception:
        return path


# =============================================================================
# STYLE
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');
html{scroll-behavior:smooth}
html, body, .stApp{font-family:'Inter',-apple-system,'Segoe UI',Roboto,sans-serif}
.stApp{background:radial-gradient(1200px 500px at 85% -5%,#dff5f1 0%,rgba(223,245,241,0) 60%),#fbf9f4}
.block-container{max-width:1060px;padding-top:2rem;padding-bottom:5rem}

/* ---------- BIGGER TEXT ---------- */
.block-container, .block-container p, .block-container li, .block-container td, .block-container th{color:#26303f}
.block-container p, .block-container li{font-size:1.2rem !important;line-height:1.85 !important}
.block-container td, .block-container th{font-size:1.08rem !important;padding:.65rem .9rem !important}
.block-container th{background:#e8f6f3;color:#10283b !important}
.block-container [data-testid=stMarkdownContainer] code{font-size:1rem;background:#e9f5f3;color:#0a6b63;padding:.12rem .4rem;border-radius:6px}
.block-container pre code{font-size:1.02rem !important;background:transparent !important;color:inherit !important}
.block-container .katex{font-size:1.25em}
.block-container h1,.block-container h2,.block-container h3{color:#10283b;font-family:'Poppins','Inter',sans-serif;font-weight:700;letter-spacing:-.01em;scroll-margin-top:5rem}
h2.sec{font-size:2.2rem;margin:3.2rem 0 1rem;padding:.2rem 0 .6rem .9rem;border-left:6px solid #0d8a80;border-bottom:2px solid #e6dfd0;border-radius:2px}
h3.sub{font-size:1.5rem;margin:2.2rem 0 .6rem;color:#0d6b64 !important}
.block-container a{color:#0d8a80;font-weight:600}

/* ---------- SIDEBAR ---------- */
[data-testid=stSidebar]{background:linear-gradient(180deg,#0b2030 0%,#0f2a3d 55%,#12445a 100%)}
[data-testid=stSidebar] *{color:#d6e2ea}
[data-testid=stSidebar] .stButton>button{background:#17384f;border:1px solid #28516c;color:#fff;border-radius:12px;font-size:1.02rem;font-weight:600;padding:.55rem .8rem;transition:all .2s}
[data-testid=stSidebar] .stButton>button:hover{border-color:#2dd4bf;color:#2dd4bf;transform:translateX(3px)}
.brand{font-family:'Poppins',sans-serif;font-size:1.7rem;font-weight:800;color:#fff;margin:.2rem 0 1rem}
.brand span{color:#2dd4bf}
.toc-title{font-size:.8rem;letter-spacing:.18em;text-transform:uppercase;color:#7fa0b5;margin:1.2rem 0 .5rem}
.toc a{display:block;text-decoration:none;color:#c3d4df !important;padding:.42rem .75rem;border-left:3px solid #25455b;font-size:1.02rem;line-height:1.35;border-radius:0 8px 8px 0;transition:all .15s}
.toc a:hover{color:#fff !important;border-left-color:#2dd4bf;background:rgba(45,212,191,.1);padding-left:1rem}
.toc a.sub{margin-left:1rem;font-size:.92rem;color:#93adbd !important}
.toc a.active{color:#2dd4bf !important;border-left-color:#2dd4bf;background:rgba(45,212,191,.16);font-weight:700}

.block-container .guide-links a{font-weight:700;font-size:1.15rem;text-decoration:none}
.block-container .guide-links a:hover{text-decoration:underline}
.st-key-go_setup button{background:transparent !important;border:none !important;box-shadow:none !important;padding:0 !important;min-height:0 !important}
.st-key-go_setup button p{color:#0d8a80 !important;font-weight:700;font-size:1.15rem !important}
.st-key-go_setup button:hover p{text-decoration:underline}

/* ---------- HERO ---------- */
.hero{position:relative;overflow:hidden;padding:3rem 2.6rem 2.5rem;border-radius:26px;background:linear-gradient(125deg,#0b2030 0%,#14506a 58%,#0d8a80 100%);color:#fff;margin-bottom:1.6rem;box-shadow:0 22px 50px rgba(15,42,61,.28)}
.hero:before{content:"";position:absolute;right:-90px;top:-90px;width:340px;height:340px;border-radius:50%;background:radial-gradient(circle,rgba(45,212,191,.45),rgba(45,212,191,0) 70%)}
.hero:after{content:"";position:absolute;left:42%;bottom:-120px;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(245,179,53,.28),rgba(245,179,53,0) 70%)}
.hero > *{position:relative;z-index:1}
.hero .kick{font-size:.9rem;letter-spacing:.18em;text-transform:uppercase;color:#9ff0e4;font-weight:700}
.hero h1{color:#fff !important;font-family:'Poppins',sans-serif;font-size:4.2rem !important;margin:.5rem 0 .7rem;line-height:1.05}
.block-container .hero p{color:#e8f6f4 !important;max-width:720px;font-size:1.25rem !important;line-height:1.75 !important;margin:0}
.pill{display:inline-block;margin:1.1rem .5rem 0 0;padding:.4rem .95rem;border:1px solid rgba(255,255,255,.4);background:rgba(255,255,255,.1);border-radius:999px;font-size:.95rem;color:#fff;backdrop-filter:blur(4px)}
.block-container a.cta{display:inline-block;margin-top:1.5rem;padding:.8rem 1.6rem;background:linear-gradient(135deg,#ffc94d,#f5a623);color:#10283b !important;border-radius:14px;font-size:1.15rem;font-weight:800;text-decoration:none;box-shadow:0 8px 22px rgba(245,166,35,.45);transition:transform .2s,box-shadow .2s;cursor:pointer}
.block-container a.cta:hover{transform:translateY(-3px);box-shadow:0 14px 28px rgba(245,166,35,.55)}

/* ---------- PIPELINE FLOW ---------- */
.flow{display:flex;flex-wrap:wrap;gap:1rem;align-items:stretch;margin:1.6rem 0}
.fbox{flex:1 1 200px;min-height:165px;background:#fff;border:1px solid #e4dccb;border-top:5px solid #0d8a80;border-radius:18px;padding:1.5rem 1.4rem;box-shadow:0 4px 14px rgba(16,40,59,.07);transition:transform .2s,box-shadow .2s}
.fbox:hover{transform:translateY(-4px);box-shadow:0 12px 26px rgba(16,40,59,.14)}
.fbox b{display:block;color:#10283b;font-size:1.35rem;margin-bottom:.2rem}
.fbox small{color:#26303f;font-size:1.1rem !important;line-height:1.55;display:block;margin-top:.4rem}
.farrow{align-self:center;color:#0d8a80;font-weight:800;font-size:2rem}
.note{background:#effaf8;border-left:5px solid #0d8a80;padding:1rem 1.2rem;border-radius:0 14px 14px 0;margin:1.2rem 0;font-size:1.1rem}
.metric{padding:1.1rem 1.2rem;border:1px solid #e4dccb;border-radius:16px;background:#fff;box-shadow:0 4px 14px rgba(16,40,59,.06)}
.metric small{display:block;color:#6b7888;font-size:.95rem}
.metric strong{font-size:1.65rem;color:#0d6b64}
table{border-collapse:collapse;width:100%;border-radius:12px;overflow:hidden;box-shadow:0 3px 12px rgba(16,40,59,.07)}
table tr:nth-child(even) td{background:#f6faf9}

/* ---------- STREAMLIT WIDGETS ---------- */
[data-testid=stFileUploader] section{border:2px dashed #0d8a80;background:#f2fbf9;border-radius:16px;padding:1.4rem}
[data-testid=stFileUploader] section:hover{background:#e6f7f4}
.stButton>button[kind=primary], .stDownloadButton>button{background:linear-gradient(135deg,#5eead4,#2dd4bf);border:none;color:#000;border-radius:14px;font-size:1.15rem;font-weight:700;padding:.75rem 1rem;box-shadow:0 8px 20px rgba(13,138,128,.35);transition:transform .2s,box-shadow .2s}
.stButton>button[kind=primary]:hover, .stDownloadButton>button:hover{transform:translateY(-2px);box-shadow:0 12px 26px rgba(13,138,128,.45);color:#000}
[data-testid=stExpander]{border:1px solid #e4dccb;border-radius:14px;background:#fff}
[data-testid=stExpander] summary p{font-size:1.15rem !important;font-weight:600}
[data-testid=stVideo], [data-testid=stImage]{border-radius:16px;overflow:hidden;box-shadow:0 12px 30px rgba(16,40,59,.2)}
[data-testid=stCode]{border-radius:12px}
[data-testid=stSlider] label p{font-size:1.1rem !important}
/* Try Your Own Video section: force dark/black text so nothing is washed out */
[data-testid=stFileUploader] *, [data-testid=stSlider] *, [data-testid=stProgress] *,
[data-testid=stAlert] *, [data-testid=stSpinner] *, [data-testid=stFileUploader] small{color:#000 !important}
[data-testid=stFileUploader] section>div span, [data-testid=stFileUploader] small{font-size:1.05rem !important;font-weight:500}
[data-testid=stFileUploader] button{background:#fff !important;border:1.5px solid #0d8a80 !important}
[data-testid=stFileUploader] button *{color:#000 !important}
.stButton>button[kind=primary] *, .stDownloadButton>button *{color:#000 !important;font-weight:800}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =============================================================================
# NAVIGATION (left index)
# =============================================================================
INDEX = [
    ("guidance", "Project Guidance", []),
    ("intro", "1. Introduction", []),
    ("demo", "2. Project Demonstration", []),
    ("try", "3. Try Your Own Video", []),
    ("pipeline", "4. Pipeline Architecture", [
        ("p1", "4.1 Video Frames"), ("p2", "4.2 Face Detection"), ("p3", "4.3 Face Region & Hue Histogram"),
        ("p4", "4.4 Track State"), ("p5", "4.5 Mean Shift Tracking"), ("p6", "4.6 IoU Association"),
        ("p7", "4.7 Track Lifecycle & Drawing")]),
    ("loop", "5. Main Loop & Frame Schedule", []),
    ("example", "6. Numerical Walk-through", []),
    ("funcs", "7. Function Reference", []),
    ("code", "8. Core Source Code", []),
    ("concl", "9. Conclusion", []),
]

NAV_JS_TEMPLATE = """
<script>
(function () {
  var TITLES = __TITLES__;
  function install() {
    if (window.__ftNav2) return;
    window.__ftNav2 = true;
    var TT = window.__ftTitles || {};
    function norm(t) { return (t || '').replace(/\\s+/g, ' ').trim(); }
    function heads() {
      return Array.prototype.slice.call(document.querySelectorAll('h1, h2, h3'));
    }
    function findTarget(id) {
      var el = document.getElementById(id);
      if (!el) el = document.querySelector('[id$="' + id + '"]');
      if (!el && TT[id]) {
        var want = norm(TT[id]), hs = heads();
        for (var i = 0; i < hs.length; i++) {
          if (norm(hs[i].textContent) === want && !hs[i].closest('[data-testid="stSidebar"]')) { el = hs[i]; break; }
        }
      }
      return el;
    }
    document.addEventListener('click', function (e) {
      var a = e.target.closest ? e.target.closest('a[href^="#"]') : null;
      if (!a) return;
      var id = a.getAttribute('href').slice(1);
      if (!id) return;
      var el = findTarget(id);
      if (!el) return;
      e.preventDefault();
      e.stopPropagation();
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
      try { history.replaceState(null, '', '#' + id); } catch (err) {}
    }, true);
    var ticking = false;
    function spy() {
      ticking = false;
      var hs = Array.prototype.slice.call(document.querySelectorAll('.block-container h2, .block-container h3'));
      if (!hs.length) return;
      var cur = hs[0];
      for (var i = 0; i < hs.length; i++) {
        if (hs[i].getBoundingClientRect().top <= 150) cur = hs[i]; else break;
      }
      var label = norm(cur.textContent);
      var links = document.querySelectorAll('.toc a');
      for (var j = 0; j < links.length; j++) {
        links[j].classList.toggle('active', norm(links[j].textContent) === label);
      }
    }
    document.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; requestAnimationFrame(spy); }
    }, true);
    setTimeout(spy, 600);
  }
  try {
    var P = window.parent;
    P.__ftTitles = TITLES;
    if (!P.__ftNav2) {
      var s = P.document.createElement('script');
      s.text = '(' + install.toString() + ')();';
      P.document.head.appendChild(s);
    }
  } catch (err) { console.log('FaceTrack nav: ' + err); }
})();
</script>
"""
_titles = {sid: t for sid, t, _ in INDEX}
for _sid, _t, _subs in INDEX:
    for _s, _tt in _subs:
        _titles[_s] = _tt
NAV_JS = NAV_JS_TEMPLATE.replace("__TITLES__", json.dumps(_titles))

if "page" not in st.session_state:
    st.session_state.page = "Report"

with st.sidebar:
    st.markdown('<div class="brand">◉ Face<span>Track</span></div>', unsafe_allow_html=True)
    if st.button("Project Report", use_container_width=True):
        st.session_state.page = "Report"
    if st.button("Setup & Installation", use_container_width=True):
        st.session_state.page = "Setup"
    if st.session_state.page == "Report":
        links = ['<div class="toc-title">Index</div><div class="toc">']
        for sid, title, subs in INDEX:
            links.append(f'<a href="#{sid}">{title}</a>')
            links += [f'<a class="sub" href="#{s}">{t}</a>' for s, t in subs]
        links.append("</div>")
        st.markdown("".join(links), unsafe_allow_html=True)

# Makes ALL "#section" links (sidebar index + hero button) scroll smoothly. It lives in the main
# page (not the sidebar) so it still runs when the sidebar is collapsed.
if st.session_state.page == "Report":
    components.html(NAV_JS, height=0)


def h2(sid, text):
    st.markdown(f'<h2 class="sec" id="{sid}">{text}</h2>', unsafe_allow_html=True)


def h3(sid, text):
    st.markdown(f'<h3 class="sub" id="{sid}">{text}</h3>', unsafe_allow_html=True)


# =============================================================================
# SETUP PAGE
# =============================================================================
def setup_page():
    st.markdown("# Setup & Installation Guide")
    st.markdown("How to run the FaceTrack website and the tracker on your own machine.")
    st.markdown("### 1. Install the packages")
    st.code("python -m pip install streamlit numpy \"opencv-python-headless<5\" imageio-ffmpeg", language="bash")
    st.markdown('<div class="note"><b>Important:</b> install OpenCV <b>4.x</b>. OpenCV 5.x builds may not include '
                '<code>cv2.CascadeClassifier</code>, which the Haar face detector needs. Do not install several '
                'OpenCV packages together.</div>', unsafe_allow_html=True)
    st.markdown("### 2. Check OpenCV")
    st.code("python -c \"import cv2; print(cv2.__version__); print(hasattr(cv2,'CascadeClassifier'))\"", language="bash")
    st.markdown("The second line must print `True`.")
    st.markdown("### 3. Run the website")
    st.code("streamlit run app.py", language="bash")
    st.markdown("### 4. Use it")
    st.markdown("Open **Project Report**, go to **3. Try Your Own Video**, upload an MP4, MOV, AVI or M4V clip and "
                "press **Detect and track faces**. Preview and download the tracked video when it finishes.")
    st.markdown("### Notes")
    st.markdown("- Only the first *N* frames are processed (default 500) to keep processing quick.\n"
                "- `imageio-ffmpeg` (or system ffmpeg) re-encodes the result to H.264 so the browser can play it. "
                "Without it, the video can still be downloaded.\n"
                "- On Streamlit Community Cloud, add `ffmpeg` to a `packages.txt` file if you want system ffmpeg.")


# =============================================================================
# REPORT PAGE
# =============================================================================
def live_demo():
    uploaded = st.file_uploader("Drop a video here", type=["mp4", "mov", "avi", "m4v"],
                                label_visibility="collapsed", key="tracking_input_video")
    max_frames = st.slider("Maximum frames to process", 50, 1000, 500, 50)
    if uploaded is None:
        return
    st.video(uploaded.getvalue())
    if not st.button("▶  Detect and track faces", type="primary", use_container_width=True):
        return
    in_path = out_path = final_path = None
    preview, progress, status = st.empty(), st.progress(0, text="Starting..."), st.empty()

    def on_frame(i, total, frame, active, detected):
        if i == 1 or i % 5 == 0:
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            stage = "Haar detection + IoU association" if detected else "Mean Shift tracking"
            status.markdown(f"Frame **{i:,}** · {active} active tracks · Stage: **{stage}**")
            if total:
                progress.progress(min(i / total, 1.0), text=f"Processing frame {i:,} of {total:,}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix or ".mp4") as tmp:
            tmp.write(uploaded.getbuffer())
            in_path = tmp.name
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        stats = process_video(in_path, out_path, max_frames, on_frame)
        progress.empty()
        status.empty()
        final_path = to_browser_mp4(out_path)
        st.success("Tracking complete")
        cols = st.columns(4)
        for col, label, value in zip(cols, ["Frames", "Track IDs", "Detections", "Resolution"],
                                     [f"{stats['frames']:,}", stats["tracks"], f"{stats['detections']:,}",
                                      f"{stats['width']} × {stats['height']}"]):
            col.markdown(f'<div class="metric"><small>{label}</small><strong>{value}</strong></div>',
                         unsafe_allow_html=True)
        data = Path(final_path).read_bytes()
        st.video(data)
        st.download_button("Download tracked video", data, file_name="facetrack_output.mp4",
                           mime="video/mp4", use_container_width=True)
    except Exception as exc:
        progress.empty()
        st.error(f"Video processing failed: {exc}")
    finally:
        for p in {in_path, out_path, final_path}:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def core_source():
    funcs = [detect_faces, iou, histogram, make_track, mean_shift_tracking, associate_tracks, draw_tracks]
    try:
        return "\n\n".join(inspect.getsource(f) for f in funcs)
    except Exception:
        return "# Source could not be read. See app.py (section: CORE LOGIC)."


def report_page():
    # ---------- hero ----------
    st.markdown(
        '<div class="hero"><div class="kick">Computer Vision · Video Analytics</div><h1>FaceTrack</h1>'
        '<p>Detect, follow and keep the identities of multiple faces across video frames using a classical '
        'pipeline: Haar Cascade detection, HSV hue histograms, Mean Shift tracking and IoU association.</p>'
        '<span class="pill">OpenCV</span><span class="pill">Haar Cascade</span><span class="pill">Mean Shift</span>'
        '<span class="pill">IoU association</span><br><a class="cta" href="#try">Try your own video →</a></div>',
        unsafe_allow_html=True)

    # ---------- guidance ----------
    st.markdown(f"*— {AUTHORS}*")
    h2("guidance", "Project Guidance")
    st.markdown(f"- **Faculty Advisor:** {FACULTY_ADVISOR}")
    st.markdown(f"*{INTERNSHIP_NOTE}*")
    if st.button("Go to Setup and Installation Guide", key="go_setup"):
        st.session_state.page = "Setup"
        st.rerun()
    st.markdown(
        f'<div class="guide-links"><a href="{GITHUB_URL}" target="_blank">Go to the Project GitHub Repository</a><br>'
        f'<a href="{REPORT_PDF_URL}" target="_blank">Get the PDF version of this Report</a></div>',
        unsafe_allow_html=True)

    # ---------- 1 ----------
    h2("intro", "1. Introduction")
    st.markdown(
        "Tracking several faces through a video means answering two questions on every frame: *where are the "
        "faces?* and *which face is which?* Running a detector on every frame is expensive and gives no identity. "
        "FaceTrack therefore combines a slow but reliable detector with a fast appearance-based tracker.")
    st.markdown(
        "A **Haar Cascade** finds faces on the first frame and then every eighth frame. For each face, an "
        "**HSV hue histogram** is stored as its appearance model. **Mean Shift** uses that histogram to move the "
        "tracking window on every frame in between. When the detector runs again, **IoU** overlap decides whether "
        "a new detection belongs to an existing track (keeping its ID) or is a new face (getting a new ID). "
        "The annotated frames are written to an output video with face IDs and motion trails.")
    st.markdown("This report explains every stage as a connected unit: the code, the variables, the mathematics, "
                "the output, and how that output becomes the input of the next stage.")

    # ---------- 2 ----------
    h2("demo", "2. Project Demonstration")
    if DEMO_VIDEO_URL:
        st.video(DEMO_VIDEO_URL)

    # ---------- 3 ----------
    h2("try", "3. Try Your Own Video")
    st.markdown("Upload a clip to detect and track faces, then preview and download the result.")
    live_demo()

    # ---------- 4 ----------
    h2("pipeline", "4. Pipeline Architecture")
    st.markdown(
        '<div class="flow">'
        '<div class="fbox"><b>Video input</b><small>read frames I<sub>t</sub> with VideoCapture</small></div><div class="farrow">→</div>'
        '<div class="fbox"><b>Face detection</b><small>grayscale + equalise, Haar Cascade → boxes</small></div><div class="farrow">→</div>'
        '<div class="fbox"><b>Hue histogram</b><small>crop ROI, HSV, 32-bin hue model</small></div><div class="farrow">→</div>'
        '<div class="fbox"><b>Mean Shift</b><small>back projection moves each window</small></div><div class="farrow">→</div>'
        '<div class="fbox"><b>IoU matching</b><small>detections ↔ tracks, new IDs</small></div><div class="farrow">→</div>'
        '<div class="fbox"><b>Draw &amp; export</b><small>boxes, IDs, trails → output video</small></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown("**Data flow:** Video file → frame I<sub>t</sub> → face boxes D<sub>t</sub> → hue histogram h<sub>i</sub> → "
                "back projection → Mean Shift window → IoU association → track state → annotated frame → output video.",
                unsafe_allow_html=True)

    h3("p1", "4.1 Video Frames")
    st.markdown("`cv2.VideoCapture` decodes the compressed file into image arrays. A video is a sequence of frames "
                "indexed by time; OpenCV stores colour in **BGR** order.")
    st.latex(r"V=\{I_1,I_2,\ldots,I_N\},\qquad I_t\in\{0,1,\ldots,255\}^{H\times W\times 3}")
    st.latex(r"I_t(x,y)=[B_t(x,y),\,G_t(x,y),\,R_t(x,y)]")
    st.markdown("`cap.read()` returns `ret` (True if a frame was decoded) and `frame`, a NumPy array of shape "
                "`(height, width, 3)`. For a 640×480 video one frame holds 640 × 480 × 3 = 921,600 values.")

    h3("p2", "4.2 Face Detection")
    st.code("gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\ngray = cv2.equalizeHist(gray)\n"
            "boxes = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28))",
            language="python")
    st.markdown("**Grayscale conversion** keeps one intensity value per pixel:")
    st.latex(r"G_t(x,y)=0.114\,B_t(x,y)+0.587\,G_t(x,y)+0.299\,R_t(x,y)")
    st.markdown("**Histogram equalisation** redistributes contrast using the cumulative distribution "
                r"$C(r)=\sum_{j\le r}p(j)$:")
    st.latex(r"G'_t(x,y)\approx 255\,C\big(G_t(x,y)\big)")
    st.markdown("**Haar Cascade** scans windows at several scales. Haar-like features compare intensity sums of "
                "neighbouring rectangles; early cascade stages cheaply reject non-face windows. "
                "`scaleFactor=1.12` is the scale step, `minNeighbors=5` the number of supporting neighbours, and "
                "`minSize=(28,28)` removes very small windows. The output is only a list of coordinates:")
    st.latex(r"D_t=\{\mathbf b_{t,1},\ldots,\mathbf b_{t,m}\},\qquad \mathbf b_{t,i}=(x_i,y_i,w_i,h_i)")

    h3("p3", "4.3 Face Region & Hue Histogram")
    st.markdown("The detector returns coordinates, not pixels, so the box is used to crop the **original colour "
                "frame** (colour is needed for the appearance model):")
    st.latex(r"ROI_{t,i}=I_t[\,y_i:y_i+h_i,\;x_i:x_i+w_i,\;:\,]")
    st.markdown("The crop is converted to **HSV**. Hue is the appearance feature; saturation and value are used "
                "only to reject unreliable pixels (very dark or nearly gray):")
    st.latex(r"M(p)=1\ \text{if}\ 25\le S(p)\le 255\ \text{and}\ 25\le V(p)\le 255,\ \text{else}\ 0")
    st.markdown("Hue (0–180 in OpenCV) is split into 32 bins of width 180/32 = 5.625, then counted and min-max "
                "normalised to 0–255:")
    st.latex(r"h(k)=\sum_{p}\mathbf 1[\text{hue}(p)\in\text{bin }k]\,M(p)")
    st.latex(r"h_n(k)=255\,\frac{h(k)-h_{\min}}{h_{\max}-h_{\min}}")
    st.markdown("If the mask removes every pixel, `mask` is set to `None` and all pixels are used. "
                "An empty crop returns `None`, so no track can be created from it.")

    h3("p4", "4.4 Track State")
    st.markdown("Each face is stored as a dictionary, the memory that links observations across time:")
    st.latex(r"T_i=(ID_i,\;\mathbf b_i,\;h_i,\;m_i,\;P_i)")
    st.markdown("`id` is the label, `box` the current rectangle, `hist` the hue histogram, `missed` the count of "
                "failed tracking updates and `trail` the recent centre points. New tracks are stored as "
                "`tracks[next_id]` and `next_id` is incremented.")

    h3("p5", "4.5 Mean Shift Tracking")
    st.markdown("**Step 1 – back projection.** The current frame is converted to HSV and every pixel looks up the "
                "stored histogram response of its hue bin, giving a probability-like image:")
    st.latex(r"B_t(x,y)=h_i\big(\text{bin}(H_t(x,y))\big)")
    st.markdown("**Step 2 – weighted centre.** Pixels inside the window are weighted by their back-projection value:")
    st.latex(r"c_x=\frac{\sum_j w_j x_j}{\sum_j w_j},\qquad c_y=\frac{\sum_j w_j y_j}{\sum_j w_j}")
    st.markdown("**Step 3 – move the window** towards the centre and repeat. It stops when the shift is small "
                "or after at most 10 iterations (epsilon = 1).")
    st.latex(r"W_{k+1}=\text{ShiftWindow}(W_k,\;\mathbf c_{k+1})")
    st.markdown("**Step 4 – acceptance test.** The Manhattan displacement of the top-left corner must be below "
                "90 pixels; otherwise the update is rejected and `missed` is increased:")
    st.latex(r"d=|x_{new}-x_{old}|+|y_{new}-y_{old}|<90")

    h3("p6", "4.6 IoU Association")
    st.markdown("Mean Shift predicts a position from appearance; the detector periodically supplies fresh positions. "
                "Intersection over Union measures how much a track box and a detection box overlap:")
    st.latex(r"\operatorname{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|},\qquad |A\cup B|=|A|+|B|-|A\cap B|")
    st.markdown("All (track, detection) pairs are sorted by IoU, highest first, and accepted greedily if the "
                "score is at least **0.12** and neither the track nor the detection is already used. "
                "A matched track takes the detector box, a freshly computed histogram, and `missed = 0`. "
                "Every unmatched detection becomes a **new track with a new ID**, so identity labels come from the "
                "tracker, not from the detector.")
    st.code("for score, track_id, di in pairs:   # sorted high -> low\n"
            "    if score < 0.12 or track_id in used_tracks or di in used_dets:\n"
            "        continue\n"
            "    # update the matched track", language="python")

    h3("p7", "4.7 Track Lifecycle & Drawing")
    st.markdown("Tracks with `missed > 16` are removed. Visible tracks (`missed == 0`) are drawn on a copy of the "
                "frame with a coloured box, a `FACE ID n` label and a trail of the last 16 centre points:")
    st.latex(r"c_x=x+\tfrac{w}{2},\qquad c_y=y+\tfrac{h}{2}")
    st.latex(r"O_t=\operatorname{Draw}(I_t,\mathcal T_t)")
    st.markdown("The annotated frame is passed to `writer.write` and, in the website, to the live preview.")

    # ---------- 5 ----------
    h2("loop", "5. Main Loop & Frame Schedule")
    st.markdown("For every frame: (1) read the frame, (2) Mean Shift updates all tracks, (3) if the frame is the "
                "first or divisible by 8, run Haar detection and IoU association, (4) remove stale tracks, "
                "(5) draw and write. Detection is the costly step, so it runs rarely; Mean Shift fills the frames "
                "in between and periodic detection corrects drift.")
    st.latex(r"\mathcal T_t=\operatorname{Update}(\mathcal T_{t-1},\mathcal D_t)")
    st.markdown(
        "| Frame | Mean Shift | Haar detection | Association |\n|---|---|---|---|\n"
        "| 1 | Yes, no prior tracks | Yes | Creates initial IDs |\n"
        "| 2, 3 | Updates existing tracks | No | None |\n"
        "| 8 | Updates existing tracks | Yes | Matches or creates IDs |\n"
        "| 9 | Updates refreshed tracks | No | None |\n"
        "| 16 | Updates existing tracks | Yes | Matches or creates IDs |")

    # ---------- 6 ----------
    h2("example", "6. Numerical Walk-through")
    st.markdown("Illustrative values, not taken from a particular video.")
    st.markdown(
        "- **Frame 1:** the detector finds `(100, 80, 60, 60)`. The 3,600-pixel crop becomes a 32-bin hue "
        "histogram and track **ID 1** is created.\n"
        "- **Frame 2:** Mean Shift returns `(102, 81, 60, 60)`. Displacement = |102−100| + |81−80| = 3 < 90, accepted.\n"
        "- **Frame 8:** the detector returns `(106, 83, 60, 60)`. Intersection = 56 × 58 = 3,248; "
        "union = 3,600 + 3,600 − 3,248 = 3,952.")
    st.latex(r"\operatorname{IoU}=\frac{3248}{3952}\approx 0.822>0.12\ \Rightarrow\ \text{assigned to ID 1}")
    st.markdown("- **A second face** appears at `(400, 100, 55, 55)` with no overlap, so it is unmatched. "
                "`make_track` creates **ID 2** and `next_id` becomes 3.")

    # ---------- 7 ----------
    h2("funcs", "7. Function Reference")
    st.markdown(
        "| Function | Receives | Returns / changes |\n|---|---|---|\n"
        "| `load_cascade` | nothing | loaded Haar Cascade classifier |\n"
        "| `detect_faces` | colour frame, classifier | list of face boxes |\n"
        "| `iou` | two boxes | one overlap score |\n"
        "| `histogram` | colour frame, one box | 32-bin normalised hue histogram or `None` |\n"
        "| `make_track` | frame, box, ID | track dictionary or `None` |\n"
        "| `mean_shift_tracking` | frame, tracks, dimensions | updates boxes and `missed` in place |\n"
        "| `associate_tracks` | frame, detections, tracks, next ID | updated tracks and next ID |\n"
        "| `draw_tracks` | frame, tracks | annotated frame |\n"
        "| `process_video` | input path, output path | writes video, returns statistics |")

    # ---------- 8 ----------
    h2("code", "8. Core Source Code")
    st.markdown("The functions below are the ones used by the live demo above.")
    with st.expander("Show core source code", expanded=False):
        st.code(core_source(), language="python")

    # ---------- 9 ----------
    h2("concl", "9. Conclusion")
    st.markdown(
        "FaceTrack is a chain of data transformations. The detector turns a grayscale frame into coordinates; "
        "the histogram function turns a box and a colour frame into an appearance vector; Mean Shift turns that "
        "vector and the next frame into a new window position; IoU turns two boxes into a matching score; "
        "association turns scores and detections into updated track states; and drawing turns those states into "
        "an annotated video.")
    st.markdown("**Design notes and limitations**")
    st.markdown(
        "- Greedy IoU matching is simple but is not guaranteed to find the globally best assignment when several "
        "faces overlap or cross.\n"
        "- Mean Shift keeps a fixed window size and relies on hue, so similar colours in the background can "
        "attract the window.\n"
        "- The Haar Cascade used here finds frontal faces.\n"
        "- Identities are labels created by the tracker; a face that is lost for more than 16 frames is removed "
        "and will receive a new ID if it returns.")
    st.markdown('<div style="text-align:center;color:#7a8696;font-size:.8rem;margin-top:2.5rem">'
                'FaceTrack · OpenCV + NumPy + Streamlit</div>', unsafe_allow_html=True)


if st.session_state.page == "Setup":
    setup_page()
else:
    report_page()