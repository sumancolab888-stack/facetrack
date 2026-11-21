"""
FaceTrack - Face detection and tracking (standalone, no web page code)

Pipeline: Haar Cascade detection -> Mean Shift tracking -> IoU association

Run:
    python face_track.py                 -> opens a file dialog to choose a video
    python face_track.py my_video.mp4    -> uses the given video directly

Install:
    pip install opencv-python numpy

Controls during preview: press 'q' or ESC to stop early.
Output is saved next to the input video as  <name>_tracked.mp4
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

MAX_FRAMES = 500          # process at most this many frames (set None for no limit)
DETECT_EVERY = 8          # re-run Haar detection every N frames
IOU_THRESHOLD = 0.12      # minimum IoU to match a detection to a track
MAX_MISSED = 16           # delete a track after this many failed frames
MAX_JUMP = 90             # max allowed |dx|+|dy| movement per frame (pixels)


# ----------------------------------------------------------------------------
# Choose video (upload option)
# ----------------------------------------------------------------------------
def choose_video():
    """Open a file dialog so the user can pick/upload a video."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select a video to detect and track faces",
            filetypes=[("Video files", "*.mp4 *.mov *.avi *.m4v *.mkv"), ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception:
        # tkinter not available -> ask in the terminal
        return input("Enter the full path of the video file: ").strip().strip('"')


# ----------------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------------
def load_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("OpenCV was loaded without CascadeClassifier. Reinstall opencv-python.")
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        raise RuntimeError("Haar Cascade file is missing. Reinstall opencv-python.")
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError("OpenCV could not load the Haar Cascade file.")
    return cascade


def detect_faces(frame, cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(28, 28))
    return [tuple(map(int, b)) for b in boxes]


# ----------------------------------------------------------------------------
# Association helpers
# ----------------------------------------------------------------------------
def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def histogram(frame, box):
    """32-bin Hue histogram of the face region (appearance model for Mean Shift)."""
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


# ----------------------------------------------------------------------------
# Main processing
# ----------------------------------------------------------------------------
def process_video(input_path, output_path, show_preview=True):
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

    tracks, next_id, frame_index, total_detections = {}, 1, 0, 0
    limit = MAX_FRAMES if MAX_FRAMES else float("inf")

    try:
        while frame_index < limit:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # ---- 1) Mean Shift: move every existing track --------------------
            for tr in list(tracks.values()):
                x, y, w, h = tr["box"]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))

                back = cv2.calcBackProject([hsv], [0], tr["hist"], [0, 180], 1)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
                _, win = cv2.meanShift(back, (x, y, w, h), criteria)
                nx, ny, nw, nh = map(int, win)

                if abs(nx - x) + abs(ny - y) < MAX_JUMP:
                    tr["box"] = (nx, ny, nw, nh)
                    tr["missed"] = 0
                else:
                    tr["missed"] += 1

            # ---- 2) Periodic detection + IoU association ---------------------
            if frame_index == 1 or frame_index % DETECT_EVERY == 0:
                detections = detect_faces(frame, cascade)
                total_detections += len(detections)

                pairs = sorted(
                    ((iou(tr["box"], box), tid, di)
                     for tid, tr in tracks.items()
                     for di, box in enumerate(detections)),
                    reverse=True,
                )
                used_tracks, used_dets = set(), set()
                for score, tid, di in pairs:
                    if score < IOU_THRESHOLD or tid in used_tracks or di in used_dets:
                        continue
                    box = detections[di]
                    hist = histogram(frame, box)
                    if hist is not None:
                        tracks[tid]["box"] = box
                        tracks[tid]["hist"] = hist
                        tracks[tid]["missed"] = 0
                        used_tracks.add(tid)
                        used_dets.add(di)

                # unmatched detections -> new tracks
                for di, box in enumerate(detections):
                    if di not in used_dets:
                        tr = make_track(frame, box, next_id)
                        if tr is not None:
                            tracks[next_id] = tr
                            next_id += 1

            # ---- 3) Remove lost tracks ---------------------------------------
            tracks = {tid: tr for tid, tr in tracks.items() if tr["missed"] <= MAX_MISSED}

            # ---- 4) Draw boxes, IDs and trails -------------------------------
            rendered = frame.copy()
            for tid, tr in tracks.items():
                if tr["missed"]:
                    continue
                x, y, w, h = tr["box"]
                color = ((53 * tid + 75) % 150 + 70,
                         (89 * tid + 55) % 150 + 70,
                         (37 * tid + 110) % 150 + 70)
                cv2.rectangle(rendered, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
                label_y = max(20, y - 7)
                cv2.rectangle(rendered, (x, label_y - 23), (x + 98, label_y + 2), color, -1)
                cv2.putText(rendered, f"FACE ID {tid}", (x + 5, label_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (9, 18, 27), 1, cv2.LINE_AA)

                center = (x + w // 2, y + h // 2)
                tr["trail"].append(center)
                tr["trail"] = tr["trail"][-16:]
                for a, b in zip(tr["trail"][:-1], tr["trail"][1:]):
                    cv2.line(rendered, a, b, (104, 230, 210), 2, cv2.LINE_AA)

            writer.write(rendered)

            # ---- 5) Live preview + console progress --------------------------
            if show_preview:
                try:
                    cv2.imshow("FaceTrack  (press q / ESC to stop)", rendered)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("\nStopped by user.")
                        break
                except cv2.error:
                    show_preview = False  # headless OpenCV: no window available

            if frame_index == 1 or frame_index % 10 == 0:
                active = sum(t["missed"] == 0 for t in tracks.values())
                denom = min(total, MAX_FRAMES) if (total and MAX_FRAMES) else total
                pct = f"{frame_index / denom * 100:5.1f}%" if denom else ""
                print(f"\rFrame {frame_index:5d} {pct} | active tracks: {active}", end="", flush=True)
    finally:
        cap.release()
        writer.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        print()

    if frame_index == 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError("No readable frames were found in this video.")

    return {
        "frames": frame_index,
        "tracks": next_id - 1,
        "detections": total_detections,
        "width": width,
        "height": height,
    }


def main():
    # Video from command line, otherwise open the upload (file-picker) dialog
    video_path = sys.argv[1] if len(sys.argv) > 1 else choose_video()
    if not video_path:
        print("No video selected. Exiting.")
        return
    if not os.path.isfile(video_path):
        print(f"File not found: {video_path}")
        return

    src = Path(video_path)
    output_path = str(src.with_name(f"{src.stem}_tracked.mp4"))

    print(f"Input : {video_path}")
    print(f"Output: {output_path}")
    print("Detecting and tracking faces...")

    try:
        stats = process_video(video_path, output_path)
    except Exception as exc:
        print(f"Video processing failed: {exc}")
        return

    print("\nTracking complete")
    print(f"  Frames processed : {stats['frames']:,}")
    print(f"  Track IDs        : {stats['tracks']}")
    print(f"  Detections       : {stats['detections']:,}")
    print(f"  Resolution       : {stats['width']} x {stats['height']}")
    print(f"  Saved to         : {output_path}")


if __name__ == "__main__":
    main()