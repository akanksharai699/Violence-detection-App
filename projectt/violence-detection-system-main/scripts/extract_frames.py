import os
import cv2

# =========================================================
# PROJECT ROOT (ABSOLUTE PATH)
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =========================================================
# INPUT VIDEO DIRECTORIES
# =========================================================
VIOLENCE_VIDEOS_DIR = os.path.join(
    BASE_DIR, "data", "raw_videos", "violence"
)

NON_VIOLENCE_VIDEOS_DIR = os.path.join(
    BASE_DIR, "data", "raw_videos", "non_violence"
)

# =========================================================
# OUTPUT FRAMES DIRECTORIES
# =========================================================
VIOLENCE_FRAMES_DIR = os.path.join(
    BASE_DIR, "data", "frames", "violence"
)

NON_VIOLENCE_FRAMES_DIR = os.path.join(
    BASE_DIR, "data", "frames", "non_violence"
)


# =========================================================
# FRAME EXTRACTION FUNCTION
# =========================================================
def extract_frames_from_folder(video_folder, output_folder, frame_gap=5):
    """
    Extract frames from all videos in a folder.

    Args:
        video_folder (str): path to folder containing videos
        output_folder (str): path where frames will be saved
        frame_gap (int): save every Nth frame
    """

    if not os.path.exists(video_folder):
        print(f"❌ Video folder not found: {video_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)

    videos = os.listdir(video_folder)

    if len(videos) == 0:
        print(f"⚠️ No videos found in: {video_folder}")
        return

    print(f"\n📂 Processing folder: {video_folder}")
    print(f"🎥 Found {len(videos)} videos")

    for video_name in videos:
        video_path = os.path.join(video_folder, video_name)

        if not video_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            print(f"⏭️ Skipping non-video file: {video_name}")
            continue

        video_id = os.path.splitext(video_name)[0]
        save_dir = os.path.join(output_folder, video_id)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_name}")
            continue

        frame_count = 0
        saved_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_gap == 0:
                frame_path = os.path.join(
                    save_dir, f"frame_{saved_frames}.jpg"
                )
                cv2.imwrite(frame_path, frame)
                saved_frames += 1

            frame_count += 1

        cap.release()
        print(f"✅ {video_name} → {saved_frames} frames saved")

    print("\n🎉 Frame extraction completed!\n")


# =========================================================
# MAIN EXECUTION
# =========================================================
# =========================================================
# RWF-2000 FRAME EXTRACTION (TRAIN + VAL)
# =========================================================
TRAIN_VIOLENCE = os.path.join(BASE_DIR, "data", "raw_videos", "train", "violence")
TRAIN_NON_VIOLENCE = os.path.join(BASE_DIR, "data", "raw_videos", "train", "non_violence")

VAL_VIOLENCE = os.path.join(BASE_DIR, "data", "raw_videos", "val", "violence")
VAL_NON_VIOLENCE = os.path.join(BASE_DIR, "data", "raw_videos", "val", "non_violence")

TRAIN_VIOLENCE_FRAMES = os.path.join(BASE_DIR, "data", "frames", "train", "violence")
TRAIN_NON_VIOLENCE_FRAMES = os.path.join(BASE_DIR, "data", "frames", "train", "non_violence")

VAL_VIOLENCE_FRAMES = os.path.join(BASE_DIR, "data", "frames", "val", "violence")
VAL_NON_VIOLENCE_FRAMES = os.path.join(BASE_DIR, "data", "frames", "val", "non_violence")


if __name__ == "__main__":
    print("\n🚀 Extracting TRAIN frames...")
    extract_frames_from_folder(TRAIN_VIOLENCE, TRAIN_VIOLENCE_FRAMES)
    extract_frames_from_folder(TRAIN_NON_VIOLENCE, TRAIN_NON_VIOLENCE_FRAMES)

    print("\n🚀 Extracting VAL frames...")
    extract_frames_from_folder(VAL_VIOLENCE, VAL_VIOLENCE_FRAMES)
    extract_frames_from_folder(VAL_NON_VIOLENCE, VAL_NON_VIOLENCE_FRAMES)

    print("\n✅ All frames extracted successfully")

