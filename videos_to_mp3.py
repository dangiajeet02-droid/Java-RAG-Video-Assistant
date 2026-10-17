import os
import subprocess

files = os.listdir("videos")

for file in files:
    # Sirf .mp4 video files ko process karein
    if file.endswith(".mp4"):
        # Extention (.mp4) ko hata kar name alag karein (e.g., '2nd C java')
        file_name = os.path.splitext(file)[0]
        
        print(f"Processing: {file_name}")
        
        # ffmpeg se video ko audio (mp3) me convert karein
        subprocess.run([
            "ffmpeg", 
            "-i", f"videos/{file}", 
            f"audios/{file_name}.mp3"
        ])
        
