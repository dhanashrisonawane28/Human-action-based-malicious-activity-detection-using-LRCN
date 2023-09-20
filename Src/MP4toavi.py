import os
import subprocess

input_directory = "C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/First_Dataset/Coughing1"
output_directory = "C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/Secound_Dataset/Coughing"  # Create this directory if it doesn't exist

# List all MP4 files in the input directory
mp4_files = [f for f in os.listdir(input_directory) if f.lower().endswith(".mp4")]

# Loop through each MP4 file and convert it to AVI
for mp4_file in mp4_files:
    input_path = os.path.join(input_directory, mp4_file)
    output_file = os.path.splitext(mp4_file)[0] + ".avi"
    output_path = os.path.join(output_directory, output_file)

    # Construct the ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-c:v", "mpeg4",
        "-c:a", "pcm_s16le",
        output_path
    ]

    # Run the ffmpeg command
    subprocess.run(cmd)

print("Conversion complete.")
