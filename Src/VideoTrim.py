import os
import subprocess

def trim_videos_in_directory(input_directory, output_directory, segment_duration=7, min_segment_duration=5):
    try:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        for filename in os.listdir(input_directory):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                input_path = os.path.join(input_directory, filename)
                output_base = os.path.splitext(filename)[0]

                duration = float(subprocess.check_output(['ffprobe', '-i', input_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")], universal_newlines=True))

                num_segments = int(duration / segment_duration)
                remaining_duration = duration % segment_duration

                if remaining_duration > min_segment_duration:
                    num_segments += 1

                for i in range(num_segments):
                    start_time = i * segment_duration
                    end_time = start_time + min(segment_duration, duration - start_time)

                    output_path = os.path.join(output_directory, f"{output_base}_segment_{i+1}.mp4")

                    cmd = [
                        'ffmpeg', '-i', input_path,
                        '-ss', str(start_time), '-t', str(end_time - start_time),
                        '-c', 'copy', output_path
                    ]

                    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                    print(f"Segment {i+1} of {filename} saved as {output_path}")

    except Exception as e:
        print(f"Error processing videos: {e}")

def main():
    # input_directory = 'C:/Users/Vishal/OneDrive/Pictures/Camera Roll/WIN_20230830_23_15_56_Pro.mp4'  # Replace with your input directory
    # output_directory = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/Secound_Dataset_Uplode/Regular_Activities'  # Replace with your output directory
    # segment_duration = 30  # Duration for each video segment in seconds
    # min_segment_duration = 5  # Minimum duration for each segment in seconds
    #
    # if not os.path.exists(output_directory):
    #     os.makedirs(output_directory)
    #
    # trim_videos_in_directory(input_directory, output_directory, segment_duration, min_segment_duration)

    input_file = 'C:/Users/Vishal/OneDrive/Pictures/Camera Roll/WIN_20230830_23_15_56_Pro.mp4'  # Replace with your input video file
    output_base = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/Secound_Dataset_Uplode/Regular_Activities'  # Base name for output segments

    segment_duration = 6  # Duration of each segment

    # Get the total duration of the input video using ffprobe
    duration = float(subprocess.check_output(
        ['ffprobe', '-i', input_file, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")],
        universal_newlines=True))

    num_segments = int(duration / segment_duration)

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = start_time + segment_duration

        output_path = f"{output_base}_{i + 1}.mp4"

        cmd = [
            'ffmpeg', '-i', input_file,
            '-ss', str(start_time), '-t', str(segment_duration),
            '-c', 'copy', output_path
        ]

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print(f"Segment {i + 1} saved as {output_path}")

    print("Trimming complete.")

if __name__ == "__main__":
    main()














# import os
# from moviepy.video.io.VideoFileClip import VideoFileClip
#
#
# def split_and_trim_video(input_path, output_directory, segment_duration=30):
#     try:
#         clip = VideoFileClip(input_path)
#
#         num_segments = int(clip.duration / segment_duration)
#         remaining_duration = clip.duration % segment_duration
#
#         if remaining_duration > 0:
#             num_segments += 1
#
#         for i in range(num_segments):
#             start_time = i * segment_duration
#             end_time = min(start_time + segment_duration, clip.duration)
#             segment_clip = clip.subclip(start_time, end_time)
#             output_path = os.path.join(output_directory, f"Fighting_{os.path.basename(input_path)[:-4]}_{i + 1}.mp4")
#             segment_clip.write_videofile(output_path)
#             segment_clip.close()
#             print(f"Segment {i + 1} of {os.path.basename(input_path)} saved as {output_path}")
#
#         clip.close()
#     except Exception as e:
#         print(f"Error processing {input_path}: {e}")
#
#
# def main():
#     input_directory = 'C:/Users/Vishal/OneDrive/Desktop/VIT_SEM_3/EDI/Raw Dataset/Anomaly-Videos-Part-1/RoadAccidents'  # Replace with your input directory
#     output_directory = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/RoadAccidents'  # Replace with your output directory
#     segment_duration = 30  # Duration for each video segment in seconds
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     for filename in os.listdir(input_directory):
#         if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
#             input_path = os.path.join(input_directory, filename)
#             split_and_trim_video(input_path, output_directory, segment_duration)
#
#
# if __name__ == "__main__":
#     main()

# import os
# from moviepy.video.io.VideoFileClip import VideoFileClip
#
#
# def trim_video(input_path, output_directory, segment_duration=30, min_segment_duration=5):
#     try:
#         clip = VideoFileClip(input_path)
#         duration = clip.duration
#
#         num_segments = int(duration / segment_duration)
#         remaining_duration = duration % segment_duration
#
#         if remaining_duration > 0:
#             num_segments += 1
#
#         for i in range(num_segments):
#             start_time = i * segment_duration
#             end_time = min(start_time + segment_duration, duration)
#             segment_clip = clip.subclip(start_time, end_time)
#
#             # If the trimmed segment is less than min_segment_duration, trim it further
#             if segment_clip.duration < min_segment_duration:
#                 start_time = (i + 1) * segment_duration - min_segment_duration
#                 end_time = (i + 1) * segment_duration
#                 segment_clip = clip.subclip(start_time, end_time)
#
#             output_path = os.path.join(output_directory, f"trimmed_segment_{i + 1}.mp4")
#             segment_clip.write_videofile(output_path)
#             segment_clip.close()
#             print(f"Segment {i + 1} saved as {output_path}")
#
#         clip.close()
#     except Exception as e:
#         print(f"Error processing {input_path}: {e}")
#
#
# def main():
#     input_path = r'C:\Users\Vishal\OneDrive\Desktop\VIT_SEM_3\EDI\Raw Dataset\Anomaly-Videos-Part-1\RoadAccidents\RoadAccidents147_x264.mp4'  # Replace with your input video path
#     output_directory = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/RoadAccidents'
#     segment_duration = 30  # Duration for each video segment in seconds
#     min_segment_duration = 5  # Minimum duration for each segment in seconds
#
#     if not os.path.exists(output_directory):
#         os.makedirs(output_directory)
#
#     trim_video(input_path, output_directory, segment_duration, min_segment_duration)
#
#
# if __name__ == "__main__":
#     main()


# import os
# import subprocess
#
#
# def trim_video_ffmpeg(input_path, output_directory, segment_duration=30, min_segment_duration=5):
#     try:
#         if not os.path.exists(output_directory):
#             os.makedirs(output_directory)
#
#         output_base = os.path.splitext(os.path.basename(input_path))[0]
#
#         duration = float(subprocess.check_output(
#             ['ffprobe', '-i', input_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=%s' % ("p=0")],
#             universal_newlines=True))
#
#         num_segments = int(duration / segment_duration)
#         remaining_duration = duration % segment_duration
#
#         if remaining_duration > 0:
#             num_segments += 1
#
#         for i in range(num_segments):
#             start_time = i * segment_duration
#             end_time = start_time + min(segment_duration, remaining_duration)
#
#             output_path = os.path.join(output_directory, f"segment_{i + 1}.mp4")
#
#             cmd = [
#                 'ffmpeg', '-i', input_path,
#                 '-ss', str(start_time), '-t', str(end_time - start_time),
#                 '-c', 'copy', output_path
#             ]
#
#             subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#
#             print(f"Segment {i + 1} saved as {output_path}")
#
#     except Exception as e:
#         print(f"Error processing {input_path}: {e}")
#
#
# def main():
#     input_path = r'C:\Users\Vishal\OneDrive\Desktop\VIT_SEM_3\EDI\Raw Dataset\Anomaly-Videos-Part-1\RoadAccidents\RoadAccidents147_x264.mp4'  # Replace with your input video path
#     output_directory = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/RoadAccidents'
#     segment_duration = 30  # Duration for each video segment in seconds
#     min_segment_duration = 5  # Minimum duration for each segment in seconds
#
#     trim_video_ffmpeg(input_path, output_directory, segment_duration, min_segment_duration)
#
#
# if __name__ == "__main__":
#     main()

