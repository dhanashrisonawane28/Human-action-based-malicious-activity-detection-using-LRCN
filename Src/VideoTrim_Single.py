import os
from moviepy.video.io.VideoFileClip import VideoFileClip


def split_and_trim_single_video(input_path, output_directory, segment_duration=30):
    try:
        clip = VideoFileClip(input_path)

        num_segments = int(clip.duration / segment_duration)
        remaining_duration = clip.duration % segment_duration

        if remaining_duration > 0:
            num_segments += 1

        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min(start_time + segment_duration, clip.duration)
            segment_clip = clip.subclip(start_time, end_time)
            output_path = os.path.join(output_directory, f"video_{os.path.basename(input_path)[:-4]}_{i + 1}.mp4")
            segment_clip.write_videofile(output_path)
            segment_clip.close()
            print(f"Segment {i + 1} of {os.path.basename(input_path)} saved as {output_path}")

        clip.close()
    except Exception as e:
        print(f"Error processing {input_path}: {e}")


def main():
    input_path = r'C:\Users\Vishal\OneDrive\Desktop\VIT_SEM_3\EDI\Raw Dataset\Anomaly-Videos-Part-1\Fighting\Fighting050_x264.mp4'
    output_directory = 'C:/Users/Vishal/PycharmProjects/AnamolyDetection/DataSet/Fighting'  # Replace with your output directory
    segment_duration = 50  # Duration for each video segment in seconds

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    split_and_trim_single_video(input_path, output_directory, segment_duration)


if __name__ == "__main__":
    main()
