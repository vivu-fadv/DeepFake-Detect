import csv
import os
import cv2
import math

base_path = '.\\train_sample_videos\\'
videos_path = os.path.join(base_path, 'Deepfakes')

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

with open(os.path.join(base_path, 'csv', 'Deepfakes.csv'), newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    metadata = {}
    for row in reader:
        metadata[row['File Path']] = row['Label'].strip().upper()
    print(len(metadata))

for filename in metadata.keys():
    video_basename = os.path.basename(filename)
    print(video_basename)
    if (video_basename.endswith(".mp4")):
        tmp_path = os.path.join(videos_path, get_filename_only(video_basename))
        print('Creating Directory: ' + tmp_path)
        os.makedirs(tmp_path, exist_ok=True)
        print('Converting Video to Images...')
        count = 0
        video_file = os.path.join(videos_path, video_basename)
        cap = cv2.VideoCapture(video_file)
        frame_rate = cap.get(5) #frame rate
        while(cap.isOpened()):
            frame_id = cap.get(1) #current frame number
            ret, frame = cap.read()
            if (ret != True):
                break
            if (frame_id % math.floor(frame_rate) == 0):
                print('Original Dimensions: ', frame.shape)
                if frame.shape[1] < 300:
                    scale_ratio = 2
                elif frame.shape[1] > 1900:
                    scale_ratio = 0.33
                elif frame.shape[1] > 1000 and frame.shape[1] <= 1900 :
                    scale_ratio = 0.5
                else:
                    scale_ratio = 1
                print('Scale Ratio: ', scale_ratio)

                width = int(frame.shape[1] * scale_ratio)
                height = int(frame.shape[0] * scale_ratio)
                dim = (width, height)
                new_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                print('Resized Dimensions: ', new_frame.shape)

                new_filename = '{}-{:03d}.png'.format(os.path.join(tmp_path, get_filename_only(filename)), count)
                count = count + 1
                cv2.imwrite(new_filename, new_frame)
        cap.release()
        print("Done!")
    else:
        continue
