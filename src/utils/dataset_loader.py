# generic must imports
import math
import os
import pickle

import torch
import numpy as np
import cv2

import config

import utils.audio_params as audio_params
import librosa as sf
from utils.audio_features import waveform_to_feature

from PIL import Image
import torchvision.transforms.functional as F
from utils.equi_to_cube import Equi2Cube
import random
import time

# defined params @TODO move them to a parameter config file
e2c = Equi2Cube(128, 256, 512)  # Equi2Cube(out_w, in_h, in_w)

MEAN = [110.63666788 / 255.0, 103.16065604 / 255.0, 96.29023126 / 255.0]
STD = [38.7568578 / 255.0, 37.88248729 / 255.0, 40.02898126 / 255.0]


def adjust_len(a, b):
    # adjusts the len of two sorted lists
    al = len(a)
    bl = len(b)
    if al > bl:
        start = (al - bl) // 2
        end = bl + start
        a = a[start:end]
    if bl > al:
        a, b = adjust_len(b, a)
    return a, b


def create_data_packet(in_data, frame_number):
    n_frame = in_data.shape[0]

    frame_number = min(frame_number,
                       n_frame)  # if the frame number is larger, we just use the last sound one heard about
    starting_frame = frame_number - config.DEPTH + 1
    starting_frame = max(0, starting_frame)  # ensure we do not have any negative frames
    data_pack = in_data[starting_frame:frame_number + 1, :, :]
    n_pack = data_pack.shape[0]

    if n_pack < config.DEPTH:
        nsh = config.DEPTH - n_pack
        data_pack = np.concatenate((np.tile(data_pack[0, :, :], (nsh, 1, 1)), data_pack), axis=0)

    assert data_pack.shape[0] == config.DEPTH

    data_pack = np.tile(data_pack, (3, 1, 1, 1))

    return data_pack, frame_number


def load_wavfile(total_frame, wav_file):
    """load a wave file and retirieve the buffer ending to a given frame

    Args:
      wav_file: String path to a file, or a file-like object. The file
      is assumed to contain WAV audio data with signed 16-bit PCM samples.

      frame_number: Is the frame to be extracted as the final frame in the buffer

    Returns:
      See waveform_to_feature.
    """
    wav_data, sr = sf.load(wav_file, sr=audio_params.SAMPLE_RATE, dtype='float32')
    assert sf.get_duration(y=wav_data, sr=sr) > 1

    features = waveform_to_feature(wav_data, sr)
    features = np.resize(features, (int(total_frame), features.shape[1], features.shape[2]))

    return features


def get_wavFeature(features, frame_number):
    audio_data, valid_frame_number = create_data_packet(features, frame_number)
    return torch.from_numpy(audio_data).float(), valid_frame_number


def load_maps(file_path):
    '''
        Load the gt maps
    :param file_path: path the the map
    :return: a numpy array as floating number
    '''

    with open(file_path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('L').resize((config.GT_HIGHT, config.GT_WIDTH), resample=Image.BICUBIC)
            data = F.to_tensor(img)
    return data


def load_video_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    assert int(float(frame_name[0:-4])) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - config.DEPTH + 1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number + 1)]
    if len(frame_list) < config.DEPTH:
        nsh = config.DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)

    frames_cube = []
    frames_equi = []

    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                img = cv2.resize(cv2.imread(imgpath), (512, 256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0

                img_c = e2c.to_cube(img)
                img_cube = []
                for face in range(6):
                    img_f = F.to_tensor(img_c[face])
                    if config.NORMALIZE_FRAMES:
                        img_f = F.normalize(img_f, MEAN, STD)
                    img_cube.append(img_f)
                img_cube_data = torch.stack(img_cube)
                frames_cube.append(img_cube_data)

                img = cv2.resize(img, (config.FRAME_WIDTH, config.FRAME_HIGHT))
                img_equi = F.to_tensor(img)
                if config.NORMALIZE_FRAMES:
                    img_equi = F.normalize(img_equi, MEAN, STD)
                frames_equi.append(img_equi)

    data_cube = torch.stack(frames_cube, dim=0)
    data_equi = torch.stack(frames_equi, dim=0)
    '''
    frames = []
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:07d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                #pdb.set_trace()
                #img = img.convert('RGB')
                img = cv2.resize(cv2.imread(imgpath), (512,256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
                img = F.to_tensor(img)
                if config.NORMALIZE_FRAMES:
                    img = F.normalize(img, MEAN, STD)
                frames.append(img)
    data_equi = torch.stack(frames, dim=0)
    '''
    return data_equi.permute([1, 0, 2, 3]), data_cube.permute([1, 2, 0, 3, 4])
    # return data_cube.permute([1, 2, 0, 3, 4])


def load_AEM_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame) if end_frame != None else ["wrong_path", ("%04d" % frame_number) + ".jpg"]
    # pdb.set_trace()
    assert int(frame_name[0:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - config.DEPTH + 1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number + 1)]
    if len(frame_list) < config.DEPTH:
        nsh = config.DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)

    frames = []

    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        try:
            # img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (10, 8)) / 255.0
            if end_frame != None:
                img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (config.AEM_WIDTH, config.AEM_HIGHT)) / 255.0
            else:
                img = cv2.resize(np.ones([64,64,1], dtype=np.float), (config.AEM_WIDTH, config.AEM_HIGHT)) * 0.5
            img = F.to_tensor(img)
            frames.append(img)
        except:
            print("ERROR LOADING AEM {} {}".format(frame_path,
                                                   '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:])))
            continue

    frames = torch.stack(frames, dim=0)
    frames = frames.permute([1, 0, 2, 3])
    return frames

    # frames = np.zeros((8, 10))
    # count = 0.0
    #
    # for i in range(len(frame_list)):
    #     imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
    #     try:
    #         img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (10, 8))
    #         img = img/255.0
    #         frames = frames + img
    #         count = count + 1
    #     except:
    #         print("ERROR LOADING AEM {} {}".format(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:])))
    #         continue
    #
    # frames = frames/count
    # if frames.sum()>0:
    #     frames = frames/frames.max()
    # frames = F.to_tensor(frames)
    # #pdb.set_trace()
    # #data = torch.stack(frames, dim=0)
    # return frames


def load_gt_frames(end_frame, frame_number, valid_frame_number):
    # load video frames, process them and return a suitable tensor
    frame_path, frame_name = os.path.split(end_frame)
    assert int(frame_name[0:-4]) == frame_number
    frame_number = min(frame_number, valid_frame_number)
    start_frame_number = frame_number - config.DEPTH + 1
    start_frame_number = max(0, start_frame_number)
    frame_list = [f for f in range(start_frame_number, frame_number + 1)]
    if len(frame_list) < config.DEPTH:
        nsh = config.DEPTH - len(frame_list)
        frame_list = np.concatenate((np.tile(frame_list[0], (nsh)), frame_list), axis=0)
    frames = np.zeros((config.GT_HIGHT, config.GT_WIDTH))  # (32,64)
    count = 0.0
    for i in range(len(frame_list)):
        imgpath = os.path.join(frame_path, '{0:04d}.{1:s}'.format(frame_list[i], frame_name[-3:]))
        try:
            img = cv2.resize(cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE), (config.GT_WIDTH, config.GT_HIGHT))  # (64, 32)
            # img = cv2.GaussianBlur(img, (7,7),cv2.BORDER_DEFAULT)
            img = img / 255.0
            frames = frames + img
            count = count + 1
        except:
            print("ATTENTION! ERROR LOADING GT")
            continue

    frames = frames / count
    #frames = frames / frames.sum()
    frames = F.to_tensor(frames)
    # pdb.set_trace()
    # data = torch.stack(frames, dim=0)
    return frames

class DatasetLoader(object):
    """
        load the audio video
    """

    def load_dataset_groups(self):
        self.dataset_groups = {}
        for group in ["TRAIN", "TEST", "VALIDATION"]:
            if group == "TRAIN":
                dataset_video_path = self.dataset_config.VIDEO_TRAIN_FOLDER
            elif group == "TEST":
                dataset_video_path = self.dataset_config.VIDEO_TEST_FOLDER
            elif group == "VALIDATION":
                dataset_video_path = self.dataset_config.VIDEO_VALIDATION_FOLDER
            else:
                print("Error loading dataset group {}".format(group))

            # Loading video set list
            train_video_list = [os.path.join(dataset_video_path, p) for p in
                                os.listdir(dataset_video_path) if "augment" not in p or not config.DATASET_SKIP_AUGMENTATIONS]
            train_video_list.sort()

            # Load fps for videos
            #train_video_fps_list = [int(open(x + '/fps.txt').readline()) for x in train_video_list]

            # Save dataset groups dictionary
            self.dataset_groups[group] = [{
                "path": path,
                "name": path.split('/')[-1],
                "fps": int(open(path + '/fps.txt').readline())
            } for path in train_video_list]


    def load_video_frame_lists(self, dataset_group = "TRAIN"):
        self.video_data = {}
        #for group_name, vid_list in self.dataset_groups.items():
        vid_list = self.dataset_groups[dataset_group]
        for i, vid_dic in enumerate(vid_list):
            name = vid_dic["name"]
            path = vid_dic["path"]
            fps = vid_dic["fps"]

            # Load video frames list
            vid_listdir = os.listdir(path)
            video_frames = [os.path.join(path, f).replace("\\", "/") for f in vid_listdir
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            video_frames.sort()
            total_frames = str(len(video_frames))

            # Load GT frames list
            gt_dir = self.dataset_config.VIDEO_SALIENCY_GT_FOLDER + name
            gt_frames = [os.path.join(gt_dir, f).replace("\\", "/") for f in vid_listdir
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            gt_frames.sort()

            # Load AEM frames list
            aem_dir = self.dataset_config.VIDEO_AEM_FOLDER + name
            aem_frames = [os.path.join(aem_dir, f).replace("\\", "/") for f in vid_listdir
                            if f.endswith(('.jpg', '.jpeg', '.png'))]
            aem_frames.sort()

            # Load audio file
            audio_files = [os.path.join(path, f).replace("\\", "/") for f in vid_listdir
                              if f.endswith('.wav')]
            audio_feature_path = os.path.join(path, "audio_feature.pkl")
            if os.path.exists(audio_feature_path):
                with open(audio_feature_path, 'rb') as handle:
                    audio_data = pickle.load(handle)
            else:
                audio_data = load_wavfile(total_frames, audio_files[0])

                # Save audio features
                with open(audio_feature_path, 'wb') as handle:
                    pickle.dump(audio_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

            self.video_data[name] = {
                "frames": video_frames,
                "gt_frames": gt_frames,
                "aem_frames": aem_frames,
                "audio_file": audio_files[0],
                "audio_data": audio_data
            }

        return self.video_data

    def load_frame_samples(self, dataset_group = "TRAIN"):
        self.samples = []
        for vid_dict in self.dataset_groups[dataset_group]:
            video_name = vid_dict["name"]
            video_path = vid_dict["path"]
            video_fps = vid_dict["fps"]

            video_frames = self.video_data[video_name]["frames"]
            gt_frames = self.video_data[video_name]["gt_frames"]
            aem_frames = self.video_data[video_name]["aem_frames"]
            audio_file = self.video_data[video_name]["audio_file"]
            audio_data = self.video_data[video_name]["audio_data"]

            max_frame_num = config.MAX_TRAIN_FRAME_NUM
            start_frame_num = int(config.TRAIN_START_SECONDS_OFFSET * video_fps)

            for i, frame in enumerate(video_frames[start_frame_num:start_frame_num+max_frame_num:config.TRAIN_FRAME_STEP]):

                idx = i*config.TRAIN_FRAME_STEP+start_frame_num
                sample = {
                    "video_name": video_name,
                    "video_path": video_path,
                    "fps": video_fps,
                    "frame_number": idx,
                    "frame": frame,
                    "gtsal_frame": gt_frames[idx],
                    "aem_frame": (aem_frames[idx] if self.dataset_config.HAS_AEM else None),
                    "audio_file": audio_file,
                    "audio_data": audio_data,
                    "cached_result": None
                }

                self.samples.append(sample)

        #self.shuffle_samples()
        return self.samples

    def shuffle_samples(self):
        random.Random(1337).shuffle(self.samples)
        return self.samples

    def __init__(self, dataset_config, dataset_group = "TRAIN", batch_size = config.TRAIN_BATCH_SIZE):
        init_time = time.time()
        self.dataset_config = dataset_config
        self.batch_size = batch_size

        # Load TRAIN,VALIDATION and TEST video and frames lists
        print("LOADING DATASET GROUPS...")
        self.load_dataset_groups()

        # Load list of frames and audio data for all videos (frames, gt, AEM, audio)
        print("LOADING DATASET FRAMES...")
        self.load_video_frame_lists(dataset_group)

        # Prepare frames for sampling
        print("SETTING UP DATASET SAMPLES...")
        self.load_frame_samples(dataset_group)
        group_frames = np.sum([len(self.video_data[vid_dic["name"]]["frames"]) for vid_dic in self.dataset_groups[dataset_group]])
        end_time = time.time()
        print("Loaded {} set from {} in {} seconds: {} videos, {} frames, {} sampled frames.".format(
            dataset_group, dataset_config.DATASET_NAME, end_time - init_time, len(self.dataset_groups[dataset_group]), group_frames, len(self.samples)))
        #cnt = len(self.samples)
        self.cached_count = 0

        if config.DATASET_CACHE_RESULTS and not os.path.exists(config.DATASET_CACHE_DIR):
            os.mkdir(config.DATASET_CACHE_DIR)

    def __len__(self):
        return math.ceil(len(self.samples) / self.batch_size)

    def __getitem__(self, item):

        sample = self.samples[item*self.batch_size: item*self.batch_size + self.batch_size]

        video_data_equi_batch = []
        video_data_cube_batch = []
        audio_data_batch = []
        AEM_data_batch = []
        gt_data_batch = []

        frames_info = []

        for i in range(len(sample)):
            cached_path = config.DATASET_CACHE_DIR+'{}_{}.pkl'.format(sample[i]['video_name'], sample[i]['frame_number'])

            if config.DATASET_CACHE_RESULTS and (sample[i]['cached_result'] != None or os.path.exists(cached_path)):

                # Load data (deserialize)
                if sample[i]['cached_result'] != None:
                    file_path = sample[i]['cached_result']
                else:
                    file_path = cached_path
                    sample[i]['cached_result'] = file_path
                with open(file_path, 'rb') as handle:
                    cached_data = pickle.load(handle)

                audio_data, video_data_equi, video_data_cube, AEM_data, gt_data, frame_info = cached_data
                #audio_data, video_data_equi, AEM_data, gt_data, frame_info = cached_data

                audio_data_batch.append(audio_data)
                video_data_equi_batch.append(video_data_equi)
                video_data_cube_batch.append(video_data_cube)
                AEM_data_batch.append(AEM_data)
                gt_data_batch.append(gt_data)
                frames_info.append(frame_info)

            else:
                audio_params.EXAMPLE_HOP_SECONDS = 1 / int(sample[i]['fps'])
                audio_data, valid_frame_number = get_wavFeature(sample[i]['audio_data'], int(float(sample[i]['frame_number'])))
                audio_data_batch.append(audio_data)

                video_data_equi, video_data_cube = load_video_frames(sample[i]['frame'],
                                                                     int(float(sample[i]['frame_number'])),
                                                                     valid_frame_number)
                # video_data_cube = load_video_frames(sample[i]['frame'], int(sample[i]['frame_number']), valid_frame_number)
                video_data_equi_batch.append(video_data_equi)
                video_data_cube_batch.append(video_data_cube)

                AEM_data = load_AEM_frames(sample[i]['aem_frame'], int(sample[i]['frame_number']), valid_frame_number)
                AEM_data_batch.append(AEM_data)

                gt_data = load_gt_frames(sample[i]['gtsal_frame'], int(sample[i]['frame_number']), valid_frame_number)
                gt_data_batch.append(gt_data)

                frame_info = sample[i]
                frame_info = {"video_name": frame_info["video_name"],
                              "frame_number": frame_info["frame_number"]}
                frames_info.append(frame_info)

                if config.DATASET_CACHE_RESULTS and (config.DATASET_MAX_CACHED_RESULTS < 0 or self.cached_count < config.DATASET_MAX_CACHED_RESULTS):
                    cached_data = [audio_data, video_data_equi, video_data_cube, AEM_data, gt_data, frames_info]
                    #cached_data = [audio_data, video_data_equi, AEM_data, gt_data, frames_info]

                    # Store cached data (serialize)
                    with open(cached_path, 'wb') as handle:
                        pickle.dump(cached_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    sample[i]['cached_result'] = cached_path
                    self.cached_count += 1

        video_data_equi_batch = torch.stack(video_data_equi_batch, dim=0)  # [10, 3, 16, 256, 512]
        video_data_cube_batch = torch.stack(video_data_cube_batch, dim=0)  # [10, 6, 3, 16, 128, 128]
        audio_data_batch = torch.stack(audio_data_batch, dim=0)  # [10, 3, 16, 64, 64]
        AEM_data_batch = torch.stack(AEM_data_batch, dim=0)  # [10, 1, 8, 16]
        gt_data_batch = torch.stack(gt_data_batch, dim=0)  # [10, 1, 8, 16]

        return video_data_equi_batch, video_data_cube_batch, audio_data_batch, gt_data_batch, AEM_data_batch, frames_info
