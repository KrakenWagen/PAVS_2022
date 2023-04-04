import re
import os
import time

import torch
import numpy as np

from PIL import Image
from tqdm import tqdm

from utils.dataset_loader import DatasetLoader

import pdb
import cv2

import config


# IMG_WIDTH = 256
# IMG_HIGHT = 320
# TRG_WIDTH = 32
# TRG_HIGHT = 40

device = torch.device("cuda:0")


class PredictSaliency(object):

    def __init__(self):
        super(PredictSaliency, self).__init__()

        #self.video_list = [os.path.join(config.VIDEO_TEST_FOLDER, p) for p in os.listdir(config.VIDEO_TEST_FOLDER)]
        #self.video_fps_list = [int(open(x+'/fps.txt').readline()) for x in self.video_list]
        #self.model = PAVS_KENSTEIN()
        #self.model= torch.load(config.MODEL_PATH)
        #self.output = config.PREDICTION_OUTPUT_FOLDER
        #if not os.path.exists(self.output):
        #        os.mkdir(self.output)
        #self.model = self.model.cuda()
        #self.model.eval()

    def get_datasets_loaders(self, dataset_config, dataset_groups=["TRAIN", "VALIDATION"], batch_size = config.TRAIN_BATCH_SIZE):
        print("Initializing dataset {} loaders ...".format(dataset_config.DATASET_NAME))
        starting_epoch = 0

        curr_loaders = {}
        for group in dataset_groups:
            curr_loaders[group] = DatasetLoader(dataset_config, group, batch_size=batch_size)

        return curr_loaders

    @staticmethod
    def load_model_file(model, filepath):
        checkpoint = torch.load(filepath)

        model.load_state_dict(checkpoint['model_state_dict'])

        return model

    def predict(self):
        # Load equator bias
        #equator_bias = cv2.resize(cv2.imread(config.ECB, 0), (10,8))
        equator_bias = cv2.resize(cv2.imread(config.ECB, 0), (8,4))
        equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        equator_bias = equator_bias.cuda()
        equator_bias = equator_bias/equator_bias.max()

        # Predict for each dataset in config
        for dataset_config in config.DATASETS:
            dataset_name = dataset_config.DATASET_NAME

            # Load models to use for prediction
            self.models = {}
            for model_class in config.MODELS:
                model_name = model_class.__name__
                best_model_path = config.SAVED_MODELS_ROOT + model_name + "/" + dataset_name + "/model_ep_best.pkl"
                if not os.path.exists(best_model_path):
                    print("ERROR: {} does not exist!".format(config.SAVED_MODELS_ROOT))

                print("Loading model {} best epoch state from {} ...".format(model_name, best_model_path))
                model = model_class()
                model = self.load_model_file(model, best_model_path).to(device=device)

                # Create output dir for loaded model
                output = config.OUTPUTS_ROOT+model_name+"/"+dataset_name
                if not os.path.exists(config.OUTPUTS_ROOT):
                    os.mkdir(config.OUTPUTS_ROOT)
                if not os.path.exists(output):
                    os.mkdir(output)

                self.models[model_name] = {"model": model, "output": output}

            # Initialize TEST dataset loader
            data_loaders = self.get_datasets_loaders(dataset_config, dataset_groups=["TEST"], batch_size=1)
            test_data_loader = data_loaders["TEST"]

            # Set EVAL mode for every model
            for model_name, model in self.models.items():
                model["model"].eval()

            batch_count = 0
            vit = iter(test_data_loader)
            for idx in tqdm(range(len(test_data_loader))):
                #################
                # LOAD DATA ONCE
                #################
                data_load_start = time.time()
                video_data_equi, video_data_cube, audio_data, gt_salmap, AEM_data, frames_info = next(vit)
                if len(frames_info) > 1:
                    print("ERROR: More than 1 frame loaded at once {} > 1".format(len(frames_info)))
                frame_info = frames_info[0]
                if isinstance(frame_info, list):
                    frame_info = frame_info[0]

                # Take last frame only
                #video_data_equi = video_data_equi[:, :, -1, :, :]
                video_data_equi = video_data_equi.to(device=device, dtype=torch.float)

                video_data_cube = video_data_cube.to(device=device, dtype=torch.float)

                # AEM average of depth frames
                #AEM_data = AEM_data.mean(2)
                AEM_data = AEM_data.to(device=device, dtype=torch.float)

                # audio_data = audio_data[:,:,-1,:,:]
                audio_data = audio_data.to(device=device, dtype=torch.float)

                # gt_salmap = gt_salmap[:, :, -1, :, :]
                gt_salmap = gt_salmap.to(device=device, dtype=torch.float)

                data_load_end = time.time()
                data_load_time = data_load_end - data_load_start

                ##########################
                # PREDICT FOR EVERY MODEL
                ##########################
                for model_name, model in self.models.items():
                    model_start = time.time()
                    model["model"].eval()

                    # Predict
                    pred_salmap = model["model"](video_data_equi, video_data_cube, audio_data, AEM_data, equator_bias)

                    # Clip
                    pred_salmap = torch.clip(pred_salmap, 0, 1)

                    if config.DEBUG_TRAINING_DATA:
                        gt_saliency = gt_salmap.cpu().data.numpy()
                        saliency = torch.clip(pred_salmap, 0, 1).cpu().data.numpy()
                        saliency = np.squeeze(saliency[0, :, :, :])
                        saliency = saliency / saliency.max()

                        frame_img = np.squeeze(video_data_equi[0, :, -1, :, :].permute([1, 2, 0]).cpu().data.numpy())
                        if config.NORMALIZE_FRAMES:
                            frame_img = frame_img - np.min(frame_img) / (np.max(frame_img) - np.min(frame_img))

                        cv2.imshow("{} frame".format(""), frame_img)
                        cv2.waitKey(1)
                        cv2.imshow("{} AEM".format(""), np.squeeze(AEM_data.mean(2)[0, :, :, :].cpu().data.numpy()))
                        cv2.waitKey(1)
                        cv2.imshow("{} GT".format(""), np.squeeze(gt_saliency[0, :, :, :]))
                        cv2.waitKey(1)
                        cv2.imshow("{} Pred".format(""), saliency[:, :])
                        cv2.waitKey(1)

                    # Prepare prediction for saving
                    saliency = pred_salmap.cpu().data.numpy()
                    saliency = np.squeeze(saliency)
                    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
                    saliency = Image.fromarray((saliency * 255).astype(np.uint8))
                    saliency = saliency.resize((640, 480), Image.ANTIALIAS)


                    # Save prediction to file
                    output_dir = '{}/{}/'.format(model["output"], frame_info["video_name"])
                    output_path = '{}{}.jpg'.format(output_dir, "%04d" % frame_info["frame_number"])
                    if not os.path.exists(output_dir):
                        os.mkdir(output_dir)
                    saliency.save(output_path, 'JPEG')

    def predict_sequences(self):
        self.predict()


if __name__ == '__main__':

    p = PredictSaliency()
    # predict all sequences
    p.predict_sequences()
    # alternatively one can call directy for one video
    #p.predict(VIDEO_TO_LOAD, FPS, SAVE_FOLDER) # the second argument is the video FPS.