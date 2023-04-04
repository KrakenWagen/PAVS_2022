import os

import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np

import config
from utils.dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import cv2
import pickle as pkl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Training in GPU: {}".format(torch.cuda.get_device_name(0)))

loss_function = nn.KLDivLoss()
loss_function_bce = nn.BCELoss()
nb_epoch = config.NUM_EPOCHS

class TrainSaliency(object):


    @staticmethod
    def save_model_file(model, optimizer, train_history, epoch, filepath):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_history': train_history,
            'model_epoch': epoch
        }, filepath)

        return filepath

    @staticmethod
    def load_model_file(model, optimizer, filepath):
        checkpoint = torch.load(filepath)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_history = checkpoint['train_history']

        return model, optimizer, train_history

    def __init__(self):
        super(TrainSaliency, self).__init__()

        self.datasets = {}
        for dataset in config.DATASETS:
            # print("Loading dataset {} ...".format(dataset.DATASET_NAME))
            #
            # # Loading train set list
            # train_video_list = [os.path.join(dataset.VIDEO_TRAIN_FOLDER, p) for p in os.listdir(dataset.VIDEO_TRAIN_FOLDER)]
            # train_video_list.sort()
            # train_video_fps_list = [int(open(x+'/fps.txt').readline()) for x in train_video_list]
            #
            # print("Train set:")
            # print(train_video_list)
            # print(train_video_fps_list)
            #
            # # Loading validation set list
            # validation_video_list = [os.path.join(dataset.VIDEO_VALIDATION_FOLDER, p) for p in os.listdir(dataset.VIDEO_VALIDATION_FOLDER)]
            # validation_video_list.sort()
            # validation_video_fps_list = [int(open(x+'/fps.txt').readline()) for x in validation_video_list]
            # print("Validation set:")
            # print(validation_video_list)
            # print(validation_video_fps_list)

            self.datasets[dataset.DATASET_NAME] = {"dataset_config": dataset,
                                                   #"train_list": train_video_list,
                                                   #"train_fps_list": train_video_fps_list,
                                                   #"validation_list": validation_video_list,
                                                   #"validation_fps_list": validation_video_fps_list
                                                   }

        # pdb.set_trace()

        self.models = {}
        for model_class in config.MODELS:
            model_name = model_class.__name__
            print("Loading model {} ...".format(model_name))
            model = model_class()
            model._weights_init()
            #model = model.to(device)

            # Create output dir for loaded model
            output = config.OUTPUTS_ROOT+model_name+"/"
            if not os.path.exists(config.OUTPUTS_ROOT):
                os.mkdir(config.OUTPUTS_ROOT)

            if not os.path.exists(output):
                os.mkdir(output)

            if not os.path.exists(config.SAVED_MODELS_ROOT):
                os.mkdir(config.SAVED_MODELS_ROOT)

            self.models[model_name] = {"model": model, "output": output}



        # Load equator bias
        #equator_bias = cv2.resize(cv2.imread(config.ECB, 0), (config.GT_WIDTH, config.GT_HIGHT))
        equator_bias = cv2.resize(cv2.imread(config.ECB, 0), (8, 4))
        equator_bias = torch.tensor(equator_bias).to(device, dtype=torch.float)
        self.equator_bias = equator_bias / equator_bias.max()

        #self.model=torch.load(config.TRAIN_MODEL_PATH)
        #self.model.load_state_dict(self._load_state_dict_(config.MODEL_PATH), strict=True)


        # self.model.eval()
    #
    # @staticmethod
    # def _load_state_dict_(filepath):
    #     if os.path.isfile(filepath):
    #         print("=> loading checkpoint '{}'".format(filepath))
    #         checkpoint = torch.load(filepath, map_location=device)
    #
    #         pattern = re.compile(r'module+\.*')
    #         state_dict = checkpoint['state_dict']
    #         # new_state_dict = {k : v for k, v in state_dict.items() if 'video_branch' in k}
    #         for key in list(state_dict.keys()):
    #             if 'video_branch' in key:
    #                 state_dict[key[:12] + '_cubic' + key[12:]] = state_dict[key]
    #
    #             if 'combinedEmbedding' in key:
    #                 state_dict[key[:17] + '_equi_cp' + key[17:]] = state_dict[key]
    #
    #         # pdb.set_trace()
    #         for key in list(state_dict.keys()):
    #             res = pattern.match(key)
    #             if res:
    #                 print('Y', key)
    #                 new_key = re.sub('module.', '', key)
    #                 state_dict[new_key] = state_dict[key]
    #                 del state_dict[key]
    #     return state_dict

    def run_validation_epoch(self, video_loader, loss_criterion):
        with torch.no_grad():
            return self.run_train_epoch(video_loader, loss_criterion, "validation")

    @staticmethod
    def optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def run_train_epoch(self, video_loader, loss_criterion, mode = "train"):
        equator_bias = self.equator_bias

        for model_name, model in self.models.items():
            model["model"].to(device)
            self.optimizer_to(model["optimizer"], device)

            if mode == "train":
                model["model"].train()
            else:
                model["model"].eval()

        batch_count = 0
        vit = iter(video_loader)
        for idx in range(len(video_loader)):
            data_load_start = time.time()

            video_data_equi, video_data_cube, audio_data, gt_salmap, AEM_data, frames_info = next(vit)

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

            # Train every model with loaded data
            for model_name, model in self.models.items():
                model_start = time.time()
                pred_salmap = model["model"](video_data_equi, video_data_cube, audio_data, AEM_data, equator_bias)

                # Clip
                pred_salmap = torch.clip(pred_salmap, 0, 1)

                if pred_salmap.shape[-1] != config.GT_WIDTH:
                    pred_salmap = nn.functional.interpolate(pred_salmap, (config.GT_HIGHT,config.GT_WIDTH), mode='bilinear', align_corners=True)

                loss = loss_criterion(pred_salmap, gt_salmap)
                # if pred_salmap.shape[-1] == config.GT_WIDTH and pred_salmap.shape[-2] == config.GT_WIDTH:
                #     loss = loss_criterion(np.squeeze(pred_salmap), np.squeeze(gt_salmap))
                # else:
                #     loss = loss_criterion(torch.nn.functional.interpolate(pred_salmap.squeeze().unsqueeze(0), (config.GT_HIGHT,config.GT_WIDTH), mode='bilinear', align_corners=True), gt_salmap.squeeze().unsqueeze(0))

                if mode == "train":
                    loss.backward()
                    optimizer = model["optimizer"]
                    optimizer.step()
                    optimizer.zero_grad()

                    model["epoch_train_loss"] += loss.cpu().data.numpy()
                else:
                    model["epoch_validation_loss"] += loss.cpu().data.numpy()
                model_end = time.time()

                if mode == "train":
                    model["epoch_train_time"] += model_end - model_start + data_load_time
                    model["epoch_train_data_time"] += data_load_time
                elif mode == "validation":
                    model["epoch_validation_time"] += model_end - model_start + data_load_time
                    model["epoch_validation_data_time"] += data_load_time

                if config.DEBUG_TRAINING_DATA:
                    gt_saliency = gt_salmap.cpu().data.numpy()
                    saliency = torch.clip(pred_salmap, 0, 1).cpu().data.numpy()
                    saliency = np.squeeze(saliency[0,:,:,:])
                    if saliency.max() > 0:
                        saliency = saliency / saliency.max()

                    frame_img = np.squeeze(video_data_equi[0, :, -1, :, :].permute([1, 2, 0]).cpu().data.numpy())
                    if config.NORMALIZE_FRAMES:
                        frame_img = frame_img - np.min(frame_img) / (np.max(frame_img) - np.min(frame_img))

                    cv2.imshow("{} frame".format(mode),frame_img)
                    cv2.waitKey(1)
                    cv2.imshow("{} AEM".format(mode), np.squeeze(AEM_data.mean(2)[0, :, :, :].cpu().data.numpy()))
                    cv2.waitKey(1)
                    cv2.imshow("{} GT".format(mode), np.squeeze(gt_saliency[0, :, :, :]))
                    cv2.waitKey(1)
                    cv2.imshow("{} Pred".format(mode), saliency[:, :])
                    cv2.waitKey(1)

        batch_count += len(video_loader)

        for model_name, model in self.models.items():
            model["model"].cpu()
            self.optimizer_to(model["optimizer"], torch.device("cpu"))

        return batch_count

    def train(self):
        # Loss function
        #loss_criterion = SphereMSE(config.GT_HIGHT, config.GT_WIDTH).to(device=device)
        loss_criterion = loss_function_bce
        #loss_criterion = SphereBCE(config.GT_HIGHT, config.GT_WIDTH).to(device=device)


        color_array = ["b", "orange", "green", "red", "purple"]
        taining_progress = {}
        for dataset_name, dataset in self.datasets.items():
            print("Beginning training with dataset {} ...".format(dataset_name))
            starting_epoch = 0

            train_data_loader = DatasetLoader(dataset["dataset_config"], "TRAIN")
            val_data_loader = DatasetLoader(dataset["dataset_config"], "VALIDATION")

            # Reset initial training progress for all models
            for model_name, model in self.models.items():
                print("Initializing training progress for model {} ...".format(model_name))

                # Optimizer (only requiring grad. parameters)
                optimizer = optim.Adam(model["model"].parameters(), lr=1e-4)
                #optimizer = optim.SGD(filter(lambda p: p.requires_grad, model["model"].parameters()), lr=1e-5, momentum=0.9,
                #                      weight_decay=1e-5)

                model["best_validation_loss"] = None
                model["train_history"] = {
                    "train_losses" : [],
                    "train_times" : [],
                    "validation_losses" : [],
                    "validation_times" : []
                }
                model["optimizer"] =  optimizer
                model["color"] =  color_array.pop(0)

            # RESUME TRAINING IF ENABLED
            if config.RESUME_TRAINING_EPOCH and config.RESUME_TRAINING_EPOCH >= 0:
                resume_training_epoch = config.RESUME_TRAINING_EPOCH
                print("ATTENTION! RESUMING TRAINING AFTER EPOCH {}".format(resume_training_epoch))

                # Plot losses
                plt.figure()
                plt.title('Train/Val losses')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                # plt.ylim([0,0.0002])
                legends = []

                # Set all models to training mode and reset epoch parameters
                for model_name, model in self.models.items():
                    saved_model_dir_path = config.SAVED_MODELS_ROOT + model_name + "/" + dataset_name + "/"
                    model_load_file_path = saved_model_dir_path + 'model_ep' + str(resume_training_epoch) + '.pkl'

                    print("Loading model {} from {}".format(model_name, model_load_file_path))
                    model["model"], model["optimizer"], model["train_history"] = self.load_model_file(model["model"], model["optimizer"], model_load_file_path)
                    model["best_validation_loss"] = np.min(model["train_history"]["validation_losses"])
                    best_val_loss_epoch = np.argmin(model["train_history"]["validation_losses"])
                    print("OK! RESUMED TRAINING FOR MODEL {} AFTER EPOCH {}. BEST EPOCH {} -> loss {}".format(model_name, resume_training_epoch, best_val_loss_epoch, model["best_validation_loss"]))

                    # Plot losses
                    plt.plot(model["train_history"]["train_losses"], "-", color=model["color"])
                    plt.plot(model["train_history"]["validation_losses"], "--", color=model["color"])

                    legends += ['{} train'.format(model_name), '{} val'.format(model_name)]
                plt.legend(legends, loc='upper right')
                plt.ylim([0,0.3])
                plt.show()

                starting_epoch = resume_training_epoch+1
            elif config.TRAIN_MODEL_PATH:
                model_load_file_path = config.SAVED_MODELS_ROOT + model_name + "/" + config.TRAIN_MODEL_PATH
                print("ATTENTION! STARTING MODEL PARAMETERS FROM {}".format(config.TRAIN_MODEL_PATH))
                print("Loading model {} from {}".format(model_name, model_load_file_path))
                model["model"], model["optimizer"], _nothing = self.load_model_file(model["model"],
                                                                                                  model["optimizer"],
                                                                                                  model_load_file_path)
                print("OK! STARTING TRAINING FOR MODEL {} FROM {}.".format(model_name,config.TRAIN_MODEL_PATH))

            # Start training models
            for epoch in tqdm(range(starting_epoch, nb_epoch)):
                train_start = time.time()

                # Set all models to training mode and reset epoch parameters
                for model_name, model in self.models.items():
                    model["model"].train()

                    model["epoch_train_loss"] = 0.0
                    model["epoch_train_data_time"] = 0.0
                    model["epoch_train_time"] = 0.0
                    model["epoch_validation_loss"] = 0.0
                    model["epoch_validation_data_time"] = 0.0
                    model["epoch_validation_time"] = 0.0

                # Load training data from current dataset
                video_loader = train_data_loader
                video_loader.shuffle_samples()

                # Train all models with different batches
                batch_count = self.run_train_epoch(video_loader, loss_criterion)

                training_end = time.time()

                for model_name, model in self.models.items():
                    train_loss = (model["epoch_train_loss"]) / (batch_count)
                    model["train_history"]["train_losses"].append(train_loss)
                    print()
                    print("=== Epoch {%s} Model{%s} Train Loss: {%.8f}  Train time: {%.4f}  Train data time: {%.4f}" % (
                        str(epoch), model_name, train_loss, model["epoch_train_time"], model["epoch_train_data_time"]))

                # Evaluation mode (validation)
                validation_start = time.time()

                # Set all models to training mode and reset epoch parameters
                for model_name, model in self.models.items():
                    model["model"].eval()
                    #model["epoch_validation_loss"] = 0.0
                    #model["epoch_validation_time"] = 0.0

                # Load training data from current dataset
                validation_batch_count = 0
                video_loader = val_data_loader
                video_loader.shuffle_samples()

                # Train all models with different batches
                validation_batch_count += self.run_validation_epoch(video_loader, loss_criterion)

                validation_end = time.time()
                for model_name, model in self.models.items():
                    validation_loss = (model["epoch_validation_loss"]) / (validation_batch_count)
                    model["train_history"]["validation_losses"].append(validation_loss)
                    print()
                    print("=== Epoch {%s} Model{%s} Val Loss: {%.8f}  Val time: {%.4f}  Val data time: {%.4f}" % (
                        str(epoch), model_name, validation_loss, model["epoch_validation_time"], model["epoch_validation_data_time"]))

                # Validation Finished
                end = time.time()

                for model_name, model in self.models.items():
                    train_loss = (model["epoch_train_loss"]) / (batch_count)
                    validation_loss = (model["epoch_validation_loss"]) / (validation_batch_count)

                    # Write to losses file
                    try:
                        with open(model["output"]+dataset_name+"_losses.csv", 'w' if epoch == 0 else 'a') as losses_file:
                            if epoch == 0:
                                losses_file.write('Epoch;Train_Loss;Train_Time;Train_Data_Load_Time;Validation_Loss;Validation_Time;Validation_Data_Load_Time\n')
                            losses_file.write("%s;%.8f;{%.4f};{%.4f};%.8f;{%.4f};{%.4f}\n" % (
                            str(epoch), train_loss, training_end - train_start, model["epoch_train_data_time"],
                            validation_loss, end - validation_start, model["epoch_validation_data_time"]))
                    except:
                        print("WARNING! Error saving losses file: {}".format(model["output"]+dataset_name+"_losses.csv"))

                    # Save model and optimizer files
                    if epoch % 1 == 0:
                        saved_model_dir_path = config.SAVED_MODELS_ROOT + model_name + "/"+ dataset_name+"/"
                        if not os.path.exists(config.SAVED_MODELS_ROOT + model_name):
                            os.mkdir(config.SAVED_MODELS_ROOT + model_name)
                        if not os.path.exists(saved_model_dir_path):
                            os.mkdir(saved_model_dir_path)

                        model_output_file_path = saved_model_dir_path + 'model_ep' + str(epoch) + '.pkl'
                        self.save_model_file(model["model"], model["optimizer"], model["train_history"], epoch, model_output_file_path)

                        if model["best_validation_loss"] == None or model["best_validation_loss"] > validation_loss:
                            print("ATTENTION! Saving best epoch {} model for {}".format(epoch, model_name))
                            model["best_validation_loss"] = validation_loss

                            best_model_output_file_path = saved_model_dir_path + 'model_ep_best.pkl'
                            self.save_model_file(model["model"], model["optimizer"], model["train_history"], epoch,
                                                 best_model_output_file_path)

                # Plot losses
                plt.figure()
                plt.title('Train/Val losses')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                #plt.ylim([0,0.0002])
                legends = []
                for model_name, model in self.models.items():
                    # Plot losses
                    plt.plot(model["train_history"]["train_losses"], "-", color=model["color"])
                    plt.plot(model["train_history"]["validation_losses"], "--", color=model["color"])

                    legends += ['{} train'.format(model_name), '{} val'.format(model_name)]
                plt.legend(legends, loc='upper right')
                plt.ylim([0, 0.3])
                plt.show()



if __name__ == '__main__':
    t = TrainSaliency()
    t.train()

