import os
import json
import wandb
import time

import torch
import numpy as np
from tqdm import tqdm
import torch.types
import torchmetrics as tm
from spatial_inference.model import MainModel
from torch.utils.data import DataLoader
from spatial_inference.dataloader import MainDataset
import matplotlib.pyplot as plt

class ModelName(object):
    def __init__(self, config):
        self.config = config
        self.data_filedir = config["data_filedir"]
        self.cache_filedir = config["cache_filedir"]
        self.city = config["city"]
        self.year = config["year"]
        self.begin_month = config["begin_month"]
        self.end_month = config["end_month"]
        self.n_history = config["n_history"]
        self.time_interval = config["time_interval"]
        self.zoom_level = config["zoom_level"]
        self.pixels = config["pixels"]
        self.opt_opt = config["opt_opt"]
        self.savedir = config["savedir"].replace(":", "_")
        self.epoch = config["epoch"]
        self.lr = config["lr"]
        self.num_workers = config["num_workers"]
        self.subgraph_scale = config["subgraph_scale"]

        # self.device = torch.device("cuda:0")
        self.device = torch.device(config["device"])
        self.dataset = MainDataset(
            self.data_filedir, self.city, self.cache_filedir,
            year=self.year, begin_month=self.begin_month, end_month=self.end_month,
            n_history=self.n_history, time_interval=self.time_interval,
            zoom_level=self.zoom_level, pixels=self.pixels,
            write_cache=True, use_cache=True, num_workers=self.num_workers, subgraph_scale=self.subgraph_scale, shuffle=True)

        self.dataloader = None
        self.model = MainModel(config).to(self.device)

        if self.opt_opt == 1:
            self.criterion = torch.nn.L1Loss()
        elif self.opt_opt == 2:
            self.criterion = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # save config
        os.makedirs(self.savedir)
        with open(os.path.join(self.savedir, "config.json"), "w") as f:
            json.dump(config, f)

    def save(self, train_losses, test_losses, iteration):
        savepath = os.path.join(self.savedir, "iteration%03d" % iteration)
        os.makedirs(savepath)
        with open(os.path.join(savepath, "train_loss.json"), "w") as f:
            json.dump(train_losses, f)
        with open(os.path.join(savepath, "test_loss.json"), "w") as f:
            json.dump(test_losses, f)

        torch.save(self.model, os.path.join(savepath, "model.pt"))

    def save_iteration(self, losses, epoch, iteration):
        savepath = os.path.join(self.savedir, "epoch_%03d_iteration%06d" % (epoch, iteration))
        os.makedirs(savepath)
        # torch.save({
        #     'model_state_dict': self.model.state_dict(),
        #     'optimizer_state_dict': self.optimizer.state_dict(),
        #     "losses": losses
        # }, os.path.join(savepath, "model.pt"))

    def run(self):
        wandb.init(entity="zijianhu98", config=self.config, project="flow_estimation")

        mse = tm.MeanSquaredError()
        mae = tm.MeanAbsoluteError()
        mape = tm.MeanAbsolutePercentageError()
        smape = tm.SymmetricMeanAbsolutePercentageError()
        min_test_losses = {"best_SMAPE": [], "best_epoch": []}
        for i in range(self.epoch):
            train_losses, test_losses = {"MSE": [], "MAE": [], "MAPE": [], "SMAPE": []}, {"MSE": [], "MAE": [], "MAPE": [], "SMAPE": []}
            # wandb.watch(self.model)
            # train
            print("Training...")
            start_time = time.time()
            self.model.train()
            self.dataset.is_training = True
            self.dataloader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers)

            for j, batch_data in enumerate(tqdm(self.dataloader)):
                start_date, graph_data, pop_images, poi_images, point_images, disk_keys, _ = batch_data

                y_true = graph_data.y.to(self.device)
                y_pred = self.model(graph_data.to(self.device),
                                    poi_images.to(self.device),
                                    point_images.to(self.device),
                                    pop_images.to(self.device)).squeeze(0)
                loss = self.criterion(y_pred, y_true)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                y_pred = y_pred.detach().cpu()
                y_true = y_true.detach().cpu()
                mse_loss = mse(y_true, y_pred).detach().cpu().numpy().tolist()
                mae_loss = mae(y_true, y_pred).detach().cpu().numpy().tolist()
                mape_loss = mape(y_true, y_pred).detach().cpu().numpy().tolist()
                smape_loss = smape(y_true, y_pred).detach().cpu().numpy().tolist()
                train_losses["MSE"].append(mse_loss)
                train_losses["MAE"].append(mae_loss)
                train_losses["MAPE"].append(mape_loss)
                train_losses["SMAPE"].append(smape_loss)
                if j % 100 == 0 and j != 0:
                    wandb.log({"train_loss_mse": np.mean(train_losses["MSE"][j - 100: j])})
                    wandb.log({"train_loss_mae": np.mean(train_losses["MAE"][j - 100: j])})
                    wandb.log({"train_loss_mape": np.mean(train_losses["MAPE"][j - 100: j])})
                    wandb.log({"train_loss_smape": np.mean(train_losses["SMAPE"][j - 100: j])})

                # if j % 50000 == 0 and j != 0:
                #     try:
                #         self.save_iteration(train_losses, i, j)
                #     except:
                #         pass

            # train summary
            print("Training, epoch: %d, MSE: %.4f, MAE: %.4f, MAPE: %.4f, SMAPE: %.4f" % (i,
                 np.mean(train_losses["MSE"]),
                 np.mean(train_losses["MAE"]),
                 np.mean(train_losses["MAPE"]),
                 np.mean(train_losses["SMAPE"]),
            ))
            end_time = time.time()
            print("Training time: ", end_time - start_time)

            # test
            print("Testing...")
            start_time = time.time()
            self.model.eval()
            self.dataset.is_training = False
            self.dataloader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers)
            with torch.no_grad():
                for j, batch_data in enumerate(tqdm(self.dataloader)):
                    start_date, graph_data, pop_images, poi_images, point_images, disk_keys, test_node_indices = batch_data

                    y_true = graph_data.y.detach().cpu()[test_node_indices]
                    y_pred = self.model(graph_data.to(self.device),
                                        poi_images.to(self.device),
                                        point_images.to(self.device),
                                        pop_images.to(self.device)).squeeze(0).detach().cpu()[test_node_indices]

                    mse_loss = mse(y_true, y_pred).detach().cpu().numpy().tolist()
                    mae_loss = mae(y_true, y_pred).detach().cpu().numpy().tolist()
                    mape_loss = mape(y_true, y_pred).detach().cpu().numpy().tolist()
                    smape_loss = smape(y_true, y_pred).detach().cpu().numpy().tolist()
                    test_losses["MSE"].append(mse_loss)
                    test_losses["MAE"].append(mae_loss)
                    test_losses["MAPE"].append(mape_loss)
                    test_losses["SMAPE"].append(smape_loss)

            if i == 0:
                min_test_losses["best_SMAPE"].append(np.mean(test_losses["SMAPE"]))
                min_test_losses["best_epoch"].append(0)
            else:
                min_test_losses["best_SMAPE"].append(np.mean(test_losses["SMAPE"]) if np.mean(test_losses["SMAPE"]) < min_test_losses["best_SMAPE"][-1]
                                                 else min_test_losses["best_SMAPE"][-1])

                min_test_losses["best_epoch"].append(i if np.mean(test_losses["SMAPE"]) < min_test_losses["best_SMAPE"][-1]
                                                 else min_test_losses["best_epoch"][-1])

            wandb.log({"test_loss_mse": np.mean(test_losses["MSE"])})
            wandb.log({"test_loss_mae": np.mean(test_losses["MAE"])})
            wandb.log({"test_loss_mape": np.mean(test_losses["MAPE"])})
            wandb.log({"test_loss_smape": np.mean(test_losses["SMAPE"])})
            wandb.log({"test_loss_best_smape": min_test_losses["best_SMAPE"][-1]})
            wandb.log({"test_loss_best_epoch": min_test_losses["best_epoch"][-1]})

            # test summary
            print("Testing, epoch: %d, MSE: %.4f, MAE: %.4f, MAPE: %.4f, SMAPE: %.4f" % (
                i,
                np.mean(test_losses["MSE"]),
                np.mean(test_losses["MAE"]),
                np.mean(test_losses["MAPE"]),
                np.mean(test_losses["SMAPE"]),
            ))
            end_time = time.time()
            print(end_time - start_time)
            self.save(train_losses, test_losses, i)

    def show(self):
        self.dataset = MainDataset(
            self.data_filedir, self.city, self.cache_filedir,
            year=self.year, begin_month=self.begin_month, end_month=self.end_month,
            n_history=self.n_history, time_interval=self.time_interval,
            zoom_level=self.zoom_level, pixels=self.pixels,
            write_cache=True, use_cache=True, num_workers=self.num_workers, subgraph_scale=self.subgraph_scale,
            shuffle=False)

        for i in range(self.epoch):
            self.model = torch.load(
                "/home/zijian-va324/Workspaces/FlowEstimation/results/%s/iteration%03d/model.pt" % (self.city, i)).to(
                self.device)
            # self.model = torch.load(
            #     "/home/zijian/Workspaces/FlowEstimation/results/%s/iteration%03d/model.pt" % (self.city, i)).to(
            #     self.device)
            self.model.eval()
            self.dataset.is_training = False
            self.dataloader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers)
            res_true = []
            res_pred = []

            with torch.no_grad():
                for j, batch_data in enumerate(tqdm(self.dataloader)):
                    start_date, graph_data, pop_images, poi_images, point_images, disk_keys, test_node_indices = batch_data

                    y_true = graph_data.y.detach().cpu()[test_node_indices]
                    y_pred = self.model(graph_data.to(self.device),
                                        poi_images.to(self.device),
                                        point_images.to(self.device),
                                        pop_images.to(self.device)).squeeze(0).detach().cpu()[test_node_indices]
                    res_pred.append(y_pred.numpy())
                    res_true.append(y_true.numpy())
                    # print(start_date)
                    # if j > 300:
                    #     break

            res_pred = np.r_[res_pred]
            res_true = np.r_[res_true]
            print(np.shape(res_pred), np.shape(res_true))
            # plt.plot(res_pred[:, 0])
            # plt.plot(res_true[:, 0])
            np.save("/home/zijian-va324/Workspaces/FlowEstimation/results/%s/iteration%03d/y_pred_new.npy" % (self.city, i),
                    res_pred)
            np.save(
                "/home/zijian-va324/Workspaces/FlowEstimation/results/%s/iteration%03d/y_true_new.npy" % (self.city, i),
                res_true)
            # np.save(
            #     "/home/zijian/Workspaces/FlowEstimation/results/%s/iteration%03d/y_pred.npy" % (self.city, i),
            #         res_pred)
            # np.save(
            #     "/home/zijian/Workspaces/FlowEstimation/results/%s/iteration%03d/y_true.npy" % (self.city, i),
            #     res_true)
            # plt.show()

        # res_pred = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-18T22:43:30.452237/iteration029/y_pred.npy")
        # res_true = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-18T22:43:30.452237/iteration029/y_true.npy")

        # res_pred = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-19T21:58:36.081666/iteration029/y_pred.npy")
        # res_true = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-19T21:58:36.081666/iteration029/y_true.npy")
        # for i in range(40):
        #     print(i)
        #     plt.plot(res_pred[:, i])
        #     plt.plot(res_true[:, i])
        #     plt.show()
        #
        #     plt.savefig("/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-18T22:43:30.452237/iteration029/%02d.jpg" % i)
        #     plt.close()

    def show_static_flow(self, epoch):
        self.dataset = MainDataset(
            self.data_filedir, self.city, self.cache_filedir,
            year=self.year, begin_month=self.begin_month, end_month=self.end_month,
            n_history=self.n_history, time_interval=self.time_interval,
            zoom_level=self.zoom_level, pixels=self.pixels,
            write_cache=True, use_cache=True, num_workers=self.num_workers, subgraph_scale=self.subgraph_scale,
            shuffle=False)

        self.model = torch.load(
            r"G:\va324\home\Workspaces\FlowEstimation\results\%s\iteration%03d\model.pt" % ("new_"+ self.city, epoch)).to(
            self.device)


        self.model.eval()
        self.dataset.is_training = False
        self.dataloader = DataLoader(self.dataset, batch_size=None, num_workers=self.num_workers)
        res_true = []
        res_pred = []

        with torch.no_grad():
            for j, batch_data in enumerate(tqdm(self.dataloader)):
                start_date, graph_data, pop_images, poi_images, point_images, disk_keys, test_node_indices = batch_data

                y_true = graph_data.y.detach().cpu()[test_node_indices]
                y_pred = self.model(graph_data.to(self.device),
                                    poi_images.to(self.device),
                                    point_images.to(self.device),
                                    pop_images.to(self.device)).squeeze(0).detach().cpu()[test_node_indices]
                res_pred.append(y_pred.numpy())
                res_true.append(y_true.numpy())
                # print(start_date)
                # if j > 300:
                #     break

        res_pred = np.r_[res_pred]
        res_true = np.r_[res_true]
        print(np.shape(res_pred), np.shape(res_true))
        print(self.city, epoch)
        # plt.plot(res_pred[:, 0])
        # plt.plot(res_true[:, 0])
        np.save(
            "/home/zijian-va324/Workspaces/FlowEstimation/results/%s/iteration%03d/y_pred_new.npy" % ("new_"+ self.city, epoch),
            res_pred)
        np.save(
            "/home/zijian-va324/Workspaces/FlowEstimation/results/%s/iteration%03d/y_true_new.npy" % ("new_"+ self.city, epoch),
            res_true)
        # np.save(
        #     "/home/zijian/Workspaces/FlowEstimation/results/%s/iteration%03d/y_pred.npy" % (self.city, i),
        #         res_pred)
        # np.save(
        #     "/home/zijian/Workspaces/FlowEstimation/results/%s/iteration%03d/y_true.npy" % (self.city, i),
        #     res_true)
        # plt.show()

        # res_pred = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-18T22:43:30.452237/iteration029/y_pred.npy")
        # res_true = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-18T22:43:30.452237/iteration029/y_true.npy")

        # res_pred = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-19T21:58:36.081666/iteration029/y_pred.npy")
        # res_true = np.load(
        #     "/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-19T21:58:36.081666/iteration029/y_true.npy")
        # for i in range(40):
        #     print(i)
        #     plt.plot(res_pred[:, i])
        #     plt.plot(res_true[:, i])
        #     plt.show()
        #
        #     plt.savefig("/home/zijian-va324/Workspaces/FlowEstimation/results/2023-02-18T22:43:30.452237/iteration029/%02d.jpg" % i)
        #     plt.close()
