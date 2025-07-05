import json
import os
import time
import torch
import pickle
import asyncio
import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime as dt
import torch_geometric as tg
from torchvision import transforms
from torchvision.io import read_image
from torch_geometric.data import Data
from torch.utils.data import Dataset, DataLoader


class MainDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_filedir, city, cache_filedir,
                 year=2021, begin_month=1, end_month=1, n_history=12, time_interval=5, zoom_level=15, pixels=1024,
                 write_cache=True, use_cache=True, num_workers=2, subgraph_scale=100, shuffle=True):
        super(MainDataset).__init__()
        self.graph_filepath = os.path.join(
            data_filedir, "%s" % city, "data/%04d" % year, "%02d_%02d" % (begin_month, end_month), "data.pickle")
        self.test_node_filepath = os.path.join(
            data_filedir, "%s" % city, "data/%04d" % year, "%02d_%02d" % (begin_month, end_month), "test_nodes_weak.json")
        self.neighbor_filepath = os.path.join(
            data_filedir, "%s" % city, "neighbors", "neighbor.pickle")
        self.cache_filedir = cache_filedir

        self.point_dir = os.path.join(data_filedir, "%s" % city, "point_maps")
        self.poi_dir = os.path.join(data_filedir, "%s" % city, "poi_maps")
        self.pop_dir = os.path.join(data_filedir, "%s" % city, "pop_maps")

        self.city = city
        self.year = year
        self.begin_month = begin_month
        self.end_month = end_month
        self.n_history = n_history
        self.time_interval = time_interval
        self.zoom_level = zoom_level
        self.pixels = pixels

        self.write_cache = write_cache
        self.use_cache = use_cache
        self.worker_num = num_workers

        self.graph = {}
        self.test_nodes = None
        self.neighbors = None
        self.point_images = None
        self.poi_images = None
        self.pop_images = None
        self.indices = []

        self.sorted_node_names = None
        self.sorted_node_lookups = None
        self.data_buffer = None
        self.y_buffer = None
        self.edge_index_buffer = None
        self.edge_attr_buffer = None

        self.grayscale_transform = transforms.Compose([transforms.ToPILImage(),
                                                       transforms.Resize([256, 256]),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize((0.5), (0.5)),
                                                       ])
        # self.grayscale_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
        self.rgb_transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.Resize([256, 256]),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                                 ])

        self.feature_dims = 6
        self.load_data()
        self.length = len(self.indices)
        self.worker_id = None
        self.counters = [-1 for i in range(self.worker_num)]
        self.counter_ends = [-1 for i in range(self.worker_num)]
        self.subgraph_dones = [False for i in range(self.worker_num)]

        self.inner_indices = np.arange(0, self.length, dtype=np.int32)

        if shuffle:
            np.random.shuffle(self.inner_indices)

        self.is_training = True
        self.test_size = int(0.2 * self.length)
        self.test_inner_indices = self.inner_indices[:self.test_size]

        self.cluster_cache_subdir = os.path.join(self.cache_filedir, "clusters", "%04d_%02d_%02d_%02d_%02d" % (
                self.year, self.begin_month, self.end_month, self.n_history, self.time_interval
            ))
        if not os.path.exists(self.cluster_cache_subdir):
            os.makedirs(self.cluster_cache_subdir)
        self.counter = 0
        self.subgraph_scale = subgraph_scale

    def load_data(self):
        print("Load neighbors")
        self.load_neighbors()
        print("Load graph")
        self.load_graph()
        print("Load buffer")
        self.load_buffers()

        print("load images")
        self.point_images = self.load_imgs(self.point_dir, 1)
        self.poi_images = self.load_imgs(self.poi_dir, 2)
        self.pop_images = self.load_imgs(self.pop_dir, 0)

        if not self.use_cache:
            self.indices.extend([(k, v) for k, v in self.gen_indices().items()])
        else:
            cache_subdir = os.path.join(self.cache_filedir, "indices")
            cache_filepath = os.path.join(cache_subdir, "%04d_%02d_%02d_%s_%02d_%02d.pickle" % (
            self.year, self.begin_month, self.end_month, self.city, self.n_history, self.time_interval
        ))
            if os.path.exists(cache_filepath):
                print("loading indices cache for city %s ..." % self.city)
                with open(cache_filepath, "rb") as f:
                    self.indices.extend([(k, v) for k, v in pickle.load(f).items()])
                print("Indices length is ", len(self.indices))
            else:
                print("cache file for city %s is not detected." % self.city)
                self.indices.extend([(k, v) for k, v in self.gen_indices().items()])

    def load_neighbors(self):
        neighbor_filename = self.neighbor_filepath
        with open(neighbor_filename, "rb") as f:
            self.neighbors = pickle.load(f)[(self.zoom_level, self.pixels)]

    def load_buffers(self):
        node_length = len(self.neighbors)
        self.sorted_node_names = sorted(self.neighbors.keys())

        self.sorted_node_lookups = {}
        self.data_buffer = torch.zeros([node_length, self.n_history, self.feature_dims], dtype=torch.float)
        self.y_buffer = torch.zeros([node_length], dtype=torch.float)
        self.edge_index_buffer = [[], []]
        self.edge_attr_buffer = {}

        for i, name in enumerate(self.sorted_node_names):
            self.sorted_node_lookups[name] = i

        for i, name in enumerate(self.sorted_node_names):
            for neighbor in sorted(self.neighbors[name]):
                if i == neighbor:
                    continue

                self.edge_index_buffer[0].append(i)
                self.edge_index_buffer[1].append(
                    self.sorted_node_lookups[neighbor]
                )
        self.edge_index_buffer = torch.tensor(self.edge_index_buffer, dtype=torch.long)

    def load_graph(self):
        if self.end_month == 12:
            daterange = pd.date_range(start='%d/1/%04d' % (self.begin_month, self.year), end='%d/1/%04d' % (1, self.year+1), freq="5T")[:-1]
        else:
            daterange = pd.date_range(start='%d/1/%04d' % (self.begin_month, self.year), end='%d/1/%04d' % (self.end_month+1, self.year), freq="5T")[:-1]

        daterange_str_dt = {}
        for each in daterange:
            daterange_str_dt[dt.datetime.strftime(each, '%Y-%m-%d %H:%M:%S')] = each
            if self.city == "constance":
                daterange_str_dt[dt.datetime.strftime(each - dt.timedelta(seconds=1), '%Y-%m-%d %H:%M:%S')] = each
            if self.city == "innsbruck":
                daterange_str_dt[dt.datetime.strftime(each - dt.timedelta(minutes=1), '%Y-%m-%d %H:%M:%S')] = each


        graph = pd.read_pickle(self.graph_filepath)
        with open(self.test_node_filepath) as f:
            self.test_nodes = set(json.load(f))

        for station_key, station_value in graph.items():
            self.graph[station_key] = {}
            if self.city in ["rotterdam", "essen"]: # since the lane number in rotterdam is zero, fk
                self.graph[station_key]["data"] = station_value["data"]
                self.graph[station_key]["data"][np.isnan(self.graph[station_key]["data"])] = 0.5
            else:
                self.graph[station_key]["data"] = station_value["data"]

            self.graph[station_key]["timestamp"] = [
                daterange_str_dt[each] for each in station_value["timestamp"]]
            self.graph[station_key]["y"] = station_value["y"]

        for station_key in self.graph.keys():
            self.graph[station_key]["data"] = torch.from_numpy(np.array(self.graph[station_key]["data"]))
            self.graph[station_key]["y"] = torch.from_numpy(np.array(self.graph[station_key]["y"]))

    def load_imgs(self, dir, option):  # option 0, graphscale; option 1, RGB scale
        images = []
        for i, station_key in enumerate(self.sorted_node_names):
            filename = os.path.join(dir, "map%d_%d" % (self.pixels, self.zoom_level), "%s.png" % str(station_key))
            if option == 0:
                images.append(self.grayscale_transform(read_image(filename)))
            elif option == 1:
                images.append(self.rgb_transform(read_image(filename)))
            elif option == 2:
                images.append(self.rgb_transform(read_image(filename)[:-1, :, :]))
            else:
                raise ModuleNotFoundError("option not found, please check again")
        return torch.stack(images, dim=0)

    def gen_indice_cur(self):
        indice_curs = {}
        for i, (dist_key, dist_value) in enumerate(self.graph.items()):
            indice_curs[dist_key] = 0
        return indice_curs

    def gen_indices(self, write_cache=True):
        indices = {}
        if self.end_month == 12:
            daterange = pd.date_range(start='%d/1/%04d' % (self.begin_month, self.year), end='%d/1/%04d' % (1, self.year+1), freq="5T")[:-1]
        else:
            daterange = pd.date_range(start='%d/1/%04d' % (self.begin_month, self.year), end='%d/1/%04d' % (self.end_month+1, self.year), freq="5T")[:-1]
        indices_cur = self.gen_indice_cur()

        for start_date in daterange:
            indices[start_date] = []
            # indices.append([city, start_date, []])
            for i, (dist_key, dist_value) in enumerate(self.graph.items()):
                station_start_index = indices_cur[dist_key]
                station_end_index = indices_cur[dist_key] + self.n_history
                if station_start_index >= len(dist_value["timestamp"]) or start_date < dist_value["timestamp"][station_start_index]:
                    continue


                if start_date == dist_value["timestamp"][station_start_index] \
                    and station_end_index < len(dist_value["timestamp"]) \
                    and dist_value["timestamp"][station_end_index] - dist_value["timestamp"][station_start_index] == dt.timedelta(minutes=self.n_history * 5):

                    indices[start_date].append((dist_key, station_start_index, station_end_index))

                indices_cur[dist_key] += 1

            if len(indices[start_date]) == 0:
                indices.pop(start_date)

        if self.write_cache:
            print("Writing cache for city %s..." % self.city)
            cache_subdir = os.path.join(self.cache_filedir, "indices")
            if not os.path.exists(cache_subdir):
                os.makedirs(cache_subdir)

            with open(os.path.join(cache_subdir, "%04d_%02d_%02d_%s_%02d_%02d.pickle" % (
                self.year, self.begin_month, self.end_month, self.city, self.n_history, self.time_interval
            )), "wb") as f:
                pickle.dump(indices, f)
        return indices

    def __len__(self):
        if self.is_training:
            return self.length * max(len(self.data_buffer) // self.subgraph_scale, 1)
        else:
            return self.test_size * max(len(self.data_buffer) // self.subgraph_scale, 1)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id
        if self.is_training:
            self.counters = [self.length // self.worker_num * i - 1 for i in range(self.worker_num)]
            self.counter_ends = [min(self.length // self.worker_num * (i + 1), self.length - 1) for i in
                                 range(self.worker_num)]
        else:
            self.counters = [self.test_size // self.worker_num * i - 1 for i in range(self.worker_num)]
            self.counter_ends = [min(self.test_size // self.worker_num * (i + 1), self.test_size - 1) for i in
                                 range(self.worker_num)]
        while self.counters[worker_id] < self.counter_ends[worker_id]:
            self.counters[worker_id] += 1
            if self.is_training:
                idx = self.inner_indices[self.counters[worker_id]]
            else:
                idx = self.test_inner_indices[self.counters[worker_id]]

            start_date, raw_data = self.indices[idx]
            data = self.data_buffer.clone()
            y = self.y_buffer.clone()
            subset = []
            subset_test_node_indices = []
            disk_keys = []
            if self.is_training:
                if self.city in ["San_Jose", "San_Diego", "Riverside"]:
                    scale_param = np.random.randint(5, 8) / 10
                else:
                    scale_param = np.random.randint(5, 10) / 10

                raw_data_indices = np.random.choice(np.arange(0, len(raw_data), dtype=np.int32),
                                                    size=int(scale_param * len(raw_data)), replace=False)
                raw_data = [raw_data[i] for i in raw_data_indices]
                for i, (dist_key, station_start_index, station_end_index) in enumerate(raw_data):
                    if dist_key in self.test_nodes:
                        continue
                    else:
                        data_idx = self.sorted_node_lookups[dist_key]
                        disk_keys.append(dist_key)
                        data[data_idx] = self.graph[dist_key]["data"][station_start_index: station_end_index]
                        y[data_idx] = self.graph[dist_key]["y"][station_end_index - 1]
                        subset.append(data_idx)
            else:
                for i, (dist_key, station_start_index, station_end_index) in enumerate(raw_data):
                    if dist_key in self.test_nodes:
                        subset_test_node_indices.append(i)
                        disk_keys.append(dist_key)
                    data_idx = self.sorted_node_lookups[dist_key]
                    data[data_idx] = self.graph[dist_key]["data"][station_start_index: station_end_index]
                    y[data_idx] = self.graph[dist_key]["y"][station_end_index - 1]
                    subset.append(data_idx)


            data = data[subset]
            y = y[subset]
            edge_index, _ = tg.utils.subgraph(subset, self.edge_index_buffer, relabel_nodes=True)

            # device = torch.device("cuda:0")
            device = torch.device("cpu")
            graph_data = Data(
                x=data, edge_index=edge_index, y=y,
                indices=torch.from_numpy(np.arange(0, len(data))).long()
            ).to(device)

            pop_images = self.pop_images[subset].to(device)
            poi_images = self.poi_images[subset].to(device)
            point_images = self.point_images[subset].to(device)
            if len(data) < 3:
                continue

            else:
                yield int(round(start_date.timestamp())),\
                    graph_data.to(device),\
                    pop_images,\
                    poi_images,\
                    point_images, \
                    disk_keys, \
                    subset_test_node_indices

if __name__ == "__main__":
    from datetime import datetime
    def test_inner(dataset, is_training):
        dataset.is_training = is_training
        print(len(dataset))
        dataloader = DataLoader(dataset, batch_size=None, num_workers=1)
        count = 0
        # for j, batch_data in enumerate(dataloader):
        for j, batch_data in enumerate(tqdm(dataloader)):
            start_date, graph_data, pop_images, poi_images, point_images, test_node_indices, subset_test_node_indices = batch_data
            print(graph_data)
            print(len(test_node_indices))
            count += 1
            # if count % 100 == 0:
            #     print(count)
            # break
        return count

    data_filedir = "../processed_data/urban_data/"
    cache_filedir = "../processed_data/urban_data/cache"
    if not os.path.exists(cache_filedir):
        os.makedirs(cache_filedir)

    dataset = MainDataset(data_filedir, "Stockton", cache_filedir,
                          year=2021, begin_month=7, end_month=8,
                          n_history=12, time_interval=5,
                          zoom_level=15, pixels=1024,
                          write_cache=True, use_cache=True, num_workers=4, subgraph_scale=200, shuffle=False)

    # test_inner(dataset, True)
    test_inner(dataset, False)

