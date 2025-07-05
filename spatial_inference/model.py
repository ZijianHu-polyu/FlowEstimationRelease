import torch
import numpy as np
import torch_geometric as tg
import torch.functional as F
from spatial_inference.base_model import GATv2Conv, ImageEncoderNet, ImageQNet, ImageKNet, ImageVNet, ImageDenseNet, ImageConnectorAtt2Dense, ImageCrossAttentionNet, ImageDecoderNet


class MainModel(torch.nn.Module):
    def __init__(self, config):
        super(MainModel, self).__init__()
        self.embedding_dim = config["model"]["embedding_dim"]
        self.n_history = config["n_history"]
        self.graph_multi_head = config["model"]["graph"]["graph_multi_head"]
        self.graph_dropout = config["model"]["graph"]["graph_dropout"]
        self.graph_alpha = config["model"]["graph"]["graph_alpha"]
        self.image_encoder_hidden_dim = config["model"]["image"]["encoder_hidden_dim"]
        self.image_encoder_output_dim = config["model"]["image"]["encoder_output_dim"]
        self.image_dense_input_dim = config["model"]["image"]["dense_input_dim"]
        self.image_dense_hidden_dim = config["model"]["image"]["dense_hidden_dim"]
        self.image_dense_output_dim = config["model"]["image"]["dense_output_dim"]
        self.image_decoder_hidden_dim = config["model"]["image"]["decoder_hidden_dim"]

        self.graph_embedding = torch.nn.Linear(6, self.embedding_dim)
        self.gat_layer_s = GATv2Conv(self.embedding_dim, self.embedding_dim, self.n_history,
                               dropout=self.graph_dropout, negative_slope=self.graph_alpha, heads=self.graph_multi_head)
        self.gat_layer_t = tg.nn.GATv2Conv(self.embedding_dim, self.embedding_dim,
                               dropout=self.graph_dropout, negative_slope=self.graph_alpha, heads=self.graph_multi_head)
        self.gru_layer_s = torch.nn.GRU(self.embedding_dim * self.graph_multi_head,
                                        self.embedding_dim * self.graph_multi_head, batch_first=True)
        self.gru_layer_t = torch.nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)

        self.fc_graph = torch.nn.Sequential(
            torch.nn.Linear(2 * self.embedding_dim * self.graph_multi_head, 2 * self.embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            torch.nn.ReLU(),
        )

        self.image_encoder_poi = ImageEncoderNet(3, self.image_encoder_hidden_dim, self.image_encoder_output_dim)
        self.image_encoder_point = ImageEncoderNet(3, self.image_encoder_hidden_dim, self.image_encoder_output_dim)
        self.image_encoder_pop = ImageEncoderNet(1, self.image_encoder_hidden_dim, self.image_encoder_output_dim)

        self.image_q_poi = ImageQNet(self.image_encoder_output_dim)
        self.image_k_poi = ImageKNet(self.image_encoder_output_dim)
        self.image_q_point = ImageQNet(self.image_encoder_output_dim)
        self.image_k_point = ImageKNet(self.image_encoder_output_dim)
        self.image_q_pop = ImageQNet(self.image_encoder_output_dim)
        self.image_k_pop = ImageKNet(self.image_encoder_output_dim)

        self.image_v_poi = ImageVNet(self.image_encoder_output_dim)
        self.image_v_point = ImageVNet(self.image_encoder_output_dim)
        self.image_v_pop = ImageVNet(self.image_encoder_output_dim)

        self.image_crossattention = ImageCrossAttentionNet(self.image_encoder_output_dim)
        self.image_connector = ImageConnectorAtt2Dense(self.image_encoder_output_dim * 3, self.image_dense_input_dim)
        self.image_dense = ImageDenseNet(self.image_dense_input_dim, self.image_dense_hidden_dim, self.image_dense_output_dim)
        self.image_decoder = ImageDecoderNet(self.image_dense_output_dim * 3, self.image_decoder_hidden_dim, self.embedding_dim)

        # self.image_backbone = ImageBackboneNet(embedding_dim)

        # Output
        self.fc1 = torch.nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.fc2 = torch.nn.Linear(self.embedding_dim, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x_graph, x_poi, x_point, x_pop):
        # Encode graph
        # graph size: [node_num, node_feat, history]
        # x_graph = torch.squeeze(x_graph, 0)
        # x_graph = torch.permute(x_graph, [2, 0, 1]) # transpose for GAT, size: [history, node_num, node_feat]
        x_graph.x = self.graph_embedding(x_graph.x) # size: [history, node_num, node_emb]
        x_graph_s = self.gat_layer_s(x_graph.x, x_graph.edge_index) # size: [history, node_num, node_emb]
        # x_graph_s = torch.permute(x_graph_s, [1, 0, 2]) # transpose for GRU: [node_num, history, node_emb]
        x_graph_s, h_s  = self.gru_layer_s(x_graph_s) # size: [node_num, history, node_emb]
        x_graph_s = x_graph_s[:, -1, :] # , and pick the last one ->[node_num, node_emb]
        # x_graph_t = torch.permute(x_graph, [1, 0, 2])  # transpose for GRU: [node_num, history, node_emb]
        x_graph_t, h_t = self.gru_layer_t(x_graph.x) # output size: [node_num, history, node_emb]
        x_graph_t = x_graph_t[:, -1, :] # , and pick the last one: [node_num, emb]
        x_graph_t = self.gat_layer_t(x_graph_t, x_graph.edge_index) # out size: [1, node_num, emb]
        # x_graph_t = torch.squeeze(x_graph_t)# out size: [ node_num, emb]
        x_graph = self.fc_graph(torch.concat([x_graph_s, x_graph_t], dim=1)) # decoder, size: [node_num, node_emb * 2] -> [node_num, node_emb]

        # Encode image
        x_poi = self.image_encoder_poi(x_poi)
        x_point = self.image_encoder_point(x_point)
        x_pop = self.image_encoder_pop(x_pop)

        x_poi_q = self.image_q_poi(x_poi)
        x_point_q = self.image_q_point(x_point)
        x_pop_q = self.image_q_pop(x_pop)

        x_poi_k = self.image_k_poi(x_poi)
        x_point_k = self.image_k_point(x_point)
        x_pop_k = self.image_k_pop(x_pop)

        x_poi_v = self.image_v_poi(x_poi)
        x_point_v = self.image_v_point(x_point)
        x_pop_v = self.image_v_pop(x_pop)


        x_poi, x_point, x_pop = self.image_crossattention(
            x_poi_q, x_poi_k, x_poi_v,
            x_point_q, x_point_k, x_point_v,
            x_pop_q, x_pop_k, x_pop_v,
            x_poi, x_point, x_pop
        )
        # print(x_graph.shape)
        #
        print(np.save("x_poi.npy", x_poi.detach().cpu().numpy()))
        print(np.save("x_point.npy", x_point.detach().cpu().numpy()))
        print(np.save("x_pop.npy", x_pop.detach().cpu().numpy()))

        x_poi, x_point, x_pop = self.image_connector(x_poi, x_point, x_pop)



        x_p = self.image_dense(x_poi, x_point, x_pop)

        # print(np.save("x_p.npy", x_p.detach().cpu().numpy()))
        x_p = self.image_decoder(x_p)

        out = torch.concat([x_graph, x_p], dim=1) # size: [node_num, node_emb * 2]
        out = self.fc1(out) # size: [node_num, node_emb]
        out = self.relu(out) # size: [node_num, node_emb]
        out = self.fc2(out).T # size: [node_num, 1]->[1, node_num]
        return out