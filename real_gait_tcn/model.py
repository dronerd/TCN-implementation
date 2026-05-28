import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation
        )
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.relu2,
            self.dropout2
        )

        # Residual connection
        self.downsample = nn.Conv1d(
            in_channels,
            out_channels,
            1
        ) if in_channels != out_channels else None

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNWithoutAttention(nn.Module):
    def __init__(
        self,
        input_features,
        tcn_channels=[32, 64, 64],
        kernel_size=3,
        dropout=0.2,
        output_dim=1
    ):
        super().__init__()

        layers = []
        in_channels = input_features

        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        self.regressor = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.transpose(1, 2)

        final_output = tcn_out[:, -1, :]
        prediction = self.regressor(final_output)

        return prediction


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = d_model // num_heads

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, values, keys, queries):
        N = queries.shape[0]

        Q = self.query(queries)
        K = self.key(keys)
        V = self.value(values)

        Q = Q.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).reshape(N, -1, self.d_model)

        out = self.fc_out(out)
        return out


class TCNWithAttention(nn.Module):
    def __init__(
        self,
        input_features,
        tcn_channels=[32, 64, 64],
        kernel_size=3,
        dropout=0.2,
        output_dim=1,
        num_heads=4
    ):
        super().__init__()

        layers = []
        in_channels = input_features

        for i, out_channels in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation,
                    dropout
                )
            )
            in_channels = out_channels

        self.tcn = nn.Sequential(*layers)

        # Attention mechanism
        self.attention = MultiHeadAttention(tcn_channels[-1], num_heads)

        self.regressor = nn.Sequential(
            nn.Linear(tcn_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        tcn_out = self.tcn(x)
        tcn_out = tcn_out.transpose(1, 2)

        # Apply attention
        attention_out = self.attention(tcn_out, tcn_out, tcn_out)

        final_output = attention_out[:, -1, :]
        prediction = self.regressor(final_output)

        return prediction