import torch.nn as nn 
import torch
import torchvision.models as models

class Net_v1(nn.Module):
    def __init__(self):
        super(Net_v1, self).__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, video):
        video_features = self.resnet(video)
        video_features = video_features.view(video_features.shape[0], -1)
        video_features = video_features.unsqueeze(0)

        video_rnn, _ = self.rnn(video_features)
        video_rnn = video_rnn.squeeze(0)
        output = self.fc(video_rnn)
        output = self.sigmoid(output)
        return output

class Net_v2(nn.Module):
    def __init__(self):
        super(Net_v2, self).__init__()

        vgg = models.vgg11(pretrained=True)
        vgg.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.video_cnn = nn.Sequential(*list(vgg.children())[:-1])

        self.video_rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.AdaptiveAvgPool1d(output_size=1)
        )

        self.audio_rnn = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(512 + 256, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, video, audio):

        features = self.video_cnn(video)
        features = features.view(features.shape[0], -1).unsqueeze(0)
        video_output, _ = self.video_rnn(features)
        video_output = video_output.squeeze(0)

        audio = audio.view(1, -1).unsqueeze(0)
        audio_features = self.audio_cnn(audio)
        audio_features = audio_features.view(audio_features.shape[0], -1).unsqueeze(1)
        audio_output, _ = self.audio_rnn(audio_features)

        audio_output = audio_output.squeeze(0).repeat(video_output.shape[0], 1)

        features = torch.cat((video_output, audio_output), dim=1)

        output = self.fc(features)
        output = self.sigmoid(output)
        return output.view(-1, 1)
    
class Net_v2_Improved(nn.Module):

    """
    A neural network model that combines video and audio features for classification tasks.
    avg time for video 2 sec
    
    The model uses:
    - MobileNetV3_Small as the feature extractor for video data.
    - GRU layers to process video and audio features over time.
    - CNN layers to process audio features.

    Input:
    - video (Tensor): shape (frames, 3, height, width), where 'frames' dimension is number of frames after video preprocessing
    `height` and `width` are the dimensions of the video frames (96x96 expected for this model), 
    and 3 represents the color channels (RGB).
    - audio (Tensor): A tensor of shape (audio_length, audio_level), where `audio_length` 
    is the length of the audio signal (audio_lenght == frames). Expected tensor shape is (audio_level, 1)

    Output:
    - output (Tensor): A tensor of shape (frames, 1), representing the binary classification score (0 or 1).
    """

    def __init__(self):
        super(Net_v2_Improved, self).__init__()

        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        mobilenet.classifier = nn.Identity()
        self.video_cnn = mobilenet

        self.video_rnn = nn.GRU(input_size=576, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)

        self.audio_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )

        self.audio_rnn = nn.GRU(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(512 + 256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):
        video_features = self.video_cnn(video)
        video_features, _ = self.video_rnn(video_features.unsqueeze(0))
        video_features = video_features.squeeze(0)

        audio_features = self.audio_cnn(audio.view(1, 1, -1))
        audio_features, _ = self.audio_rnn(audio_features.view(1, 1, -1))
        audio_features = audio_features.squeeze(0).repeat(video_features.shape[0], 1)
        
        combined_features = torch.cat((video_features, audio_features), dim=1)
        output = self.fc(combined_features)
        return output

class Net_v3(nn.Module):
    def __init__(self):
        super(Net_v3, self).__init__()

        self.video_audio_cnn = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )

        self.rnn = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, video, audio):

        audio = audio.unsqueeze(1).unsqueeze(2)
        audio = audio.repeat(1, 1, video.shape[-2], video.shape[-1])

        video_audio = torch.cat((video, audio), dim=1)
        
        cnn_output = self.video_audio_cnn(video_audio)
        cnn_output = cnn_output.reshape(cnn_output.shape[0], -1).unsqueeze(0)

        rnn_output, _ = self.rnn(cnn_output)
        rnn_output = rnn_output.squeeze(0)

        output = self.fc(rnn_output)
        return output