import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBiGRU(nn.Module):
    def __init__(self, num_classes=27, hidden_size=256, num_layers=2, dropout=0.5):
        
        super(ResNetBiGRU, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        frozen_modules = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1, resnet.layer2
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        
        batch_size, time_steps, c, h, w = x.size()
        
        x = x.view(batch_size * time_steps, c, h, w)
        
        x = self.backbone(x)
        x = self.spatial_pool(x)
        x = x.view(batch_size * time_steps, -1)
        
        x = self.projection(x)
        
        x = x.view(batch_size, time_steps, -1)
        
        rnn_out, _ = self.rnn(x)
        
        out = self.classifier(rnn_out)
        
        return out

class ResNetBiLSTM(nn.Module):
    def __init__(self, num_classes=27, hidden_size=256, num_layers=2, dropout=0.5):
        super(ResNetBiLSTM, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        frozen_modules = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1, resnet.layer2
        ]
        for module in frozen_modules:
            for param in module.parameters():
                param.requires_grad = False

        self.projection = nn.Sequential(
            nn.Linear(512, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) 
        )
        
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, time_steps, c, h, w = x.size()
        
        x = x.view(batch_size * time_steps, c, h, w)
        x = self.backbone(x)           
        x = self.spatial_pool(x)       
        x = x.view(batch_size * time_steps, -1) 
        x = self.projection(x)         
        
        x = x.view(batch_size, time_steps, -1)
        
        rnn_out, _ = self.rnn(x)
        
        out = self.classifier(rnn_out)
        
        return out

if __name__ == "__main__":
    dummy_batch = torch.randn(4, 43, 3, 224, 224)
    
    gru_model = ResNetBiGRU(num_classes=27, hidden_size=256, num_layers=2, dropout=0.5)
    lstm_model = ResNetBiLSTM(num_classes=27, hidden_size=256, num_layers=2, dropout=0.5)
    
    gru_predictions = gru_model(dummy_batch)
    lstm_predictions = lstm_model(dummy_batch)
    
    print(f"Input shape:  {dummy_batch.shape}")
    print(f"Output shape: {gru_predictions.shape}")
    print(f"BiLSTM Output shape: {lstm_predictions.shape}")
    
    gru_ctc_predictions = gru_predictions.permute(1, 0, 2)
    lstm_ctc_predictions = lstm_predictions.permute(1, 0, 2)
    print(f"Ready for CTCLoss shape: {gru_ctc_predictions.shape}")
    print(f"Ready for CTCLoss shape: {lstm_ctc_predictions.shape}")

    
    
    