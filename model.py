import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.unit0_conv = nn.Conv3d(14,32,
                               kernel_size=3,stride=1, bias=True)
        self.unit0_func = nn.ReLU()

        self.unit1_pool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.unit1_conv = nn.Conv3d(32, 64,
                                    kernel_size=3,padding=1, stride=1, bias=True)
        self.unit1_func = nn.ReLU()

        self.unit2_conv = nn.Conv3d(64, 64,
                                    kernel_size=1, stride=1, bias=True)
        self.unit2_func = nn.ReLU()

        self.unit3_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unit3_conv = nn.Conv3d(64, 64,
                                    kernel_size=3,padding=1, stride=1, bias=True)
        self.unit3_func = nn.ReLU()

        self.unit4_conv = nn.Conv3d(64, 128,
                                    kernel_size=1, stride=1, bias=True)
        self.unit4_func = nn.ReLU()

        self.unit5_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.unit5_conv = nn.Conv3d(128, 128,
                                    kernel_size=3,padding=1 ,stride=1, bias=True)
        self.unit5_func = nn.ReLU()

        # fully connected layer in original classification
        #self.final_fc = nn.Linear(16000, 2)

        # extra 2-layer MLP
        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16000, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),

            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0)
        )

        # extra classifier layer
        self.classifier = nn.Linear(128, 2)
        
        self.freeze_all_except_unitx()

    def forward(self, x):
        out = self.unit0_conv(x)
        out = self.unit0_func(out)

        out = self.unit1_pool(out)
        out = self.unit1_conv(out)
        out = self.unit1_func(out)

        out = self.unit2_conv(out)
        out = self.unit2_func(out)

        out = self.unit3_pool(out)
        out = self.unit3_conv(out)
        out = self.unit3_func(out)

        out = self.unit4_conv(out)
        out = self.unit4_func(out)

        out = self.unit5_pool(out)
        out = self.unit5_conv(out)
        out = self.unit5_func(out)

        #out = out.reshape(out.size(0), -1)
        #out = self.final_fc(out)
        
        out = self.mlp_head(out) 
        out = self.classifier(out) 

        return out


    def freeze_all_except_unitx(self):
        """
        Freezes all layers except mlp_head, classifier
        """
        for name, param in self.named_parameters():
            if any(x in name for x in [ 'mlp_head', 'classifier']):
                param.requires_grad = True
            else:
                param.requires_grad = False



    def load_pretrained_weights(self, path):
        """
        Loads weights from a pre-trained model
        """
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict, strict=False)