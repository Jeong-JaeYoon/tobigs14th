# models.py 파일과 논문을 바탕으로 빈칸을 채워주세요.
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.cnn_module = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1),
            nn.Conv2d(64, 128, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1),
            nn.Conv2d(128, 256, kernel_size = (3,3), stride=(1,1), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 1), padding = (0, 1)),
            nn.Conv2d(256, 512, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            nn.Conv2d(512, 512, kernel_size = (3,3), stride = (1,1), padding = (1,1)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 1), padding = (0, 1)),
            nn.Conv2d(512, 512, kernel_size = (2,2), stride = (1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
            )
            
        self.rnn_model = nn.Sequential(
            nn.LSTM(512,256,bidirectional = True),
            nn.Linear(512, 256, bias = True)
            nn.LSTM(256,256,bidirectional = True),
            nn.Linear(512, 37, bias = True)
            )


        self.embedding = nn.Linear(512, 37)

    def forward(self, input):
        conv = self.cnn_module(input)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output, _ = self.rnn_model(conv)
        seq_len, batch, h_2 =  output.size()
        output = output.view(seq_len * batch, h_2)
        output = self.embedding(output)
        output = output.view(seq_len, batch, -1)
        return output

