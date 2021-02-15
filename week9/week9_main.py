from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import torch

from models import CRNN
from utils import CRNN_dataset
from tqdm import tqdm
import argparse
import os


def hyperparameters() :
    """
    argparse는 하이퍼파라미터 설정, 모델 배포 등을 위해 매우 편리한 기능을 제공합니다.
    파이썬 파일을 실행함과 동시에 사용자의 입력에 따라 변수값을 설정할 수 있게 도와줍니다.

    argparse를 공부하여, 아래에 나온 argument를 받을 수 있게 채워주세요.
    해당 변수들은 모델 구현에 사용됩니다.

    ---변수명---
    변수명에 맞춰 type, help, default value 등을 커스텀해주세요 :)
    
    또한, argparse는 숨겨진 기능이 지이이이인짜 많은데, 다양하게 사용해주시면 우수과제로 가게 됩니다 ㅎㅎ
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type = str, default = "/content/drive/My Drive/week9/assignment1/dataset", help = 'path to dataset')
    parser.add_argument('--savepath', type = str, default = 'best_model', help = 'file name for saving  best model')
    parser.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    parser.add_argument('--epochs', type = int, default = 30, help = 'number of epochs')
    parser.add_argument('--optim', type = str, default = 'adam', choices = ["adam", "rmsprop"], help = 'select optimizer')
    parser.add_argument('--lr', type = float, default = 0.01, help = 'learning rate')
    parser.add_argument('--device', type = int, default = 1, help = 'gpu number')
    parser.add_argument('--img_width', type = int, default = 100, help = 'width of input image')
    parser.add_argument('--img_height', type = int, default = 32, help = 'height of input image')
    
    return parser.parse_args()


def main():
    args = hyperparameters()


    train_path = os.path.join(args.path, 'train')
    test_path = os.path.join(args.path, 'test')

    # gpu or cpu 설정
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu') 

    # train dataset load
    train_dataset = CRNN_dataset(path=train_path, w=args.img_width, h=args.img_height)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # test dataset load
    test_dataset = CRNN_dataset(path=test_path, w=args.img_width, h=args.img_height)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    

    # model 정의
    model = CRNN(args.img_height, nc = 1,nh = 256, nclass = 37, n_rnn = 2)
    print(model)
 
    # loss 정의
    criterion = nn.CTCLoss()
    
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                            betas=(0.5, 0.999))
    elif args.optim == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    else:
        assert False, "옵티마이저를 다시 입력해주세요. :("

    model = model.to(device)
    best_test_loss = 100000000
    for i in range(args.epochs):
        
        print('epochs: ', i)

        print("<----training---->")
        model.train()
        for inputs, targets in tqdm(train_dataloader):
            inputs = inputs.permute(0, 1, 3, 2) # inputs의 dimension을 (batch, channel, h, w)로 바꿔주세요. hint: pytorch tensor에 제공되는 함수 사용
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets 
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = nn.functional.log_softmax(model(inputs), dim = -1)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            """
            CTCLoss의 설명과 해당 로스의 input에 대해 설명해주세요.

            음성인싱이나 필기 인식처럼 입력에 레이블 할당 위치를 정하기 어려운 연속적인 시퀀스를 다루는 문제에 사용하는 손실함수이다. CTCLoss 함수의 input은 output의 log probability이다.

            """

            loss = criterion(preds, target_text, preds_length, target_length) / batch_size 
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

        print("<----evaluation---->")

        """
        model.train(), model.eval()의 차이에 대해 설명해주세요.
        .eval()을 하는 이유가 무엇일까요?
        
        model.train()의 경우, model을 학습시킬 때는 사용하고, model.eval()은 model을 검증할 때 사용한다. 굳이 이렇게 나눠서 .eval()을 하는 이유는 dropout이나 batchnorm이 있는 모델의 경우 학습할 때와 테스트할 때 모델이 달라지기 때문이다. 학습 시에는 dropout이나 batchnorm을 사용하는 반면, 테스트 시에는 dropout이나 batchnorm을 사용하지 않는다. 뿐만 아니라 .eval()은 학습 시 발생하는 역전파 등을 진행하지 않는다.
        
        """

        model.eval() 
        loss = 0.0

        for inputs, targets in tqdm(test_dataloader):
            inputs = inputs.permute(0, 1, 3, 2)
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            target_text, target_length = targets
            target_text, target_length = target_text.to(device), target_length.to(device)
            preds = nn.functional.log_softmax(model(inputs), dim = -1)
            preds_length = Variable(torch.IntTensor([preds.size(0)] * batch_size))

            loss += criterion(preds, target_text, preds_length, target_length) / batch_size
        
        print("test loss: ", loss)
        if loss < best_test_loss:
            # loss가 bset_test_loss보다 작다면 지금의 loss가 best loss가 되겠죠?
            best_test_loss = loss
            # args.savepath을 이용하여 best model 저장하기
            torch.save({'loss': best_test_loss}, args.savepath)
            print("best model 저장 성공")



if __name__=="__main__":
    main()
