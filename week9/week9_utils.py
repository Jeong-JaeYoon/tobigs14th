from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import collections
from glob import glob
import os
from PIL import Image

"""
main.py 함수를 참고하여 다음을 생각해봅시다.

1. CRNN_dataset은 어떤 모듈을 상속받아야 할까요?
2. CRNN_dataset의 역할은 무엇일까요? 왜 필요할까요?
3. 1.의 모듈을 상속받는다면 __init__, __len__, __getitem__을 필수로 구현해야 합니다. 각 함수의 역할을 설명해주세요.
    1) torch.utils.data에 있는 Dataset을 받아야 합니다.
    2) CRNN_dataset은 데이터의 전처리와 gradient를 구하는데 용이하게 해준다.
    3) __init__ : 데이터셋의 전처리를 해준다.
       __len__ : 데이터셋의 길이를 계산한다.
       __getitem__ : 데이터셋에서 특정 1개의 샘플을 가져온다.
"""


class CRNN_dataset(Dataset):
    def __init__(self, path, w=100, h=32, alphabet='0123456789abcdefghijklmnopqrstuvwxyz', max_len=36):
        self.max_len=max_len
        self.path = path
        self.files = glob(path+'/*.jpg') 
        self.n_image = len(self.files)
        assert (self.n_image > 0), "해당 경로에 파일이 없습니다. :)"

        self.transform = transforms.Compose([
            transforms.Resize((w, h)), # image 사이즈를 w, h를 활용하여 바꿔주세요.
            transforms.ToTensor() # tensor로 변환해주세요.
        ])
        """
        strLabelConverter의 역할을 설명해주세요.
        1. text 문제를 풀기 위해 해당 함수는 어떤 역할을 하고 있을까요?

        들어온 프레임 별 예측을 label sequence로 변환하는데 사용합니다.

        2. encode, decode의 역할 설명

        encode : 들어온 프레임 별 예측(str 형식)을 텍스트로 인코딩한다.
        
        decode : 인코딩된 텍스트를 다시 str 형식으로 디코딩한다.

        """
        self.converter = strLabelConverter(alphabet) 
        
    def __len__(self):
        return self.n_image # hint: __init__에 정의한 변수 중 하나

    def __getitem__(self,idx):
        label = self.files[idx].split('_')[1]
        img = Image.open(self.files[idx]).convert('L')
        img = self.transform(img)
        """
        max_len이 왜 필요할까요? # hint: text data라는 점

        max_len으로 label에 제한을 둠으로써, label이 무한정 커지는 것을 방지한다. 또한 text data에 적합한 RNN계열의 모델은 vanishing gradient 문제가 발생할 수 있으므로 max_len으로 그 범위를 제한하는 것이다.

        """

        if len(label) > self.max_len:
            label = label[:self.mfax_len]
        label_text, label_length = self.converter.encode(label)

        if len(label_text) < self.max_len:
            temp = torch.ones(self.max_len-len(label), dtype=torch.int)
            label_text = torch.cat([label_text, temp])

        return img, (label_text, label_length) # hint: main.py를 보면 알 수 있어요 :)



# 아래 함수는 건드리지 마시고, 그냥 쓰세요 :)
class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-' 

        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts
