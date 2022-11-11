# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
# 경험 리플레이를 실행할 때 다양한 배치로부터 샘플을 무작위로 가져와야 된다
# So, random library가 필요하다
import os
# 모델을 loading할 때 유용한 라이브러리이다
# 모델이 준비되면 그 모델을 저장할 수 있는 코드와 로딩할 수 있는 코드를 실행해야 한다
# 컴퓨터를 종료하고 나서 이전에 탐색 훈련을 하던 뇌를 재사용하고 싶을 경우 
# 차의 뇌를 저장하고 로딩할 수 있게 된다
import torch
# 신경망을 실행할 때 우리는 파이토치를 사용하기 때문에 torch 라이브러리가 필요하다
# 동적 그래픽을 다룰 수 있기 때문에 다른 인공지능 패키지 보다 파이토치를 사용하는걸 추천
import torch.nn as nn
# torch.nn 라이브러리를 torch에서 불러온다
# 신경망을 실행할 수 있는 모든 툴이 포함된 모듈이다
# 신경망이 센서에서 얻은 3개의 신호와 방향, -방향을 입력정보로 받고
# 최적의 행동을 출력 결과로 반환한다
# 다양한 행동의 Q값을 반환한다 
# > Softmax 함수를 사용해서 차의 목표를 달성할 수 있는 가장 최적의 행동을 반환
import torch.nn.functional as F
# functional package는 신경망을 실행할 때 사용하는 다양한 함수를 포함한다
# 일반적으로 마지막 함수로 huber loss를 사용하는데 이 함수는 수렴을 도와주고 
# nn module의 sub module인 functional에 포함되어 있다
import torch.optim as optim 
# optim : optimizer / 확률적 경사 하강법을 수행할 때 필요한 최적화기를 불러오는 역할
import torch.autograd as autograd
# 변수 클래스를 가져오기 위한 라이브러리 : autogard
# 고급 배열인 tensor를 변환해서 경사가 포함된 변수로 바꾸기 위해서 변수 클래스를 불러온다
# tensor 자체가 필요하다기 보다는 tensor를 경사가 포함된 변수로 표현하는 과정이 필요하다
# > 변수 클래스를 사용해서 텐서와 경사정보가 포함된 변수로 텐서를 변환해야 된다 
from torch.autograd import Variable

# Creating the architecture of the Neural Network
# 신경망을 객체로 만들어야 되니까 클래스를 만든다
# > 훨씬 편리 -> Why? : 구성하려는 대상의 모델이 클래스이기 때문에

class Network(nn.Module):
# Network class에 객체 프로그래밍 기술인 상속을 사용     
# 부모 class로부터 모든 tool을 물려받는 기술 
# So, 우리가 만들 네트워크 클래스가 nn.Module이라는 상위 클래스의 자식 클래스가 된다
# 신경망 실행에 필요한 모든 tool을 Module Class로 부터 받을 수 있다 => 상속
    def __init__(self, input_size, nb_action):
    # init 함수 : class를 만들 때 항상 나오는 함수
    # 기본적으로 객체의 변수를 정의해주는 역할 
    # 객체 변수는 여기서 신경망을 의미 / 전역 변수와 반대로 객체에 붙어있는 변수
    # init 함수에서 신경망 구조를 정의, 입력층에서 이 작업이 수행된다
    # 입력층에는 5개의 뉴런이 입력된다 / 입력 상태의 인코딩 벡터가 5차원으로 구성되어 있으니까
    # 은닉층 정의 / 먼저 1개의 은닉층으로 시작, 나중에 여러 신경망 구조를 시도할 수 있다
    # 출력층 정의 / 출력층에는 매시간 수행가능한 행동 정보가 들어있다
    # init 함수에서 이러한 작업을 진행한다
    # def __init__(self, ... 에서 self의 역할 : 우리가 만들 클래스에서 만들어질 객체를 나타내는 인수
    # 이 클래스를 만들면서 지시문이나 신경 모델을 만들려고 하는데 클래스가 준비 되면 신경망을 
    # 우리가 원하는만큼 많이 만들 수 있다
    # 각각의 신경망이 클래스의 객체가 된다 
    # 객체를 다양한 목적으로 사용할 거라서 객체의 변수가 어떤 것인지 찾아내야 된다 
    # So, 여기에 self를 입력해서 객체를 구체적으로 명시한다
    # 입력 뉴런 숫자 = input_size, 5가 된다, 입력 벡터가 5차원이니까 
    # 방향, -방향은 목표 방향으로 계속 달리게 하는 요소
    # 행동을 알려주는 신경망의 출력 뉴런 = nb_action
    # 왼쪽, 앞쪽, 오른쪽으로 가는 3개의 행동이 가능 > nb_action에 3개의 행동이 들어온다 
        super(Network, self).__init__()
        # super 함수 : nn.Module을 상속하는 함수
        # Module tool에서 상속을 받으려면 상속 기능을 사용해야 된다
        # super 함수 안에 먼저 Network를 입력해서 네트워크 클래스를 명시한다 > 부모 클래스 모듈에서 상속받아야하니까
        self.input_size = input_size
        # 입력층을 명시 / 객체에 붙을 변수를 소개
        # 네트워크 클래스에서 객체를 만들 때마다 입력 크기가 명시된다 
        # Ex. 5를 입력 > self.input_size = 5가 입력, 입력 크기 객체 변수(=input_size)의 가치도 5가 된다 
        # 신경망 입력층에 5개의 입력 뉴런이 생기게된다
        self.nb_action = nb_action
        # self.nb_action = 3, 네트워크 객체에 붙은 행동 변수도 3이 된다 
        self.fc1 = nn.Linear(input_size, 30)
        # 완전 연결 / 신경망의 여러 층 사이의 완전 연결을 의미
        # 지금은 하나의 은닉층으로만 된 구성된 신경망을 만드므로 완전 연결이 2개가 된다 
        # 첫번째 완전 연결 : 입력층과 은닉층 사이 / 두번째 완전 연결 : 은닉층과 출력층 사이에 위치
        # 첫번째 완전 연결 : fc1 / self를 사용해서 fc1이 내 객체 변수라는 걸 명시함, self.fc1
        # 완전연결? : 입력층의 모든 뉴런과 출력층의 모든 뉴런이 서로 완전히 연결되는 것을 의미
        # 완전 연결을 사용하려면 Linear 함수를 사용하고 인수를 입력 / 인수 : in_features : 연결하고 싶은 첫 번째 층의 뉴런 수를 나타냄 
        # 두번째 인수는 out_features : 연결하고 싶은 두 번째 층의 뉴런 수를 나타냄
        # 두 번째 층 : 오른쪽에 있는 은닉층 / 마지막 인수 : bias = True, 편향을 가지기 위해서 기본 가치를 설정 
        # 뉴런에 가중치만 붙는 것이 아니라 각 layer에 가중치와 편향도 붙는다 
        # 30? : 은닉층에 30개의 뉴런을 선택했다는 의미 / 30개의 은닉 뉴런을 만들겠다는 의미이다 
        self.fc2 = nn.Linear(30, nb_action)
        # 두번째 완전 연결 : 은닉층과 출력층 사이
        # So, 네트워크 클래스에서 객체를 만들 때 마다 객체가 초기화 된다
        # 객체가 만들어지자마자 4개의 변수들(input_size, nb_action, fc1, fc2)가 정의 된다
        # 우리가 만드는 각각의 객체가 신경망과 연동되어서 입력 뉴런 5개, 은닉 뉴런 30개, 출력 뉴런 3개와 연결된다 
        # 더 자세한 내용은 33.자율주행차 - 4단계에서 확인 
    def forward(self, state):
    # 신경망의 뉴런과 신호를 활성화하는 함수
    # rectifier 활성화 함수도 사용 / 우리는 순수하게 비선형 문제만 다룰건데 
    # rectifier 함수가 선형성을 부수기 때문에 이 함수를 사용한다
    # But, forward함수를 사용해서 신경망 출력 결과인 Q값을 반환할건데
    # 각 행동에 1개의 Q값이 들어있고 나중에 Q값의 최댓값을 구하거나
    # 소프트맥스 방식을 사용하는 두 방법의 하나를 통해서 최종 행동을 반환
    # Q값 : 왼쪽, 앞쪽, 오른쪽으로 가는 3개의 행동을 일컫는다  
    # 첫 번째 인수 : self, 두 번째 인수 : 입력 정보 state, 상태가 곧 신경망 입력 정보나 다름 없으므로  
        x = F.relu(self.fc1(state))
        # 은닉 뉴런을 활성화한다 / 은닉 뉴런을 x변수로 부르자 / x가 은닉 뉴런을 의미한다
        # 어떻게 활성화할까? : 입력 뉴런을 가져온다.
        # 첫 번째 완전 연결인 fc1을 사용해서 은닉 뉴런을 가져온다. 그리고 활성화 함수인 rectifier 함수를 은닉 뉴런에 적용
        # relu 함수를 불러온다
        # 첫번째 완전 연결이 객체 변수 > self를 먼저 입력해서 self.fc1을 입력한다 / 신경망의 첫 번째 완전 연결
        # 은닉뉴런을 위에서 활성화 했으므로 출력 뉴런을 반환해야 된다.
        q_values = self.fc2(x)
        # 출력 뉴런 : 행동을 의미 / 직접적인 행동을 의미하는게 아니라 Q값을 의미한다
        # 큐러닝을 사용해서 각 행동의 Q값을 구한다
        # q_values인 이유? = 출력 뉴런과 연동, 출력 뉴런은 Q값이니까
        # 두 번째 완전 연결인 fc2를 가져온다
        # 이제 Q값이 신경망의 출력 뉴런이 된다
        return q_values
        # 이 가치들을 반환해야 하므로 return이 사용
        # 가능한 각 행동의 Q값이 반환된다 / 왼쪽, 앞쪽, 오른쪽 중에 하나

        # 은닉 뉴런을 50개로 설정하고 싶다면 30으로 입력된 두 곳을 50으로 고친다
        # 연결을 추가해서 은닉층을 추가할수도 있다

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
    # __init__ 함수의 변수
    # 100개 사건의 100개의 변화가 담긴 메모리 / 용량 = 100, 늘리면 더 긴 memory가능
    # capacity는 100이 될거다 / 경험 리플레이를 100개의 변화로 대체할거라서
    # 즉, capacity는 다른 경험 replay나 다른memory를 만들 수 있는 변수이다.
        self.capacity = capacity
        # 사건 메모리에 저장할 변화의 최대 개수를 의미하는 용량이 된다
        # And, 리플레이 메모리 클래스의 객체를 만들 때 입력할 인수와 동일
        # self.capacity  = 객체에 붙는 변수의 이름
        # capacity = 메모리 클래스 객체를 만들 때 입력되는 인수 / 혼동 조심
        self.memory = []
        # 이 메모리는 100개의 지난 사건을 저장하므로 간단한 리스트가 된다
        # 동작 시작에는 메모리의 리스트가 비어있겠지만 미래 상태로 변할 때마다 매 순간 변화가 입력된다.
    
    def push(self, event):
    # push함수의 역할
    # 1. 메모리에 새로운 변화, 사건을 추가하는 것
    # 2. 메모리에 100개의 변화만 저장되게 하는 것 / 변화가 100개 이상이 될 수 있다.
    #  push함수의 인수
    # 1. 객체 변수를 의미하는 self
    # 2. 사건변수가 인수이자 입력 정보로 필요
    # 객체변수인 이 메모리에 입력정보를 추가한다 > 그러니 event를 입력
    # 메모리에 추가할 이 사건, 변화는 4개의 요소를 갖는다 / 이 부분은 나중에 언급 
    # 메모리에 사건을 추가해야 되니까 event를 입력 > 메모리가 용량요소를 가지도록 한다
        self.memory.append(event)
        # memory는 객체변수이므로 self.memory를 입력해야된다.
        # 메모리에 사건을 추가하기 위해서 append를 사용한다
        # append함수를 사용하려면 사건을 추가하려는 리스트를 먼저 입력해야 된다. / 이때 리스트는 당연히 메모리이다.
        # append 함수 내부에는 메모리에 추가할 입력 정보인 사건(event)을 입력한다
        # event를 입력하면 새로운 사건이 추가될 것이다.
        # 마지막 상태, 새로운 상태, 마지막 행동, 마지막 보상이 메모리에 추가된다

        # 두번째로 할 일은 메모리가 항상 같은 용량을 유지하게 한다
        # Ex. 용량을 십만이라고 가정 / 메모리에 항상 십만개의 변화, 사건만 저장되고 그 이상은 저장되지 않도록
        # 사건이 10만개가 되면 -> 메모리에도 10만개의 사건이 저장된다
        # 그러니 if 조건문을 사용해서 넘지 말아야 할 상한선을 설정
        # 상한선을 넘으면 메모리의 첫번째 변화나 사건을 삭제하고 len 함수를 입력해서 메모리 길이를 가져온다
        # 메모리에 저장된 요소의 개수를 의미 = 메모리의 길이
        if len(self.memory) > self.capacity:
            del self.memory[0]
	# self.memory의 요소 개수가 용량보다 커지면 첫번째 요소가 삭제되고 메모리가 항상 같은 용량을 유지한다
        # del : 삭제를 위해 사용
    
    def sample(self, batch_size):
    # sample 함수 : 일정 용량의 메모리에서 무작위로 샘플을 가져올 것이다 : 이걸 사용하면 딥 큐러닝 과정을 향상시킬 수 있다.
    # sample함수의 인수 : self, batch_size
    # batch_size : sample을 가져올 때 일정 크기로 가져오니까 sample 크기를 선택해야된다 (더 정확히는 배치 크기)
    
        samples = zip(*random.sample(self.memory, batch_size))
	# samples 변수 : 메모리의 샘플들을 가지고 있는 변수
	# 메모리에서 sample을 가져와야 되니까 메모리가 필요 + 배치크기도 필요
	# (일정 배치 크기의 샘플을 가져 올 거니까)
	# 무작위로 샘플을 가져오므로 라이브러리도 무작위로 불러온다
	# 무작위로 불러온 라이브러리(random)에서 sample함수를 사용한다
	# random.sample 함수는 고정된 배치 크기를 가지고 있는 메모리에서 무작위로 sample을 가져오는 역할
        # zip* 함수의 역할 / reshape함수와 역할이 같다
	# Ex. 일정 요소를 가진 리스트가 있을 때, (첫 번째 요소는 (1,2,3), 두 번째 요소는(4,5,6)이다.)
	# > 3개의 요소가 들어간 2개의 리스트가 존재
	# 여기에 *가 들어간 zip함수를 쓴다면? : zip(*list)는 형태가 다른 새로운 리스트와 같다
	# (1,4),(2,3),(3,6) 이런 다른 형태가 된다
	# 즉, zip* 함수는 리스트의 형태를 바꾼다
	# 그렇다면 zip(*list)함수를 왜 해야할까?
	# 이제 메모리에 사건을 추가해야 하는데 사건은 상태, 행동, 보상의 형태를 갖는다. / 하지만 알고리즘은 이런 형태를 원하지않아
	# 알고리즘이 원하는 sample형태 : 3개의 sample로 이루어진 형태
	# 상태 샘플, 행동 샘플, 보상 샘플로 이루어져 있다
	# 이 (1,2,3)가 상태1, 행동1, 보상1이고 (4,5,6)이 상태2, 행동2, 보상2 라고 할 때
	# 우리가 원하는 형태 : 상태 배치에 상태1, 상태2가 들어가고 행동 배치에는 행동1, 행동2가 보상 배치에는 보상1, 보상2가 들어가는 형태
	# 이런 형태를 알고리즘에서도 기대한다
	# 이 배치를 파이토치 변수로 압축할 수 있어서

	# 상태, 행동, 보상별로 각각의 배치를 따로 만들고 각 배치를 파이토치 변수에 따로 집어넣어서 각각의 경사를 구한다
	# 왜? : 서로 구분할 수 있어서
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
	# sample을 직접 반환할 수는 없다 > 왜? : sample을 파이토치 변수에 넣을거라서
	# 각각의 샘플에 이 작업을 하기 위해서 map함수를 사용한다
	# map함수 : 텐서와 경사를 포함하고 있는 토치 함수를 샘플로 mapping 해준다
# Implementing Deep Q Learning
# 인공지능이 마르코프 의사결정 과정에 기반을 둔다
class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
    # init 함수에서 변수를 정의한다 / 이 변수는 클래스에서 만들어질 미래 객체에 붙는다
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        action = probs.multinomial(1) # 1 is the number of samples to draw
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        
        self.optimizer.zero_grad()
        # td_loss.backward(retain_variables = True)
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
