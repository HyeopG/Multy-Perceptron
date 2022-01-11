# Multi-Perceptron  XOR
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Multi_Perceptron:
    def __init__(self):
        self.input_layer_num = 35   # input 갯수
        self.h1_layer_num = 20      # 1층 layer 갯수
        self.h2_layer_num = 15      # 2층 layer 갯수
        self.output_layer_num = 10        # 결과 layer 갯수
        self.x = []         # 입력값
        self.target = []    # 목표값
        self.w1 = self.SetWeight(self.input_layer_num, self.h1_layer_num)           # 1층 weight
        self.w2 = self.SetWeight(self.h1_layer_num, self.h2_layer_num)              # 2층 layer Weight
        self.w3 = self.SetWeight(self.h2_layer_num, self.output_layer_num)          # 3층 layer Weight
        self.result = [[0.0 for i in range(self.h1_layer_num)], [0.0 for i in range(self.h2_layer_num)],
                       [0.0 for i in range(self.output_layer_num)]]
        self.loss = [[0.0 for i in range(self.h1_layer_num)], [0.0 for i in range(self.h2_layer_num)],
                     [0.0 for i in range(self.output_layer_num)]]
        self.result_list_1 = []   # 1층 결과 모음
        self.result_list_2 = []   # 2층 결과 모음
        self.b = 1     # bias는 항상 1
        self.running = 0.1  # 이동 값
        self.epoch = 5000   # 반복 횟수
        self.tss = 0
        self.p_tss = []
        self.p_epoch = []
        # self.path = "0.txt"
        # self.SetFile(self.path)     # 파일 열면서 초기화!

        self.Run() # 시작.

        while True:
            if self.UI() == 'q':
                break


    def UI(self):
        try:
            print("바꾸실 파일 이름을 입력해주세요. ( ex) test.txt ) : ")
            print("뒤로 가시려면 q를 입력해주세요. ")
            path = input()
            if path == 'q':
                return 'q'
            self.SetFile(path)

            count = 0
            for i in range(10):
                print("(", i, ")", end="")
                print("output:", self.Exam(self.x)[count], "\t\tth(output) :", self.Threshold(self.Exam(self.x)[count]))
                count += 1
        except:
            print("Error")
            return

    def SetWeight(self, in_num, out_num):
        t_arr = []
        for i in range(out_num):
            arr = []
            for j in range(in_num+1):   # bias 포함 -> +1
                arr.append(random.uniform(-0.1, 0.1))
            t_arr.append(arr)
        return t_arr

    def SetFile(self, path):
        self.target = [0 for i in range(10)]    # 목표 초기화

        f = open(path, 'r', encoding='UTF-8')   # 파일 읽기
        target = f.readline()                   # 한 줄 읽기

        # 목표값 설정
        if target[0] != '?':
            self.target[int(target[0])] = 1
        # 입력값 설정
        lines = f.read().replace('\n', ' ')     # '\n'를 ' '로 교체
        str_x = lines.split(" ")               # ' '마다 나누기
        self.x = list(map(float, str_x))                # str문자를 float형으로

        f.close()

    def Threshold(self, result):
        if result < 0.5:
            return 0
        else:
            return 1

    def Sigmoid(self, result):
        return 1 / (1+np.exp(-result))

    def D_Sigmoid(self, sig):
        return sig * (1-sig)

    def WeightedSum(self, x, layer, number):    # layer = 층, number = 몇번째 것인지
        w = []
        # 웨이트 넣기.
        if layer == 1:
            for i in range(len(self.w1[number])):        # 받는 Weight 갯수
                w.append(self.w1[number][i])
        elif layer == 2:
            for i in range(len(self.w2[number])):
                w.append(self.w2[number][i])
        elif layer == 3:
            for i in range(len(self.w3[number])):
                w.append(self.w3[number][i])
        else:
            print("WeightedSum - layer 입력 에러")
            return

        temp_x = x + [self.b]
        np_x = np.array(temp_x)
        np_w = np.array(w)

        result = np.sum(np_x * np_w)

        return result

    def Exam(self, x):
        self.result_list_1 = []
        self.result_list_2 = []
        # 1층 연산
        for i in range(self.h1_layer_num):
            self.result[0][i] = self.Sigmoid(self.WeightedSum(x, 1, i))
            self.result_list_1.append(self.result[0][i])   # 1층 연산 끝낸 값 모음

        # 2층 연산
        for i in range(self.h2_layer_num):
            self.result[1][i] = self.Sigmoid(self.WeightedSum(self.result_list_1, 2, i))
            self.result_list_2.append(self.result[1][i])

        # 3층 연산
        for i in range(self.output_layer_num):
            self.result[2][i] = self.Sigmoid(self.WeightedSum(self.result_list_2, 3, i))

        return self.result[2]

    def Backpropagation(self, x, target):
        self.loss[0] = [0.0 for i in range(self.h1_layer_num)]  # 1층 loss 초기화
        self.loss[1] = [0.0 for i in range(self.h2_layer_num)]  # 2층 loss 초기화
        pss_sum = 0

        # result loss 구하기
        for i in range(self.output_layer_num):
            self.loss[2][i] = target[i] - self.result[2][i]
            self.WeightedUpdate(self.result_list_2, self.loss[2][i], 3, i)  # 3층 Weight 업데이트
            pss_sum += math.pow(self.loss[2][i], 2)

        # h2 loss 구하기
        self.LossSum(self.loss[2], self.loss[1], 3)

        # h1 loss 구하기
        self.LossSum(self.loss[1], self.loss[0], 2)

        # 2층 Weight 업데이트
        for i in range(self.h2_layer_num):
            self.WeightedUpdate(self.result_list_1, self.loss[1][i], 2, i)

        # 1층 Weight 업데이트
        for i in range(self.h1_layer_num):
            self.WeightedUpdate(x, self.loss[0][i], 1, i)

        pss = pss_sum / 10
        self.tss += pss
        print("pss :", pss)

    def LossSum(self, output_loss, input_loss, layer):
        if layer == 2:
            for i in range(len(input_loss)):
                for j in range(len(output_loss)):
                    input_loss[i] += output_loss[j] * self.w1[j][i]
                input_loss[i] *= self.D_Sigmoid(self.result[layer-2][i])
        elif layer == 3:
            for i in range(len(input_loss)):
                for j in range(len(output_loss)):
                    input_loss[i] += output_loss[j] * self.w2[j][i]
                input_loss[i] *= self.D_Sigmoid(self.result[layer-2][i])
        else:
            print("LossSum - layer 입력 오류")

    def WeightedUpdate(self, x, loss, layer, number):
        # Weight Update
        if layer == 1:
            for i in range(len(self.w1[number])):        # 받는 Weight 갯수
                if i != len(self.w1[number])-1:             # -1 -> bias는 따로 계산해야해서
                    self.w1[number][i] += self.running * loss * x[i]
                else:

                    self.w1[number][i] += self.running * loss * self.b

        elif layer == 2:
            for i in range(len(self.w2[number])):
                if i != len(self.w2[number])-1:
                    self.w2[number][i] += self.running * loss * x[i]
                else:
                    self.w2[number][i] += self.running * loss * self.b

        elif layer == 3:
            for i in range(len(self.w3[number])):
                if i != len(self.w3[number])-1:
                    self.w3[number][i] += self.running * loss * x[i]
                else:
                    self.w3[number][i] += self.running * loss * self.b
        else:
            print("WeightedUpdate - layer 입력 에러")
            return

    def Run(self):
        for epoch in range(self.epoch):
            self.tss = 0
            count = 0
            for i in range(10):
                filename = str(count) + ".txt"
                self.SetFile(filename)
                self.Exam(self.x)
                self.Backpropagation(self.x, self.target)
                count += 1

            self.Show()
            self.p_tss.append(self.tss/10)
            self.p_epoch.append(epoch)
            # print("result\n", self.result[2], "\n", self.result[1], "\n", self.result[0])
            # print("loss\n", self.loss[2], "\n", self.loss[1], "\n", self.loss[0])
            # print("weight\n", self.w1, "\n", self.w2, "\n", self.w3)
            print("epoch :", epoch+1, "\ttss :", self.tss / 10, "\n")

            if self.tss < 0.01:
                self.Mat()
                break
            elif epoch == self.epoch-1:
                self.Mat()

    def Show(self):
        try:
            count = 0
            for i in range(10):
                filename = str(count) + ".txt"
                self.SetFile(filename)
                print("(", i, ")", end="")
                print("target :", self.target[count], "\toutput :", self.Exam(self.x)[count], "\tth(output) :", self.Threshold(self.Exam(self.x)[count]))
                count += 1
        except:
            print("Error : 입력 에러 발생.")
            return

    def Mat(self):
        plt.plot(self.p_epoch, self.p_tss)
        plt.xlabel("epoch")
        plt.ylabel("M_tss", rotation = 0, labelpad=15)
        plt.show()

Multi_Perceptron()