# Multi-Perceptron  XOR
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Multi_Perceptron:
    def __init__(self):
        self.x = [(0, 0), (0, 1), (1, 0), (1, 1)]       # 입력값
        self.target = [0, 1, 1, 0]      # 목표값
        self.loss = [[0.0, 0.0], [0.0]]
        self.result = [[0.0, 0.0], [0.0]]
        self.w1 = [[round(random.random(), 1) - 0.5, round(random.random(), 1) - 0.5], [round(random.random(), 1) - 0.5, round(random.random(), 1) - 0.5]]    # input Weight
        self.w2 = [round(random.random(), 1) - 0.5, round(random.random(), 1) - 0.5]    # hidden layer Weight
        self.b = 1     # bias는 항상 1
        self.bw = [[round(random.random(), 1) - 0.5, round(random.random(), 1) - 0.5], [round(random.random(), 1) - 0.5]]    # bias Weight
        self.running = 0.1  # 이동 값
        self.epoch = 10000   # 반복 횟수
        self.tss = 0
        self.p_tss = []
        self.p_epoch = []
        self.Run() # 시작.

    def Threshold(self, result):
        if result < 0.5:
            return 0
        else:
            return 1


    def Sigmoid(self, result):
        return 1 / (1+np.exp(-result))

    def D_Sigmoid(self, sig):
        return sig * (1-sig)

    def WeightedSum(self, x, layer, number=1):    # layer = 층, number = 알파(?) 갯수
        temp_x = [x[0], x[1]]
        w = []
        # 웨이트 넣기.
        if layer == 1:
            for i in range(len(self.w1[number-1])):        # 받는 Weight 갯수
                w.append(self.w1[number-1][i])
        elif layer == 2:
            for i in range(len(self.w2)):
                w.append(self.w2[i])
        else:
            print("error : layer 입력 에러")
            return

        temp_x.append(self.b)
        np_x = np.array(temp_x)

        w.append(self.bw[layer-1][number-1])
        np_w = np.array(w)

        result = np.sum(np_x * np_w)

        return result

    def Exam(self, x):

        # 1층 연산
        self.result[0][0] = self.Sigmoid(self.WeightedSum(x, 1, 1))
        self.result[0][1] = self.Sigmoid(self.WeightedSum(x, 1, 2))

        # 2층 연산
        result_list = [self.result[0][0], self.result[0][1]]   # 1층 연산 끝낸 값 모음
        self.result[1][0] = self.Sigmoid(self.WeightedSum(result_list, 2))

        return self.result[1][0]

    def Backpropagation(self, x, target):
        # loss 구하기
        self.loss[1][0] = target - self.result[1][0]

        # 업데이트를 먼저하고 밑에 수정.
        result_list = [self.result[0][0], self.result[0][1]]         # 1층 연산 끝낸 값 모음
        self.WeightedUpdate(result_list, self.loss[1][0], 2)         # 2층 업데이트

        self.loss[0][0] = (self.loss[1][0] * self.w2[0]) * self.D_Sigmoid(self.result[0][0])
        self.loss[0][1] = (self.loss[1][0] * self.w2[1]) * self.D_Sigmoid(self.result[0][1])

        self.WeightedUpdate(x, self.loss[0][0], 1, 1)
        self.WeightedUpdate(x, self.loss[0][1], 1, 2)

        pss = math.pow(self.loss[1][0], 2)
        self.tss += pss
        print("pss :", pss)

    def WeightedUpdate(self, x, loss, layer, number=1):
        # Weight Update
        if layer == 1:
            for i in range(len(self.w1[number-1])):        # 받는 Weight 갯수
                self.w1[number-1][i] += self.running * loss * x[i]
        elif layer == 2:
            for i in range(len(self.w2)):
                self.w2[i] += self.running * loss * x[i]
        else:
            print("error : layer 입력 에러")
            return

        self.bw[layer-1][number-1] += self.running * loss * self.b

    def Run(self):
        for epoch in range(self.epoch):
            self.tss = 0
            count = 0
            for x1, x2 in self.x:
                self.Exam([x1, x2])
                self.Backpropagation([x1, x2], self.target[count])
                count += 1

            self.Show()
            self.p_tss.append(self.tss/4)
            self.p_epoch.append(epoch)
            print("epoch :", epoch, "tss :", self.tss / count, "\n")

            if self.tss < 0.01:
                self.Mat()
                break
            elif epoch == self.epoch-1:
                self.Mat()


    def Show(self):
        try:
            count = 0
            for x1, x2 in self.x:
                print("(", x1, ",", x2, ")", end="")
                print("target :", self.target[count], "output :", self.Exam([x1, x2]), "th(output) :", self.Threshold(self.Exam([x1, x2])))
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