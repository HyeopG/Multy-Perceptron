# Multi-Perceptron  XOR
import random
import math
import numpy as np
import matplotlib.pyplot as plt

class Multi_Perceptron:
    def __init__(self):
        self.input_layer_num = 784   # input 갯수
        self.h1_layer_num = 300      # 1층 layer 갯수
        self.h2_layer_num = 100      # 2층 layer 갯수
        self.output_layer_num = 10        # 결과 layer 갯수
        self.target = []    # 목표값
        self.dataset = []   # mnist
        self.size = 0
        self.w1 = self.SetWeight(self.input_layer_num, self.h1_layer_num)           # 1층 weight
        self.w2 = self.SetWeight(self.h1_layer_num, self.h2_layer_num)              # 2층 layer Weight
        self.w3 = self.SetWeight(self.h2_layer_num, self.output_layer_num)          # 3층 layer Weight
        self.result = [[0.0 for i in range(self.h1_layer_num)], [0.0 for i in range(self.h2_layer_num)],
                       [0.0 for i in range(self.output_layer_num)]]
        self.loss = [[0.0 for i in range(self.h1_layer_num)], [0.0 for i in range(self.h2_layer_num)],
                     [0.0 for i in range(self.output_layer_num)]]
        self.result_list_1 = []   # 1층 결과 모음
        self.result_list_2 = []   # 2층 결과 모음
        self.b = 5     # bias는 항상 1
        self.running = 0.03  # 이동 값
        self.epoch = 10   # 반복 횟수
        self.tss = 0
        self.test_result = []

        self.Run() # 시작.

        # while True:
        #      if self.Test() == 'q':
        #          break

    def Test(self):
        try:
            # print("테스트를 진행합니다. 진행하시려면 아무 키나 입력해주세요. 종료 -> q")
            # sw = input()
            #
            # if sw == 'q':
            #     return 'q'

            self.SetMnist("MNIST\\t10k-images.idx3-ubyte", "MNIST\\t10k-labels.idx1-ubyte")

            # print("바꾸실 이미지 파일 이름을 입력해주세요. ( ex) test.idx3-ubyte ) : ")
            # print("뒤로 가시려면 q를 입력해주세요. ")
            # imgPath = input()
            #
            # print("바꾸실 라벨링 파일 이름을 입력해주세요. ( ex) test.idx1-ubyte ) : ")
            # print("뒤로 가시려면 q를 입력해주세요. ")
            # labelPath = input()

            # if imgPath == 'q' or labelPath == 'q':
            #     return 'q'

            # self.SetMnist(imgPath, labelPath)
            self.test_result = []
            self.tss = 0

            for i in range(self.size):
                if i % 1000 == 0:
                    print(i+1, "\t")
                x = self.dataset[i]
                self.Exam(x)

                count = 0
                max = self.result[2][0]
                select = 0  # 선택
                for result in self.result[2]:
                    if max < result:        # 가장 큰 결과값 선택!
                        max = result
                        select = count
                    count += 1
                if select == self.target[i].index(1):    # 결과숫자와 타겟숫자 비교
                    self.test_result.append(1)
                else:
                    self.test_result.append(0)

            print("total :", self.size, "right :", self.test_result.count(1))

        except:
            print("Error")
            return

    # 아직 사용안함.
    def SaveWeight(self):
        with open("MNIST\\w1", mode="w+") as w1:
            w1.write(str(self.w1))
            w1.close()
        with open("MNIST\\w2", mode="w+") as w2:
            w2.write(str(self.w2))
            w2.close()
        with open("MNIST\\w3", mode="w+") as w3:
            w3.write(str(self.w3))
            w3.close()

    def SetWeight(self, in_num, out_num):
        t_arr = []
        for i in range(out_num):
            arr = []
            for j in range(in_num+1):   # bias 포함 -> +1
                arr.append(random.uniform(-0.1, 0.1))
            t_arr.append(arr)
        return t_arr

    def SetMnist(self, imgPath, labelPath):
        # 두개의 파일 중 하나라도 없으면 리턴
        if imgPath == "" or labelPath == "":
            return

        # img dataset에 넣기
        with open(imgPath, mode='rb') as f:
            data = f.read(16)  # 헤더읽기

            # sizeData, Dsize = 이미지전체 개수!
            sizeData = (hex(data[4]) + hex(data[5]) + hex(data[6]) + hex(data[7])).split("0x")
            sizeData.pop(0)

            Dsize = int("0x" + sizeData[0] + sizeData[1] + sizeData[2] + sizeData[3], 16)

            # dataX, dataY = x와 y의 크기
            sizeX = (hex(data[8]) + hex(data[9]) + hex(data[10]) + hex(data[11])).split("0x")
            sizeX.pop(0)
            sizeY = (hex(data[12]) + hex(data[13]) + hex(data[14]) + hex(data[15])).split("0x")
            sizeY.pop(0)

            dataX = int("0x" + sizeX[0] + sizeX[1] + sizeX[2] + sizeX[3], 16)
            dataY = int("0x" + sizeY[0] + sizeY[1] + sizeY[2] + sizeY[3], 16)

            # size = 이미지 전체 갯수
            self.size = Dsize

            # dateset에 mnist이미지 삽입.
            self.dataset = []
            count = 0
            data = f.read()
            for size in range(Dsize):
                savedata = []
                for x in range(dataX * dataY):
                    savedata.append(data[count] / 255.0)
                    count += 1
                self.dataset.append(savedata)
            f.close()

        # label target에 넣기.
        with open(labelPath, mode='rb') as f:
            data = f.read(8)    # 헤더 읽기

            # sizeData, Dsize = 이미지전체 개수!
            sizeData = (hex(data[4]) + hex(data[5]) + hex(data[6]) + hex(data[7])).split("0x")
            sizeData.pop(0)

            Dsize = int("0x" + sizeData[0] + sizeData[1] + sizeData[2] + sizeData[3], 16)

            # target에 mnist라벨 삽입.
            self.target = []
            data = f.read()
            for size in range(Dsize):
                target = [0 for i in range(10)]  # 목표 초기화
                target[int(data[size])] = 1
                self.target.append(target)
            f.close()

    def PattonError(self):
        pss_sum = 0
        for i in range(self.output_layer_num):
            pss_sum += math.pow(self.loss[2][i], 2)
        return pss_sum / self.output_layer_num

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

        # result loss 구하기
        for i in range(self.output_layer_num):
            self.loss[2][i] = target[i] - self.result[2][i]
            self.WeightedUpdate(self.result_list_2, self.loss[2][i], 3, i)  # 3층 Weight 업데이트

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
        imgPath = "MNIST\\train-images.idx3-ubyte"
        labelPath = "MNIST\\train-labels.idx1-ubyte"
        self.SetMnist(imgPath, labelPath)

        for epoch in range(self.epoch):
            self.tss = 0
            for i in range(self.size):
                print(i, "\t", end="")
                x = self.dataset[i]
                self.Exam(x)
                self.Backpropagation(x, self.target[i])
                pss = self.PattonError()
                self.tss += pss
                print("pss :", pss)
            print("epoch :", epoch+1, "\ttss :", self.tss / self.size, "\n")

            if self.tss < 0.01:
                break
        self.Test()   # 임시로 만든 테스트.
        print("EndTest")

Multi_Perceptron()