
# ##########################################################################
#
# A Basic Neural Network
# Todd S. Peterson
#
# December 18 2019    
#
#
#
#
# ##########################################################################
import math
import random

def Sigma (num):
      return 1/(1+math.e**(-1*num))

def rand():
      r = (random.random() -.5)/10
      r = round(r,7)
      #print (f"random = {r}")
      return r
      

GlobalError = .0002
#TrainingExamples = [[-.6,.2],[.8]],[[ .5,.5],[1.0]]

TrainingExamples = [[[.1,.1],[.1]],[[.9,.9],[.1]],
                    [[.1,.9],[.9]],[[.9,.1],[.9]]]

LearningRate = .1
NetworkSize = [2,4,1]
NumEpochs = 10

class Network:
      def __init__(self,Size):
            self.Size = Size
            #      if Size != 3:
            #            raise SizeError
            #except SizeError:
            #      print("The Network Size is not equal to 3")
            #      return
                  
            self.InputSize = self.Size[0]
            self.HiddenSize = self.Size[1]
            self.OutputSize = self.Size[2]
            self.OutputError = self.Size[2];
            self.HiddenError = self.Size[1];
            
                  
            self.Input = [0.0] * self.InputSize
            self.Output = [0.00] * self.OutputSize
            self.OutputError = [0.00] * self.OutputSize
            
            self.Hidden = [0.0] * self.HiddenSize
            self.HiddenError = [0.0] * self.HiddenSize
                        
            self.I = 0
            self.D = 0
            self.T = 1
            
            
            for i in range (self.InputSize):
                  self.Input[i] = TrainingExamples[self.D][self.I][i]
            
            #print(f"Input =  {self.Input}")            
            
                          
            
            self.setupWeights()            
            
      
      
      def setupWeights(self):
            
            self.HOWeights = []
            self.IHWeights = []            
            
            for h in range (self.HiddenSize):
                  self.IHWeights.append([])

            for h in (self.IHWeights):
                  for i in range (self.InputSize):
                        h.append(rand())


            for o in range (self.OutputSize):
                  self.HOWeights.append([])

            for o in (self.HOWeights):
                  for h in range (self.HiddenSize):
                        o.append(rand())


            #print(f"IHWeights = {self.IHWeights}")
            #print(f"HOWeights = {self.HOWeights}")

                        
                          
      
                  
      def Train(self):
            print("Starting Training")
            print("*****************")
            self.Error = 99.99
            while (self.Error > GlobalError):
                  for e in range(NumEpochs):
                        self.Epoch(TrainingExamples)
                        self.Error = self.ComputeError()
                  print(f"Error = {self.Error}")
            self.Demo()
            
            
            
      def Demo(self):
            print("*****Demo*****")
            for r in range(len(TrainingExamples)):
                  self.Input = TrainingExamples[r][self.I]
                  print(f"Input = {self.Input}")
                  self.FeedForward()
                  #self.computeOutput()
                  self.TargetArray = TrainingExamples[r][self.T]
                  #self.SetTargetArray(r)
                  print(f"Target = {self.TargetArray}") 
                  print(f"Output = {self.Output}")
                             
      def SetTargetArray(self, rnum):
            pass
                        
      def Epoch(self,TrainingExamples):
            #print("***Epoch***")
            NumRules = len(TrainingExamples)
            #NumRules = 2
            for rnum in range (NumRules):
                  #print(f"rnum = {rnum}")
                  self.Input = TrainingExamples[rnum][self.I]
                  #print(f"Input = {self.Input}")
                  #self.Output = TrainingExamples[rnum][1]
                  self.TargetArray = TrainingExamples[rnum][self.T]
                  #print(f"TargetArray = {self.TargetArray}")
                  self.FeedForward()
                  self.BackProp(rnum)
                  
      def ComputeError(self):                                   
            NTE = len(TrainingExamples)
            #print()
            Error = 0.00
            for rnum in range(NTE):
                  #print(f"RuleNum = {rnum}")
                  self.Input = TrainingExamples[rnum][0]
                  #self.Output = TrainingExamples[rnum][1]
                  self.TargetArray = TrainingExamples[rnum][self.T]
                  self.FeedForward()
                  #self.computeOutputErrors(rnum)
                  #print(f"Error.A = {Error}")
                  for o in range (self.OutputSize):
                        Error += (self.TargetArray[o] - self.Output[o])**2
                        #Error += (self.OutputError[o])
                        #print(f"Error.B = {Error}")
            #print(f"Error1 = {Error}")
            #Error = Error ** 2
            #print(f"Error2 = {Error}")
            Error *= .5
            #print(f"Error3 = {Error}")
            return Error
                  
                  
      
      
      def computeHidden(self):
            #print(f"***computeHidden***")
            #print(f"Hidden = {self.Hidden}")
            #print(f"Input = {self.Input}")
            #print(f"Input to Hidden Weights = {self.IHWeights }") 
            for h in range (self.HiddenSize):
                  #print(f"IHWeights[{h}] = {self.IHWeights[h]}")
                  #print(" =", end=' ') 
                  #print(self.IH)
                  self.Hidden[h] = 0.00
                  #print("self.Hidden[h] = 0")
                  for i in range (self.InputSize):
                        
                        self.Hidden[h] += self.IHWeights[h][i] * self.Input[i]
                        self.Hidden[h] = round(self.Hidden[h],7)
                        #print(f"hidden[{h}] += {self.IHWeights[h][i]} * {self.Input[i]}") 
                        #print(f"hidden unit [{h}] = {self.Hidden[h]}")
                  self.Hidden[h] = Sigma(self.Hidden[h])
            #print(f"Hidden = {self.Hidden}")
                                         
                        
      def computeOutput(self):          
            #print(f"HOWeights = {self.HOWeights}")
            
            for o in range(self.OutputSize):
                  #print(f"Output Size = {self.OutputSize}")
                  #print('Hidden =', end=' ')
                  #print(self.Hidden)
                  #print("HOWeights =", end=' ')
                  #print(self.HOWeights)
                  self.HO = self.HOWeights[o]
                  #print(self.HO)
                  self.Output[o] = 0.0
                  #print('Output = ', end = " ")
                  #print (self.Output)
                  for h in range (self.HiddenSize):
                        self.Output[o] += (self.HOWeights[o][h] * self.Hidden[h])
                        self.Output[o] = round(self.Output[o],7)                    
                        #print(f"Output[{o}] += {self.HOWeights[o][h]} * {self.Hidden[h]}") 
                        #print(f"Output unit[{o}] = {self.Output[o]}")
                  self.Output[o] = Sigma (self.Output[o])
                  #print(f"Output[{o}] = {self.Output[o]}")
      
      
      def FeedForward(self): # work here
            self.computeHidden()
            #self.Hidden = [3,-4]
            #print("***Print FeedForward***")
            #print (f"Hidden = {self.Hidden}")
                   
            
            self.computeOutput()
            
            #print("***after ComputeOutput ***")
            #print(f"Output = {self.Output}")
            
            
            
      def BackProp(self, rnum):
            self.computeOutputErrors(rnum)
            self.computeOutputToHiddenWeights()
            self.computeHiddenErrors()
            self.computeInputToHiddenWeights()
            
      def computeOutputErrors(self, ruleNum):
            #print("***computeOutputErrors***")
            #print(f"ruleNum = {ruleNum}")
            self.TargetArray = TrainingExamples[ruleNum][1]
            #print(f"TargetArray = {self.TargetArray}")
            for(o) in range(self.OutputSize):
                  #print(f"self.Output = {self.Output}")
                  self.OutputError[o] = 0
                  out = self.Output[o]
                  #print(f"out = {out}")
                  #print(f"1-out = {1-out}")
                  #print(f"out * (1-out) = {out*(1-out)}")
                  #print(f"Target - out = {self.TargetArray[o] - out}")
                  #self.OutputError[o] = (1-out) * out * 
                  self.OutputError[o] += (self.TargetArray[o] - out)
                  self.OutputError[o]= round(self.OutputError[o],7)
            #print(f"Output = {self.Output}")
            #print(f"TargetArray = {self.TargetArray}")
            #print(f"OutputError = {self.OutputError}")
      
      def computeHiddenErrors(self):
            #print(f"HiddenError = {self.HiddenError}")
            for h in range(self.HiddenSize):
                  self.HiddenError[h] = 0.0
                  hid = self.Hidden[h]
                  for o in range(self.OutputSize):
                        self.HiddenError[h] += self.OutputError[o] * self.HOWeights[o][h]
                        #print (f"HiddenError[{h}] += {self.OutputError[o]} * {self.HOWeights[o][h]}")
                  self.HiddenError[h] = (1-hid) * (hid) * self.HiddenError[h]
                  self.HiddenError[h] = round(self.HiddenError[h], 7)
                        
            #print(f"HiddenError = {self.HiddenError}")
      
      
      def computeInputToHiddenWeights(self):
            #print(f"Input = {self.Input}")
            #print(f"InputToHiddenWeights = {self.IHWeights}")
            #print(f"HiddenError = {self.HiddenError}")
            for i in range(self.InputSize):
                  #r = rnum
                  for h in range(self.HiddenSize):
                        #print(f"IHWeights[h][i] = {self.IHWeights[h][i]}")
                        self.IHWeights[h][i] += LearningRate * self.Input[i] * self.HiddenError[h]
                        self.IHWeights[h][i] = round(self.IHWeights[h][i], 7)
                        #print(f"self.IHWeights[{h}][{i}] += {LearningRate} * {self.Input[i]} * {self.HiddenError[h]}")
                  #print(f"= {self.IHWeights[h][i]}")
            #print(f"IHweights = {self.IHWeights}")
            
      
      
      
      def computeOutputToHiddenWeights(self):
            #print("***computeOutputToHiddenWeights***")
            #print(f"HOWeights =  {self.HOWeights}")
            #print(f"OutputError = {self.OutputError}")
            #print(f"Hidden = {self.Hidden}")
            for(h) in range(self.HiddenSize):
            #      print(f"self.hidden[{h}] = {self.Hidden[h]}")
                  for(o) in range (self.OutputSize):
                        self.HOWeights[o][h] += self.OutputError[o] * self.Hidden[h] * LearningRate
             #           print(f"self.HOWeights[{o}][{h}] += {self.OutputError[o]} * {self.Hidden[h]} * {LearningRate}")
                        self.HOWeights[o][h] = round(self.HOWeights[o][h], 7)
              #          print(f"HOWeights[{h}] = {self.HOWeights[o][h]}")      
                  
      


def main():
      Net = Network(NetworkSize)
      #print(f"Input = {Net.Input}")
      #print(f"Hidden = {Net.Hidden}")
      #Net.Epoch(TrainingExamples)
      
      #Net.computeHidden()
      #print("***Finished***")
      Net.Train()
      #Net.FeedForward()
      #Net.computeOutputErrors()
      #print(Net.HOWeights)
      #Net.computeOutputToHiddenWeights()
      #Net.computeHidden()
      #Net.computeOutput()
      #Net.BackProp(0)
      
      
if __name__ == "__main__":
      main()


