import random

a0 = 4
a1 = 2
a2 = 6
a3 = 3
X1 = []
X2 = []
X3 = []
Xn1 = []
Xn2 = []
Xn3 = []
Y = []
Y1 = []
Y2 = []
print("Постійні коефіцієнти")
print("a0 = ", a0)
print("a1 = ", a1)
print("a2 = ", a2)
print("a3 = ", a3)

for i in range(0,8):
    X1.append(random.randint(1, 21))
    X2.append(random.randint(1, 21))
    X3.append(random.randint(1, 21))
print("X1: " + str(X1))
print("X2: " + str(X2))
print("X3: " + str(X3))

print("Рівняння регресії:")
for i in range(8):
    Y.append(a0 + a1*X1[i] + a2*X2[i] + a3*X3[i])
print("Y: " + str(Y))

print("Нульовий рівень кожного фактору:")
X01 = (max(X1)+min(X1))/2
X02 = (max(X2)+min(X2))/2
X03 = (max(X3)+min(X3))/2
print("X01 = ", X01)
print("X02 = ", X02)
print("X03 = ", X03)

dX1 = X01-min(X1)
dX2 = X02-min(X2)
dX3 = X03-min(X3)
print("dX1 = ", dX1)
print("dX2 = ", dX2)
print("dX3 = ", dX3, '\n')

for i in range(0, 8):
    Xn1.append((X1[i] - X01)/dX1)
    Xn2.append((X2[i] - X02)/dX2)
    Xn3.append((X3[i] - X03)/dX3)
print("Xn1: " + str(Xn1))
print("Xn2: " + str(Xn2))
print("Xn3: " + str(Xn3) + '\n')

Y_et = a0 + a1*X01 + a2*X02 + a3*X03
def listsum(numList):
    theSum = 0
    for i in numList:
        theSum = theSum + i
    theSum1 = theSum/8
    return theSum1
r = listsum(Y)
print("Середнє значення Y ", r)
for i in range(8):
    k = Y[i] - r
    Y1.append(k)
for n in Y1:
    if n > 0:
        Y2.append(n)
o = r + min(Y2)
print("Завдання по варіанту : ",o)
print("Yэт: " + str(Y_et))
