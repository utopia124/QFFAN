import math


def sparse_adversarial_loss(pred, y, lamda):
    n = len(pred)
    loss = 0
    for i in range(n):
        if y[i] == 0:
            loss += -math.log(1 - pred[i])
        else:
            loss += -lamda * math.log(pred[i])
    loss = loss / n
    return loss


pred = [0.123, 0.234, 0.345, 0.123, 0.123, 0.123, 0.123]
y = [0, 0, 1, 1, 0, 0, 1]
loss1 = sparse_adversarial_loss(pred, y, 8)
print(loss1)
