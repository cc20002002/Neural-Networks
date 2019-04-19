import matplotlib.pyplot as plt
import numpy as np
f = open('8987.json.csv', 'r')
x = f.readlines()
f.close()
acc_8987 = list(map(float, x))[:75]

f = open('batch_size_60000.json.csv', 'r')
x = f.readlines()
f.close()
acc_batch_size_60000 = list(map(float, x))[:75]

f = open('no_weigh_decay.json.csv', 'r')
x = f.readlines()
f.close()
acc_no_weigh_decay = list(map(float, x))[:75]

f = open('no_momentum.json.csv', 'r')
x = f.readlines()
f.close()
acc_no_momentum = list(map(float, x))[:75]

f = open('dropout_05.json.csv', 'r')
x = f.readlines()
f.close()
acc_dropout_05 = list(map(float, x))[:75]

f = open('bn.csv', 'r')
x = f.readlines()
f.close()
acc_bn = list(map(float, x))[:75]


f = open('training.csv', 'r')
x = f.readlines()
f.close()
training_bn = list(map(float, x))[:75]

figure = plt.figure()
x = np.arange(200)

plt.plot(training_bn, '+')
plt.plot(acc_8987)
plt.plot(acc_dropout_05, '--')
plt.plot(acc_no_momentum, '--')
plt.plot(acc_no_weigh_decay, '--')
plt.plot(acc_batch_size_60000, '--')
plt.plot(acc_bn, '--')

plt.ylabel("Accuracy(%)")
plt.xlabel('Epoch')
plt.ylim(70,96)
plt.legend(['Benchmark (training accuracy)','Benchmark (CV accuracy)', 'Drop out rate increased to 0.5', 'Without momentum', 'Without weight decay', 'Batch training', 'Without BN'], loc='lower right')
plt.grid(linestyle='--')
plt.show()
figure.set_size_inches(6,5)
figure.savefig("acc.pdf", bbox_inches='tight')
figure.savefig("acc.eps", bbox_inches='tight')