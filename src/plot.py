import matplotlib.pyplot as plt

x = [0, 0.5, 0.9]
y1 = [79.67, 79.94, 79.67]
y2 = [81.48, 80.11, 77.61]

plt.plot(x, y1, 'o-', label='test accuracy (%)')
plt.plot(x, y2, 'o-', label='test f-score (%)')
plt.xticks(x)
for i in range(len(x)):
    plt.text(x[i], y1[i], str(y1[i]), ha='center', va='bottom')
    plt.text(x[i], y2[i], str(y2[i]), ha='center', va='top')
plt.xlabel('Dropout rate')
plt.ylabel('Test metric')
plt.legend()
plt.show()

# 0: 79.67 81.48
# 0.9: 79.67 77.61