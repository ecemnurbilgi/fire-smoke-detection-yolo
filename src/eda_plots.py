
import matplotlib.pyplot as plt

classes = ["Smoke", "Fire"]
counts = [7794, 9638]

plt.figure(figsize=(6,4))
plt.bar(classes, counts)
plt.title("Class Distribution (Train Set)")
plt.ylabel("Number of Instances")
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import numpy as np

categories = ["Small", "Medium", "Large"]

fire = [64.42, 29.84, 5.74]
smoke = [20.22, 23.88, 55.90]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, fire, width, label="Fire")
plt.bar(x + width/2, smoke, width, label="Smoke")

plt.xticks(x, categories)
plt.ylabel("Percentage (%)")
plt.title("Bounding Box Size Distribution (Train Set)")
plt.legend()

plt.tight_layout()
plt.show()