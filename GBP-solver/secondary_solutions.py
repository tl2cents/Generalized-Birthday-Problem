# 20241203-112221_sol_count_96_7.pkl
import pickle

with open('./GBP-solver/20241203-112221_sol_count_96_7.pkl', 'rb') as f:
    data = pickle.load(f)

n = 96 
k = 7
K = 2**k
print(f"n = {n}, k = {k}")
print("Avg. #Sol(t=0): ", data[K] / 10000)
print("Avg. #Sec-Sol(t=1): ", data[K - 2] / 10000)
total = sum(data[k] for k in data.keys())
print("Avg. #Sec-Sol(t>=2): ",(total - data[K] -data[K-2]) /10000)
print("Avg. #All-Sol(t>=2): ", total / 10000)

# 20241120-110719_sol_count_160_9.pkl
with open('./GBP-solver/20241120-110719_sol_count_160_9.pkl', 'rb') as f:
    data = pickle.load(f)

n = 160
k = 9
K = 2**k
print(f"n = {n}, k = {k}")
print("Avg. #Sol(t=0): ", data[K] / 10000)
print("Avg. #Sec-Sol(t=1): ", data[K - 2] / 10000)
total = sum(data[k] for k in data.keys())
print("Avg. #Sec-Sol(t>=2): ",(total - data[K] - data[K-2]) / 10000)
print("Avg. #All-Sol(t>=2): ", total / 10000)
