import time

# time how long it takes to run the code
start = time.time()
total = 0
for i in range(1, 10000):
    for j in range(1, 10000):
        total += i + j

print(f"The result is {total} and it took {time.time() - start} seconds to run.")