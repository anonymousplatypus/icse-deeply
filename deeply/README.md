# Summary 

Code for [COCGN](https://github.com/utkd/cogcn) [paper](https://arxiv.org/pdf/2102.03827.pdf) with slight modifications to the front-end

To Run
```
python cogcn.py
```
#  Results

## Forward/Call edges only (no backward/return edges), No Root node

### JPETSTORE 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            3 |    2.276 |    0.15  |   0.063 |   1.819 |    2.667 |         9.84 |
|            5 |    1.965 |    0.159 |   0.086 |   2.79  |    2     |        11.12 |
|            7 |    1.785 |    0.191 |   0.062 |   2.778 |    1.286 |        12.85 |
|            9 |    2.174 |    0.394 |   0.16  |   5.499 |    2.778 |        13.85 |
|           11 |    1.863 |    0.379 |   0.162 |   6.111 |    1.909 |        14.98 |
|           13 |    1.993 |    0.496 |   0.173 |   6.992 |    2.077 |        15.88 |
|           15 |    1.729 |    0.428 |   0.148 |   6.182 |    1.667 |        16.71 |
|           17 |    1.903 |    0.561 |   0.163 |   7.319 |    1.824 |        19.03 |


### PLANTS 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            2 |    3.379 |    0.139 |   0.113 |   1.695 |    4.5   |         8.93 |
|            4 |    2.618 |    0.407 |   0.176 |   2.637 |    5     |         9.94 |
|            6 |    2.324 |    0.512 |   0.22  |   3.353 |    5.333 |        10.63 |
|            8 |    1.368 |    0.479 |   0.225 |   3.543 |    3.25  |        11.4  |
|           10 |    1.397 |    0.547 |   0.19  |   3.511 |    2.8   |        12.06 |
|           12 |    1.639 |    0.67  |   0.13  |   4.279 |    3.083 |        13.36 |


### DAYTRADER 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            3 |    2.18  |    0     |   0.094 |   2     |    0     |        11.84 |
|            5 |    1.894 |    0.277 |   0.088 |   2.626 |    7.6   |        13.93 |
|            7 |    1.089 |    0.002 |   0.085 |   3.667 |    0.143 |        16.15 |
|            9 |    1.036 |    0.004 |   0.078 |   3.885 |    0.222 |        16.85 |
|           11 |    1.007 |    0.126 |   0.058 |   3.55  |    1.636 |        17.98 |
|           13 |    0.787 |    0.006 |   0.071 |   4.462 |    0.231 |        19.77 |
|           15 |    0.661 |    0.004 |   0.048 |   3.709 |    0.133 |        20.65 |
|           17 |    0.91  |    0.168 |   0.08  |   5.889 |    1.294 |        22.44 |
|           19 |    0.941 |    0.342 |   0.125 |   7.993 |    2.474 |        23.06 |
|           21 |    1.087 |    0.394 |   0.136 |   8.538 |    2.714 |        25.25 |
|           23 |    1.01  |    0.366 |   0.12  |   8.041 |    2.261 |        25.36 |
|           25 |    1.094 |    0.405 |   0.134 |   9.47  |    1.96  |        26.94 |
|           27 |    0.998 |    0.411 |   0.133 |   9.676 |    1.889 |        28.88 |
|           29 |    1.016 |    0.46  |   0.139 |  10.337 |    2.069 |        29.71 |
|           31 |    0.932 |    0.501 |   0.102 |   9.5   |    1.968 |        29.83 |
|           33 |    1.137 |    0.572 |   0.119 |   9.724 |    2.212 |        30.86 |
|           35 |    1.088 |    0.602 |   0.1   |   9.36  |    2.2   |        32    |


### ACMEAIR 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            3 |    1.311 |    0.179 |   0.079 |   1.724 |    2     |         9.69 |
|            5 |    1.449 |    0.38  |   0.161 |   2.642 |    3.8   |        10.7  |
|            7 |    1.583 |    0.578 |   0.106 |   3.709 |    3.571 |        11.14 |
|            9 |    1.647 |    0.582 |   0.173 |   4.326 |    3     |        11.88 |
|           11 |    1.499 |    0.652 |   0.097 |   4.403 |    2.636 |        12.78 |


## With Forward/Call edges & Backward/Return edges, No Root Node

### JPETSTORE 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            3 |    1.986 |    0     |   0.016 |   1     |    0     |         9.98 |
|            5 |    1.37  |    0.122 |   0.052 |   1.916 |    1.4   |        11.91 |
|            7 |    1.645 |    0.154 |   0.1   |   2.805 |    1.286 |        13.32 |
|            9 |    1.97  |    0.224 |   0.168 |   5.342 |    1.444 |        14.97 |
|           11 |    2.222 |    0.41  |   0.163 |   6.484 |    2.091 |        16.19 |
|           13 |    2.329 |    0.577 |   0.17  |   7.023 |    2.538 |        16.75 |
|           15 |    2.16  |    0.556 |   0.108 |   7.053 |    2.133 |        17.85 |
|           17 |    1.953 |    0.606 |   0.078 |   6.329 |    2.059 |        20.12 |


### PLANTS 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            2 |    1.869 |    0.022 |   0.266 |   1.648 |    1     |         9.25 |
|            4 |    2.618 |    0.338 |   0.279 |   2.965 |    4.5   |        10.49 |
|            6 |    1.928 |    0.435 |   0.26  |   3.214 |    3.833 |        11.47 |
|            8 |    2.147 |    0.667 |   0.201 |   3.472 |    4.5   |        12.48 |
|           10 |    1.718 |    0.612 |   0.269 |   4.597 |    3.4   |        13.83 |
|           12 |    1.639 |    0.698 |   0.074 |   4.03  |    3.167 |        14.82 |



### DAYTRADER 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            3 |    2.185 |    0.002 |   0.176 |   2.996 |    0.333 |        12.83 |
|            5 |    1.308 |    0.002 |   0.072 |   2.909 |    0.2   |        15.02 |
|            7 |    1.121 |    0.002 |   0.091 |   3.996 |    0.143 |        17.11 |
|            9 |    0.872 |    0.002 |   0.057 |   2.8   |    0.111 |        17.25 |
|           11 |    0.808 |    0.002 |   0.045 |   2.909 |    0.091 |        19.19 |
|           13 |    0.763 |    0.002 |   0.048 |   3.909 |    0.077 |        20.3  |
|           15 |    0.733 |    0.006 |   0.077 |   5.376 |    0.2   |        21.94 |
|           17 |    0.622 |    0.038 |   0.061 |   4.596 |    0.176 |        23.79 |
|           19 |    0.614 |    0.069 |   0.09  |   6.877 |    0.526 |        25.36 |
|           21 |    0.899 |    0.198 |   0.103 |   7.443 |    1.143 |        26.19 |
|           23 |    1.033 |    0.415 |   0.12  |   9.296 |    2.087 |        26.77 |
|           25 |    1.049 |    0.424 |   0.133 |   9.691 |    2     |        27.11 |
|           27 |    1.132 |    0.52  |   0.142 |   9.891 |    2.444 |        28.21 |
|           29 |    1.206 |    0.527 |   0.108 |   9.688 |    2.345 |        30.24 |
|           31 |    1.024 |    0.498 |   0.117 |   9.766 |    1.935 |        30.79 |
|           33 |    1.333 |    0.706 |   0.091 |   8.2   |    2.727 |        31.85 |
|           35 |    1.364 |    0.714 |   0.082 |   9.613 |    2.486 |        34.19 |


### ACMEAIR 

|   Partitions |   BCS[-] |   ICP[-] |   SM[+] |   MQ[+] |   IFN[-] |   WC_time[-] |
|--------------|----------|----------|---------|---------|----------|--------------|
|            3 |    2.823 |    0.23  |   0.05  |   2.067 |    5     |        11.03 |
|            5 |    1.353 |    0.359 |   0.096 |   2.338 |    3     |        12.73 |
|            7 |    1.623 |    0.499 |   0.173 |   3.766 |    3.286 |        12.19 |
|            9 |    1.801 |    0.67  |   0.163 |   4.508 |    3.444 |        13.23 |
|           11 |    1.289 |    0.614 |   0.079 |   3.928 |    2.636 |        15.69 |
