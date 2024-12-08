# qcktsynthesis
Quantum Circuit Synthesis  of binary functions using Reed Muller encoding

# Papers Referred
1. An Algorithm for Synthesis of Reversible Logic Circuits. Pallav Gupta, Student Member, IEEE, Abhinav Agrawal, and Niraj K. Jha, Fellow, IEEE, IEEE TRANSACTIONS ON COMPUTER-AIDED DESIGN OF INTEGRATED CIRCUITS AND SYSTEMS, VOL. 25, NO. 11, NOVEMBER 2006
2. Gate Optimization of NEQR Quantum Circuits via PPRM Transformation Shahab Iranmanesh1, Hossein Aghababa2, and Kazim Fouladi3, arXiv:2409.14629v1 [quant-ph] 22 Sep 2024

# Running the program
1. Just download all the python files in a folder.
2. Navigate to the directory
3. Type "python revopt.py"

# Expected output
2024-12-09 00:17:38,485 - root - CRITICAL - <module> - Testing: This is a critical message

Input function (Truth tables)
[0, 0, 1, 1, 0, 1, 0, 1]
[0, 1, 1, 0, 0, 0, 1, 1]
[1, 1, 1, 1, 0, 0, 0, 0]

Best solution path:
=======Qubit State=======

[[1], [0, 2], [1, 2]]

[[0], [1], [0, 2]]

[1, [2]]

gate operation :  2  =  2  xor  []

=======Qubit State=======

[[0], [0, 2], [1, 2]]

[[1], [0, 2]]

[[2]]

gate operation :  1  =  1  xor  [0, 2]

=======Qubit State=======

[[0], [1, 2]]

[[1]]

[[2]]

gate operation :  0  =  0  xor  [1, 2]

=======Qubit State=======

[[0]]

[[1]]

[[2]]

How to understand this output:
Qubits are numbered starting from 0 upwards:
In a Qubit state and the input function truth table there are as many rows as qubits
First row is for qubit 0, 2nd row for qubit 1 and so on

# Limitations
Only 1 test case has been tested. It should work. Try changing the inputs and see
