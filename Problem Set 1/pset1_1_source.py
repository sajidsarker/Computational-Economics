#!/usr/bin/python3

# Import Python System Libraries
import sys
import time

# Import relevant Python Libraries
import matplotlib.pyplot as plt
import numpy as np

# Defining Helper Functions
def cout(text):
	t_cout = sys.stdout
	sys.stdout = open('pset1_output.txt', 'a')
	print(text)
	sys.stdout = t_cout
	print(text)
	return True

# Defining Estimation Loops
def RecursiveLoop(M_V0):
	# Constructing Utility Matrix
	tM_Vc1 = Production(M_Ki, v_parameter[0], v_parameter[1])
	tM_Vc2 = (1 - v_parameter[3]) * M_Ki
	tM_Vc3 = Utility( tM_Vc1 + tM_Vc2 - M_Kj )

	# Constructing Value Matrix
	tM_V = tM_Vc3 + v_parameter[2] * M_V0

	# Constructing Value Vectors (argmax)
	v_V0 = np.reshape( np.nanmax(M_V0, axis=1), (i_n, 1) )
	tv_V = np.reshape( np.nanmax(tM_V, axis=1), (i_n, 1) )

	M_Values = np.zeros( (1, i_n) )

	# Diagnose Value Matrix
	#cout( np.transpose(v_V0) )
	#cout( np.transpose(tv_V) )
	cout("")

	# Test for Convergence
	cout( "Convergence Test: " + str(np.linalg.norm(v_V0-tv_V)) + " < " + str(v_parameter[4]) )

	if np.linalg.norm(v_V0-tv_V) > v_parameter[4]:
		# Enter Recursion
		cout("Convergence Failed ...")
		cout("Entering Recursion ...")
		M_Values = RecursiveLoop(tM_V)
	else:
		cout("Convergence Successful ...")

	cout("Collapsing Recursion ...")

	# Append Result to Matrix of Value Iterations
	return np.append(M_Values, np.transpose(tv_V), axis=0)

# Defining Utility Function
def Utility(x):
	tM_U = np.log(x)
	tM_U[x <= 0] = -1
	return tM_U

# Defining Production Function
def Production(x, coefficient, power):
    return coefficient * np.power(x, power * np.ones(x.shape))

# Discretisation
i_n = 1000
# Capital is discretised into finite bins

# Declaring Capital Vectors
v_Ki = np.arange(0, i_n, 1)
v_Ki = np.reshape(v_Ki, (i_n, 1))
v_K = v_Ki
v_Ki[0] = 12 / i_n
v_KP = np.arange(0, 12, 12 / i_n)
v_Kj = np.transpose(v_Ki)

# Capital Matrices
M_Ki = np.repeat( v_Ki, i_n, axis=1 )
M_Kj = np.repeat( v_Kj, i_n, axis=0 )
cout(M_Ki)
cout(M_Kj)

# Declaring Initial Guess for Values
M_InitialValues = np.zeros( (i_n, i_n) )

# Declaring Container for Iterated Values
M_IteratedValues = np.zeros( (1, i_n) )

cout("######")

# Set-up Output File Header [pset1_output.txt]
cout("[OUTPUT] Problem Set 1: Question 1 - S M Sajid Al Sanai")
t_time = time.asctime( time.localtime(time.time()) )
cout(t_time)
cout("")

# Question 1, Part A
cout("Question 1. (a)")
cout("i. A=20, alpha=0.3, beta=0.6, delta=0.5, epsilon=0.01")
cout("")

# Declaring Parameters
v_parameter = np.zeros( (5,) )
v_parameter[0] = 20		# A
v_parameter[1] = 0.3	# alpha
v_parameter[2] = 0.6	# beta
v_parameter[3] = 0.5	# delta
v_parameter[4] = 0.01	# epsilon

# Call Recursive Loop
M_IteratedValues = RecursiveLoop(M_InitialValues)
M_IteratedValues = np.delete(M_IteratedValues, 0, axis=0)
cout("")

# Display Matrix of Iterated Values
cout(M_IteratedValues)
cout("")

# Generate Plots
cout("Generating Plots ...")
cout("")
plt.plot(v_KP, M_IteratedValues[M_IteratedValues.shape[0]-1, :])
plt.title("Value Function Iteration")
plt.xlabel("Capital (K)")
plt.ylabel("Value")
plt.show()

# Question 1, Part B
cout("Question 1. (b)")
cout("i. Value Function Iteration")
cout("A=1, [alpha=0.3], [beta=0.6], delta=1, epsilon=0.01")
cout("")

# Declaring Parameters
v_parameter[0] = 1		# A
v_parameter[1] = 0.3	# alpha
v_parameter[2] = 0.6	# beta
v_parameter[3] = 1		# delta
v_parameter[4] = 0.01	# epsilon

# Call Recursive Loop
M_IteratedValues = RecursiveLoop(M_InitialValues)
M_IteratedValues = np.delete(M_IteratedValues, 0, axis=0)
cout("")

# Display Matrix of Iterated Values
cout(M_IteratedValues)
cout("")

# Generate Plots
cout("Generating Plots ...")
cout("")
plt.plot(v_KP, M_IteratedValues[M_IteratedValues.shape[0]-1, :])
plt.title("Value Function Iteration")
plt.xlabel("Capital (K)")
plt.ylabel("Value")
plt.show()

cout("ii. Analytical Form")
cout("A=1, [alpha=0.3], [beta=0.6], delta=1, epsilon=0.01")
cout("")

f_ab = v_parameter[1] * v_parameter[2]
f_B  = v_parameter[1] / (1 - f_ab)
f_A  = v_parameter[2] * f_B * np.log(f_ab)
v_VA = f_A * np.ones( (i_n, 1) ) + f_B * np.log(v_Ki)

#cout(v_VA)
#cout("")

# Generate Plots
cout("Generating Plots ...")
cout("")
plt.plot(v_KP, M_IteratedValues[M_IteratedValues.shape[0]-1, :])
plt.plot(v_KP, v_VA)
plt.plot(v_KP, M_IteratedValues[M_IteratedValues.shape[0]-1, :], v_KP, v_VA)
plt.title("Comparison of Value Function Iteration with Analytical Form")
plt.xlabel("Capital (K)")
plt.ylabel("Value")
plt.show()