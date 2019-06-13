#!/usr/bin/python3

# Import Python System Libraries
import sys
import time

# Import relevant Python Libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.linalg import lu, lu_factor, lu_solve
from scipy import optimize

# Defining Helper Functions
def cout(text):
	t_cout = sys.stdout
	sys.stdout = open('pset1_output.txt', 'a')
	print(text)
	sys.stdout = t_cout
	print(text)
	return True

# Defining Estimation Loops
def RecursiveLoop_ChoiceSpecific(M_V0, parameter):
	# Constructing Utility Vector
	tM_V = np.append( Utility(0, v_x, parameter), Utility(1, v_x, parameter), axis=1 )

	# Constructing Temporary Value Vector
	tc1 = v_parameter[6] * np.ones( (i_n, i_d) )
	tc2 = np.log(np.exp(M_V0[:, 0]) + np.exp(M_V0[:, 1]))
	tc2 = np.repeat(np.reshape(tc2, (i_n, 1)), 2, axis=1)
	tc2[:, 0] = np.matmul(M_TransitionProbability0, tc2[:, 0])
	tc2[:, 1] = np.matmul(M_TransitionProbability1, tc2[:, 1])
	tc2 *= parameter[1]
	tM_V += tc1 + tc2

	# Constructing Iteration Value Vector
	M_Values = tM_V
	#cout("")

	# Test for Convergence
	#cout( "Convergence Test: " + str(np.linalg.norm(M_V0.flatten()-tM_V.flatten())) + " < " + str(parameter[2]) )

	if np.linalg.norm(M_V0.flatten()-tM_V.flatten()) > parameter[2]:
		# Enter Recursion
		cout("Convergence Failed ...")
		#cout("Entering Recursion ...")
		M_Values = RecursiveLoop_ChoiceSpecific(tM_V, parameter)
	else:
		cout("Convergence Successful ...")

	#cout("Collapsing Recursion ...")
	return M_Values

def RecursiveLoop_Integrated(v_PR0, v_V0, parameter):
	# Declare Probability of Replacement across Realisable State Variable
	tv_ProbabilityReplacement = v_PR0

	# Define Errors by Replacement Decision
	tM_Errors = np.zeros( (i_n, i_d) )
	tM_Errors[:, 0] = np.reshape(parameter[6] * np.ones( (i_n, 1) ) - np.log(np.ones( (i_n,1) ) - tv_ProbabilityReplacement), (i_n,))
	tM_Errors[:, 1] = np.reshape(parameter[6] * np.ones( (i_n, 1) ) - np.log(tv_ProbabilityReplacement), (i_n,))

	# Constructing Temporary Value Vector
	tuc0 = Utility(0, v_x, parameter) + tM_Errors[:, 0]
	tuc1 = Utility(1, v_x, parameter) + tM_Errors[:, 1]
	tv_R = np.multiply(np.ones( (i_n, 1) ) - tv_ProbabilityReplacement, tuc0) + np.multiply(tv_ProbabilityReplacement, tuc1)
	tv_R = np.reshape(np.diag(tv_R), (i_n, 1))
	tM_G = np.multiply(tv_ProbabilityReplacement, M_TransitionProbability1) + np.multiply((np.ones( (i_n, 1) ) - tv_ProbabilityReplacement), M_TransitionProbability0)

	# Calculate Iterated Values
	# (Do not use Inverse or you will not graduate)
	#tM_V = np.linalg.inv(np.identity(i_n) - parameter[1] * tM_G)
	#tM_V = np.matmul(tM_V, tv_R)
	tM_V = lu_solve(lu_factor(np.identity(i_n) - parameter[1] * tM_G), tv_R)

	# Update Probabilities and Errors
	tv_ProbabilityReplacement = ConditionalChoice(Utility(0, v_x, parameter) + parameter[1] * np.matmul(M_TransitionProbability0, tM_V), Utility(1, v_x, parameter) + parameter[1] * np.matmul(M_TransitionProbability1, tM_V), 1)

	# Constructing Iteration Value Vector
	M_Values = tM_V
	#cout("")

	#cout( "Convergence Test: " + str(np.linalg.norm(v_V0.flatten()-tM_V.flatten())) + " < " + str(parameter[2]) )

	if np.linalg.norm(v_V0.flatten()-tM_V.flatten()) > parameter[2]:
		# Enter Recursion
		cout("Convergence Failed ...")
		#cout("Entering Recursion ...")
		M_Values, tv_ProbabilityReplacement = RecursiveLoop_Integrated(tv_ProbabilityReplacement, tM_V, parameter)
	else:
		cout("Convergence Successful ...")

	#cout("Collapsing Recursion ...")
	return M_Values, tv_ProbabilityReplacement

def ForwardSimulation_ChoiceSpecific(M_PR0, M_Errors, binomial_seed):
	np.random.seed(binomial_seed)
	tv_PathDecision = np.zeros( (i_t, 1) )
	tv_PathPolicy   = np.zeros( (i_t, 1) )
	for t in range(i_t):
		tx = tv_PathPolicy[t].astype(int)
		t_EstimatedDifference = np.log(M_PR0[tx, 1]) - np.log(M_PR0[tx, 0])
		if t_EstimatedDifference > M_Errors[t, 1] - M_Errors[t, 0]:
			tv_PathDecision[t] = 1
			if t < i_t - 1:
				tv_PathPolicy[t+1] = 0
		else:
			tv_PathDecision[t] = 0
			if t < i_t - 1:
				tv_PathPolicy[t+1] = min(tv_PathPolicy[t] + np.random.binomial(1, v_parameter[0]), 10)
	return tv_PathDecision, tv_PathPolicy

# Defining Conditional Choice Probability Function
def ConditionalChoice(Value0, Value1, i):
	t_denominator = np.exp(Value0) + np.exp(Value1)
	t_numerator   = (1 - i) * np.exp(Value0) + i * np.exp(Value1)
	return t_numerator / t_denominator

# Defining Utility Function
def Utility(i, x, parameter):
	uc1 = -Cost(x, parameter)
	uc2 = -parameter[5] * np.ones( (x.shape[0], 1) )
	uc1 = np.reshape( uc1, (x.shape[0], 1) )
	uc2 = np.reshape( uc2, (x.shape[0], 1) )
	return (1 - i) * uc1 + i * uc2

# Defining Cost Function
def Cost(x, parameter):
	return parameter[3] * x + parameter[4] * np.power(x, 2)

# Defining Transition Probability Matrix Generation Function
def TransitionProbability(parameter, replacement):
	p = np.array( (1 - v_parameter[0], v_parameter[0], 0, 0, 0, 0, 0, 0, 0, 0, 0) )
	P = np.reshape(np.tile( p, (1, i_n) ), (i_n, i_n))
	if replacement != 1:
		for i in range(i_n):
			P[i, :] = np.roll(P[i, :], i)
		P[i_n-1, i_n-1] = 1
		P[i_n-1, 0] = 0
	return P

# Discretisation
i_n = 11
i_d = 2
i_t = 5000

# Declaring State Vector
v_x = np.reshape(np.arange(0, i_n, 1), (i_n, 1))

# Declaring Time Vector
v_t = np.reshape(np.arange(0, i_t, 1), (i_t, 1))

'''
'''

cout("######")

# Set-up Output File Header [pset1_output.txt]
cout("[OUTPUT] Problem Set 1: Question 2 - S M Sajid Al Sanai")
t_time = time.asctime( time.localtime(time.time()) )
cout(t_time)
cout("")

# Question 2, Part B
cout("Question 2. (b) i. Choice Specific Value Function")
cout("")

# Declaring Parameters
v_parameter = np.zeros( (7,) )
v_parameter[0] = 0.8	# lambda
v_parameter[1] = 0.95	# beta
v_parameter[2] = 0.001	# epsilon
v_parameter[3] = 0.3	# theta1
v_parameter[4] = 0.0	# theta2
v_parameter[5] = 4.0	# theta3 R replacement cost
v_parameter[6] = 0.5772	# euler constant

# Declaring Initial Guess for Values
M_InitialValues = np.zeros( (i_n, i_d) )

# Declaring Container for Iterated Values
M_IteratedValues = np.zeros( (i_n, i_d) )

# Declaring Transition Probabilities
M_TransitionProbability0 = TransitionProbability(v_parameter, 0)
M_TransitionProbability1 = TransitionProbability(v_parameter, 1)

# Declaring Conditional Choice Probabilities
M_CCProbability = np.zeros( (i_n, i_d) )

# Call Recursive Loop
M_IteratedValues = RecursiveLoop_ChoiceSpecific(M_InitialValues, v_parameter)
cout("")

# Display Vector of Iterated Values
cout(M_IteratedValues)
cout("")

# Generate Plots
cout("Generating Plots ...")
cout("")
plt.plot(v_x, M_IteratedValues[:, 0], v_x, M_IteratedValues[:, 1])
plt.title("Choice Specific Value Function Iteration")
plt.xlabel("Mileage Realisations")
plt.ylabel("Value")
plt.show()

tM_IteratedValues = M_IteratedValues

# Generate Conditional Choice Probabilities
cout("Conditional Choice Probabilities:")
M_CCProbability[:, 0] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 0)
M_CCProbability[:, 1] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 1)
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, M_CCProbability, axis=1))
cout("")

'''
'''

cout("Question 2. (b) ii. Integrated Value Function")
cout("")

# Declaring Initial Guess for Probabilities
v_InitialProbabilities = 0.1 * np.ones( (i_n, 1) )

# Declaring Initial Guess for Values
M_InitialValues = np.zeros( (i_n, 1) )

# Call Recursive Loop
M_IteratedValues, M_CCProbability = RecursiveLoop_Integrated(v_InitialProbabilities, M_InitialValues, v_parameter)
cout("")

# Display Vector of Iterated Values
cout(M_IteratedValues)
cout("")

# Generate Plots
cout("Generating Plots ...")
cout("")
plt.plot(v_x, M_IteratedValues, v_x, Utility(1, v_x, v_parameter) + v_parameter[1] * np.matmul(M_TransitionProbability1, M_IteratedValues))
plt.title("Integrated Value Function Iteration")
plt.xlabel("Mileage Realisations")
plt.ylabel("Value")
plt.show()

# Generate Conditional Choice Probabilities
cout("Conditional Choice Probabilities:")
M_CCProbability = np.append(np.ones( (i_n, 1) ) - M_CCProbability, M_CCProbability, axis=1)
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, M_CCProbability, axis=1))
cout("")

# Generate Plot Comparison
cout("Generating Plots ...")
cout("")
plt.plot(v_x, M_IteratedValues, v_x, tM_IteratedValues[:, 0])
plt.title("Comparison of Value Function Iterations")
plt.xlabel("Mileage Realisations")
plt.ylabel("Value")
plt.show()

'''
'''

# Question 2, Part C
cout("Question 2. (c) Forward Simulation")
cout("")

# Load Uniformly Distributed Errors as Type I Extreme Value
M_TypeIErrors = np.loadtxt('./draw.out')
M_TypeIErrors = np.log( -np.log( M_TypeIErrors ) )

# Declaring Initial Guess for Values
M_InitialValues = np.zeros( (i_n, i_d) )

# Declaring Container for Iterated Values
M_IteratedValues = np.zeros( (i_n, i_d) )

# Declaring Transition Probabilities
M_TransitionProbability0 = TransitionProbability(v_parameter, 0)
M_TransitionProbability1 = TransitionProbability(v_parameter, 1)

# Declaring Conditional Choice Probabilities
M_CCProbability = np.zeros( (i_n, i_d) )

# Call Recursive Loop
M_IteratedValues = RecursiveLoop_ChoiceSpecific(M_InitialValues, v_parameter)
cout("")

# Generate Conditional Choice Probabilities
M_CCProbability[:, 0] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 0)
M_CCProbability[:, 1] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 1)

# Generate Policy and Decision Paths
v_PathDecision, v_PathPolicy = ForwardSimulation_ChoiceSpecific(M_CCProbability, M_TypeIErrors, 399169)

v_Decision = np.zeros( (i_n, 1) )
for t in range(i_t):
	if v_PathDecision[t] == 1:
		v_Decision[int(v_PathPolicy[t])] += 1

# Generate Plots
cout("Generating Plots ...")
cout("")

plt.bar(["Do not replace","Replace"], [i_t - np.sum(v_PathDecision), np.sum(v_PathDecision)])
plt.title("Simulated Frequencies of Replacement Decisions")
plt.xlabel("Replacement Decision")
plt.ylabel("Frequency")
plt.show()

plt.plot(v_Decision)
plt.title("Simulated Mileage Before Replacement")
plt.xlabel("Mileage")
plt.ylabel("Frequency of Replacements")
plt.show()

# Diagnose Simulated Path
#cout(np.append(v_PathPolicy, v_PathDecision, axis=1))
#cout("")

'''
'''

# Question 2, Part D
cout("Question 2. (d) Maximum Likelihood Estimation")
cout("")

def RecursiveLoop_Inner(M_V0, parameter):
		# Constructing Utility Vector
	tM_V = np.append( Utility(0, v_x, parameter), Utility(1, v_x, parameter), axis=1 )

	# Constructing Temporary Value Vector
	tc1 = v_parameter[6] * np.ones( (i_n, i_d) )
	tc2 = np.log(np.exp(M_V0[:, 0]) + np.exp(M_V0[:, 1]))
	tc2 = np.repeat(np.reshape(tc2, (i_n, 1)), 2, axis=1)
	tc2[:, 0] = np.matmul(M_TransitionProbability0, tc2[:, 0])
	tc2[:, 1] = np.matmul(M_TransitionProbability1, tc2[:, 1])
	tc2 *= parameter[1]
	tM_V += tc1 + tc2

	# Constructing Iteration Value Vector
	M_Values = tM_V
	#cout("")

	# Test for Convergence
	#cout( "Convergence Test: " + str(np.linalg.norm(M_V0.flatten()-tM_V.flatten())) + " < " + str(parameter[2]) )

	if np.linalg.norm(M_V0.flatten()-tM_V.flatten()) > parameter[2]:
		# Enter Recursion
		#cout("Convergence Failed ...")
		#cout("Entering Recursion ...")
		M_Values = RecursiveLoop_Inner(tM_V, parameter)
	else:
		cout("Convergence Successful ...")

	#cout("Collapsing Recursion ...")
	return M_Values

def RecursiveLoop_OuterNFPA(theta, p_lambda, p_beta, p_epsilon, p_euler_constant, decision, policy):
	# Obtain Expected Values
	parameter = np.zeros((7, 1))
	parameter[0] = p_lambda
	parameter[1] = p_beta
	parameter[2] = p_epsilon
	parameter[3] = theta[0]
	parameter[4] = theta[1]
	parameter[5] = theta[2]
	parameter[6] = p_euler_constant
	M_EV = RecursiveLoop_Inner(np.zeros((i_n, i_d)), parameter)

	# Declare Log Likelihood Loop
	f_LogLikelihood = 0

	# Call Log Likelihood Loop
	for t in range(1, i_t):
		# Determine Mileage Transition Probability
		i_delta_mileage = policy[t] - policy[t-1]
		if i_delta_mileage == 1:
			f_probability_mileage = p_lambda
		elif i_delta_mileage == 0 and decision[t-1] == 0:
			f_probability_mileage = 1 - p_lambda
		elif policy[t] == 0:
			f_probability_mileage = 1 - p_lambda
		elif policy[t] == 1 and decision[t-1] == 1:
			f_probability_mileage = p_lambda

		# Determine Replacement Probability
		f_probability_replacement = ProbabilityReplacement(M_EV, parameter)
		t_probability_replacement =f_probability_replacement[0, int(policy[t])]
		if decision[t] == 1:
			f_probability_replacement = t_probability_replacement
		else:
			f_probability_replacement = 1 - t_probability_replacement

		if f_probability_replacement <= 0:
			f_probability_mileage = 0.0001

		# Sum over Log Likelihoods
		f_LogLikelihood += np.log(f_probability_replacement) + np.log(f_probability_mileage)

	# Return Sum of Log Likelihood
	cout("Generated Log Likelihood ... " + str(-f_LogLikelihood))
	return -f_LogLikelihood

def ProbabilityReplacement(M_EV, parameter):
	V0 = np.exp(Utility(0, v_x, parameter) + parameter[1] * M_EV[:, 0])
	V1 = np.exp(Utility(1, v_x, parameter) + parameter[1] * M_EV[:, 1])
	return V1 / (V0 + V1)

# Declaring Initial Guess for Parameters before MLE
v_InitialTheta = np.zeros( (3,) )
v_InitialTheta[0] = 0.3
v_InitialTheta[1] = 0.0
v_InitialTheta[2] = 4.0

cout("Initial Guess for Theta: " + str(v_InitialTheta))
cout("")

cout("Running Minimisation Routine:")
v_OutputTheta = sp.optimize.minimize(RecursiveLoop_OuterNFPA, x0=v_InitialTheta, args=(v_parameter[0], v_parameter[1], v_parameter[2], v_parameter[6], v_PathDecision, v_PathPolicy), method='BFGS')
v_MinimisedTheta = v_OutputTheta.x
cout("")

cout("Minimised Theta:")
cout(str(v_OutputTheta))
cout("")

# Simplex Nealder-Mead is preferable to BFGS which goes to negative values

'''
'''

# Question 2, Part E
cout("Question 2. (e) i. Forward Simulation with Minimised Parameters")
cout("")

# Declaring Container for Iterated Values
M_IteratedValues = np.zeros( (i_n, i_d) )

# Declaring Conditional Choice Probabilities
M_CCProbability0 = np.zeros( (i_n, i_d) )

# Call Recursive Loop
M_IteratedValues = RecursiveLoop_ChoiceSpecific(M_InitialValues, np.append(v_parameter[0:2], np.append(v_MinimisedTheta, v_parameter[6])))
cout("")

# Generate Conditional Choice Probabilities
cout("Conditional Choice Probabilities:")
M_CCProbability0[:, 0] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 0)
M_CCProbability0[:, 1] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 1)
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, M_CCProbability0, axis=1))
cout("")

# Generate Policy and Decision Paths
v_PathDecision, v_PathPolicy = ForwardSimulation_ChoiceSpecific(M_CCProbability0, M_TypeIErrors, 399169)

v_Decision = np.zeros( (i_n, 1) )
v_States   = np.zeros( (i_n, 1) )
for t in range(i_t):
	if v_PathDecision[t] == 1:
		v_Decision[int(v_PathPolicy[t])] += 1
	v_States[int(v_PathPolicy[t])] += 1

# Long Run Replacement Probabilities
cout("Long Run Replacement Probabilities:")
cout(v_Decision)
cout(v_States)
v_LRReplacementProbability0 = np.divide(v_Decision, v_States)
cout(v_LRReplacementProbability0)
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, np.append(np.ones((i_n, 1)) - v_LRReplacementProbability0, v_LRReplacementProbability0, axis=1), axis=1))
cout("")

# Generate Plots
cout("Generating Plots ...")
cout("")

plt.bar(["Do not replace","Replace"], [i_t - np.sum(v_PathDecision), np.sum(v_PathDecision)])
plt.title("Simulated Frequencies of Replacement Decisions")
plt.xlabel("Replacement Decision")
plt.ylabel("Frequency")
plt.show()

plt.plot(v_Decision)
plt.title("Simulated Mileage Before Replacement")
plt.xlabel("Mileage")
plt.ylabel("Frequency of Replacements")
plt.show()

'''
'''

cout("Question 2. (e) ii. Long Run Replacement Probabilities (SS)")

def Recursion(p_theta, p_beta, p_lambda): # ergodic distribution
	# P is a transition matrix
	# Rows must sum to one
	parameter = np.zeros((7, 1))
	parameter[0] = v_parameter[0]
	parameter[1] = v_parameter[1]
	parameter[2] = v_parameter[2]
	parameter[3] = p_theta[0]
	parameter[4] = p_theta[1]
	parameter[5] = p_theta[2]
	parameter[6] = v_parameter[6]

	tM_EV = RecursiveLoop_Inner(np.zeros((i_n, i_d)), parameter)
	#value_fn_engine(epsilon=1e-10, max_iter=100, beta=0.95, theta=theta)[0]
	prob_rep = ProbabilityReplacement(tM_EV, parameter)
	l = 11
	P = np.zeros((l, l))
	for i in range(l): # i is starting state, j is end state
		for j in range(l):
			if j == 0:
				P[i, j] += (1 - p_lambda) * prob_rep[0,i]
			if j == 1:
				P[i, j] += p_lambda * prob_rep[0,i]
			if (j - i)==1:
				P[i, j] += p_lambda * (1 - prob_rep[0,i])
			if j == i:
				P[i, j] += (1 - p_lambda) * (1 - prob_rep[0,i])
	P[10, 10] = 1 - P[10, 0] - P[10, 1]
	# P = np.matrix.round(P, 3)
	# print(P)
	# rsum = np.sum(P, axis=1)
	# print(rsum)

	Q = np.ones(l) / l
	tol = 1e-10; epsilon = 1; i = 0
	while tol < epsilon and i < 100:
		Q_new = Q.dot(P)
		diff = Q_new - Q
		epsilon = np.linalg.norm(diff)
		Q = Q_new
		i += 1
	return Q

Q = Recursion(p_theta=v_MinimisedTheta, p_beta=0.95, p_lambda=0.8)

cout("(Comparative) Long Run Replacement Probabilities:")
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, np.append(np.ones((i_n, 1)) - np.reshape(Q, (i_n, 1)), np.reshape(Q, (i_n, 1)), axis=1), axis=1))
cout("")

'''
'''

cout("Question 2. (e) iii. Counterfactual with Subsidy on Minimised Parameters")

# Declaring Container for Iterated Values
M_IteratedValues = np.zeros( (i_n, i_d) )

# Declaring Conditional Choice Probabilities
M_CCProbability1 = np.zeros( (i_n, i_d) )

# Call Recursive Loop
parameter = np.zeros((7, 1))
parameter[0] = v_parameter[0]
parameter[1] = v_parameter[1]
parameter[2] = v_parameter[2]
parameter[3] = v_MinimisedTheta[0]
parameter[4] = v_MinimisedTheta[1]
parameter[5] = v_MinimisedTheta[2] * 0.9
parameter[6] = v_parameter[6]
M_IteratedValues = RecursiveLoop_ChoiceSpecific(M_InitialValues, parameter)
cout("")

# Generate Conditional Choice Probabilities
M_CCProbability1[:, 0] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 0)
M_CCProbability1[:, 1] = ConditionalChoice(M_IteratedValues[:, 0], M_IteratedValues[:, 1], 1)

# Generate Policy and Decision Paths
v_PathDecision, v_PathPolicy = ForwardSimulation_ChoiceSpecific(M_CCProbability1, M_TypeIErrors, 399169)

v_Decision = np.zeros( (i_n, 1) )
v_States   = np.zeros( (i_n, 1) )
for t in range(i_t):
	if v_PathDecision[t] == 1:
		v_Decision[int(v_PathPolicy[t])] += 1
	v_States[int(v_PathPolicy[t])] += 1

# Long Run Replacement Probabilities
cout("(10% Subsidy) Long Run Replacement Probabilities:")
v_LRReplacementProbability1 = np.divide(v_Decision, v_States)
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, np.append(np.ones((i_n, 1)) - v_LRReplacementProbability1, v_LRReplacementProbability1, axis=1), axis=1))
cout("")

cout("(Differential) Long Run Replacement Probabilities:")
cout("[[ x; Pr(i=0|x,theta); Pr(i=1|x,theta); ]]")
cout(np.append(v_x, v_LRReplacementProbability1 - v_LRReplacementProbability0, axis=1))
cout("")