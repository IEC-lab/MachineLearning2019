import sys

#Assume y=kx+b,we need the present data to predict the k and b

#Training data set
#each element in x represents (x1)
x = [1,2,3,4,5,6]
#y[i] is the output of y = b+ k * x[1]
y = [13,14,20,21,25,30]
#set the error
epsilon = 1
#learning rate,you can change the rate to see different output
alpha = 0.01
diff = [0,0]
max_itor = 20
error1 = 0
error0 =0
cnt = 0
m = len(x)
#init the parameters to zero
k = 0
b = 0
while 1:
	cnt=cnt+1
	diff = [0,0]
	for i in range(m):
		diff[0]+=b+ k* x[i]-y[i]
		diff[1]+=(b+k*x[i]-y[i])*x[i]
	b=b-alpha/m*diff[0]
	k=k-alpha/m*diff[1]
	error1=0
	for i in range(m):
		error1+=(b+k*x[i]-y[i])**2
	if abs(error1-error0)< epsilon:
		break
	print('b :{0},k :{1},error:{2}'.format(b,k,error1))
	if cnt>20:
		print ('cnt>20')
		break
print('b :{0},k :{1},error:{2}'.format(b,k,error1))
