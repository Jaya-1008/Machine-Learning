import random
import statistics

# Function to count vowels and consonants in a string
def count_vowels_consonants(word):
    vowel_count=0
    consonant_count=0
    for c in word:
        if c in "aeiouAEIOU":
            vowel_count=vowel_count + 1
        elif c.isalpha():
            consonant_count+=1
    return vowel_count,consonant_count

# Function for matrix multiplication of 2 matrices
def matrix_multiplication(A,B,r1,c1,c2):
    mul=[]
    for i in range(r1):
        temp=[]
        for j in range(c2):
            m=0
            for k in range(c1):
                m=m+A[i][k]*B[k][j]
            temp.append(m)
        mul.append(temp)
    return mul

# Function to count common elements in 2 lists
def common_elements(l1,l2):
    com_elem_count=0
    for i in l1:
        for j in l2:
            if i==j:
                com_elem_count+=1
                break
    return com_elem_count

# Function to find transpose of a matrix
def transpose_matrix(mat,r,c):
    transpose=[]
    for i in range(c):
        row=[]
        for j in range(r):
            row.append(mat[j][i])
        transpose.append(row)
    return transpose

# Function to calculate mean, median and mode of list of 100 numbers generated randomly b/w 100 and 150
def random_statistics():
    l=[]
    for i in range(100):
        l.append(random.randint(100, 150))
    mean=statistics.mean(l)
    median=statistics.median(l)
    mode=statistics.mode(l)
    return l, mean, median, mode

#Q1
word=input("Enter a String:")
vowel_count,consonant_count=count_vowels_consonants(word)
print("Vowel count is:",vowel_count)
print("Consonant count is:",consonant_count)

#Q2
r1=int(input("Enter no.of rows in matrix A:"))
c1=int(input("Enter no.of columns in matrix A:"))
r2=int(input("Enter no.of rows in matrix B:"))
c2=int(input("Enter no.of columns in matrix B:"))
A=[]
B=[]
if c1==r2:
    print("Enter matrix A:")
    for i in range(r1):
        row=list(map(int, input().split()))
        A.append(row)
    print("Enter matrix B:")
    for i in range(r2):
        row=list(map(int, input().split()))
        B.append(row)
    mul=matrix_multiplication(A, B, r1, c1, c2)
    print(mul)
else:
    print("Multiplication can't be done as columns of A and rows of B is not equal")

#Q3
l1=list(map(int, input("Enter integers of l1:").split()))
l2=list(map(int, input("Enter integers of l2:").split()))
com_elem_count=common_elements(l1, l2)
print("No.of common elements in l1 and l2 is:", com_elem_count)

#Q4
r=int(input("Enter no.of rows in matrix:"))
c=int(input("Enter no.of columns in matrix:"))
mat=[]
print("Enter matrix:")
for i in range(r):
    row=list(map(int, input().split()))
    mat.append(row)
transpose=transpose_matrix(mat, r, c)
print(transpose)

#Q5
l,mean,median,mode = random_statistics()
print(l)
print("Mean is:", mean)
print("Median is:", median)
print("Mode is:", mode)