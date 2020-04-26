import numpy as np
from collections import deque
import depthfirst
from mazes import Maze

def main():
    m = 10
    n = 10

    intv = 0.01

    pValues=np.arange(0, 1+intv, intv)

    for p in pValues:
        print(p)

        M = np.zeros((m,n))
        s = np.random.uniform(0,1,m*n)
        
        

        suma = 0
        #print(s.size)
        suave = 0
        for i in range(0,m):
            for j in range(0,n):
                #print(s[index], p <= s[index])
                if(p > s[(j+n*i)]):
                    M[i][j] = 1
                else:
                    suave = suave + 1
                suma = suma + s[(j+n*i)]


        # 0 is soft soil
        # 1 is hard soil
        #print('tamaño de la matriz: ',M.shape)
        #print('media de s', (suma/(n*m)))
        #print('probabilidad de que un bloque de tierra sea dura: ' ,p)
        #print('probabilidad de que un bloque de tierra sea suave: ' ,1-p)
        #print('porcentaje de tierra suave en la matriz:',(suave/(n*m)))
        A = Maze(M)
        solution = depthfirst.solve(A)[1][2]
        #print(solution[1][2])



if __name__ == "__main__":
    main()