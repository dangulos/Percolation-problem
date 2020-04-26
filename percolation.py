import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from collections import deque
import depthfirst
from mazes import Maze

def percolation(m,n,intervals, attempts):
    m = int(m)
    n = int(n)
    intervals = int(intervals)
    attempts = int(attempts)
    deltaP = 1/intervals
    #Zalpha/2 para alpha = 99%
    ZAlphaHalf = 2.575

    pValues=np.arange(0, 1+deltaP, deltaP)

    column_names = ["p", "theta", "error"]
    df = pd.DataFrame(columns = column_names)

    for p in pValues:
        #print('Probability',p)
        results = 0
        for attempt in range(0,attempts):

            M = np.zeros((m,n))
            s = np.random.uniform(0,1,m*n)
            
            

            suma = 0
            #print(s.size)
            suave = 0
            for i in range(0,m):
                for j in range(0,n):
                    #print(s[index], p <= s[index])
                    if(p < s[(j+n*i)]):
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
            if(solution):
                results += 1
            #print(solution[1][2])
        #print('Result',results/200) 
        percentage = results/attempts
        error=ZAlphaHalf*np.sqrt((percentage*(1-percentage))/attempts)
        #print('theta(',p,') = ',percentage, '+-',error)
        dft = pd.DataFrame({"p":[p],"theta":[percentage],"error":[error]},columns = column_names)
        #print(dft)
        df = df.append(dft, ignore_index=True)

    df.insert(3, "maxTheta", df["theta"]+df["error"]) 
    df.insert(4, "minTheta", df["theta"]-df["error"]) 
    print(df)
    df.to_csv('data.csv')
    #df.plot()
    
    df.plot(x='p', y=['theta','maxTheta','minTheta'])


    """
    plt.table(cellText= df.to_numpy(),
              rowLabels=range(0,intervals+1),
              colLabels=column_names,
              loc='center')
    plt.subplots_adjust(left=0.2, top=0.8)
    """
    plt.show()

def main():#how to run: python percolation.py 10 10 [-i or --intervals] number of intervals [-a or --attempts] number of attempts
    parser = argparse.ArgumentParser()
    parser.add_argument("m")
    parser.add_argument("n")
    parser.add_argument("-i", "--intervals", nargs='?', const='intervals', default='10')
    parser.add_argument("-a", "--attempts", nargs='?', const='attempts', default='200')
    #print(parser)
    args = parser.parse_args()
    #print(args)

    percolation(args.m,args.n,args.intervals, args.attempts)

if __name__ == "__main__":
    main()