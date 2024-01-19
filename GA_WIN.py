import numpy as np
import matplotlib.pyplot as plt
import math
import os
import random
import pandas as pd
import shutil
from bayes_opt import BayesianOptimization

# ==== 自動"讀取"相關 ====
def iter_files(directory):                                                 # 找到資料夾路徑
    for filename in os.listdir(directory):
        yield os.path.join(directory, filename)

def readfile():                                                            # 讀取檔案用在GA_solver
    with open(file_name, 'r') as f:
        file_list = [list(map(lambda x: int(x) ,line.split())) for line in f]
    return file_list

# ==== 自動"輸出"相關 ====
def fitFunc_2(x):                                                          # 適應度函數
    S = np.zeros((NUM_JOB, NUM_MACHINE))    # S[i][j] = Starting time of job i at machine j
    C = np.zeros((NUM_JOB, NUM_MACHINE))    # C[i][j] = Completion time of job i at machine j    
    B = np.zeros(NUM_MACHINE, dtype=int)    # B[j] = Available time of machine j  
    opJob = np.zeros(NUM_JOB, dtype=int)    # opJob[i] = current operation ID of job i
    
    for i in range(NUM_BIT):
        m = mOrder[x[i]][opJob[x[i]]]
        if opJob[x[i]] != 0:
            S[x[i]][m] = max([B[m], C[x[i]][mOrder[x[i]][opJob[x[i]]-1]]])
        else:
            S[x[i]][m] = B[m]        
        C[x[i]][m] = B[m] = S[x[i]][m] + pTime[x[i]][opJob[x[i]]]
        opJob[x[i]] += 1
    return S, C, B, -max(B)

def gantt(s, t):                                                           # 輸出甘特圖
    fig, ax = plt.subplots(figsize=(10, 5)) #圖片大小
    ax.set_ylim(0, len(s)) #y大小
    
    s = np.transpose(s)  # 將 start 轉置
    t = np.transpose(t)  # 將 end 轉置
    
    for i in range(len(s)): #一條甘特圖
        #start_times = data[i][::2]
        #end_times = data[i][1::2]
        start_nums = s[i]
        end_nums = t[i]
        for j in range(len(start_nums)): #一段時間段
            start = start_nums[j]
            end = end_nums[j]
            ax.barh(i, end - start, left=start, height=0.9, align='center', alpha=0.8) #設定左右與長度
            # ax.annotate(str(start), xy=(start, i), xytext=(start, i-0.15), ha='center', va='top') #標注起始時間
            # ax.annotate(str(end), xy=(end, i), xytext=(end, i-0.15), ha='center', va='top') #標注結束時間

    ax.set_yticks(range(len(s))) #x軸範圍
    ax.set_yticklabels(['Machine{}'.format(i+1) for i in range(len(s))]) #job標注
    ax.set_xlabel('Time')
    plt.savefig(new_folder_path + "gantt.png")
    #plt.show()

def transform_to_order(x):                                                 # 產生用在flexsim上的排程solution
    L = []
    for i in range(NUM_JOB):
        L.append([])
    for i in range(len(x)):
        L[x[i]].append(i+1)
    return L

def mkdir():                                                               # 創建新資料夾
    current_path = os.getcwd()              # 獲取當前.py文件的路徑
    new_folder_name = file_lname + "_solution"                                      # 新資料夾的名稱
    new_folder_path = os.path.join(current_path, new_folder_name)          # 新資料夾的完整路徑
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path + '\\'

def time_xlsx(Start, Completion):                                          # 創建time excel檔
    Total = []
    L = []
    for i in range(len(Start)):
        Q = []
        L.append("Job" + str(i+1))
        for j in range(len(Start[i])):
            Q.append(Start[i][j])
            Q.append(Completion[i][j])
        Total.append(Q)

    Q = []
    for j in range(len(Start[0])):
        Q.append("StartTime_M" + str(j+1))
        Q.append("CompletionTime_M" + str(j+1))

    df = pd.DataFrame(Total)
    df.index = L  
    df.columns = Q  
    with pd.ExcelWriter(new_folder_path +'best_solution.xlsx', mode='a') as writer:
        df.to_excel(writer, sheet_name='Time')

def sequence_xlsx(Start):                                                  # 創建sequence excel檔
    TStart = Start.T
    Machine_order = []
    Seq = []
    Mac = []
    x = 0
    for i in range(len(TStart[0])):
        Seq.append("Seq" + str(i+1))
    for cards in TStart:
        x = x+1
        sorted_cards = [card[0] for card in sorted(enumerate(cards, 1), key=lambda x: x[1])]
        Machine_order.append(sorted_cards)
        Mac.append("Machine" + str(x))
    df = pd.DataFrame(Machine_order)
    df.index = Mac
    df.columns = Seq   
    with pd.ExcelWriter(new_folder_path +'best_solution.xlsx', mode='a') as writer:
        df.to_excel(writer, sheet_name='Sequence')

def route_xlsx():                                                          # 用在flexsim上的source輸出成excel檔
    data = []
    L = []
    b = []
    for j in range(len(pTime[0]) + 1):
        b.append("Operation" + str(j+1))
    for i in range(len(pTime)):
        order = str(i+1)
        Route = "Route_" + order
        SetupTime = "SetupTime_" + order
        PTime = "pTime_" + order
        L.append(Route)
        L.append(SetupTime)
        L.append(PTime)
        data.append([])
        data.append([])
        data.append([])
    for i in range(len(pTime)):
        for j in range(len(pTime[i])):
            data[3*i].append('Buffer_' + str(mOrder[i][j]+1))
            data[3*i + 1].append(0)
            data[3*i + 2].append(pTime[i][j])
    for i in range(len(pTime)):
        data[3*i].append('Sink')
        data[3*i + 1].append(0)
        data[3*i + 2].append(0)
    df = pd.DataFrame(data, index = L, columns = b).T
    output_name = new_folder_path + file_lname + "_route.xlsx"
    df.to_excel(output_name)

def order_xlsx(x, y):                                                      # 用在flexsim上的排程輸出成excel檔
    data = transform_to_order(x)                                           # 假設你的二維列表是這樣：
    columns = ['Operation' + str(i + 1) for i in range(NUM_MACHINE)]                # 創建列名稱和行名稱
    index = ['Job' + str(i + 1) for i in range(NUM_JOB)]                        # 將二維列表轉換為 pandas DataFrame，並指定列名稱和行名稱
    df = pd.DataFrame(data, columns=columns, index=index)
    output_name = new_folder_path  + file_lname  + "_solution_" + str(GA_ITERATION) + "_result_" + str(-y) + ".xlsx"
    df1 = source(x)
    df2 = pd.DataFrame({'group_num': [NUM_CHROME], 'crossover_rate': [Pc], 'mutation_rate': [Pm], 'result': [-y]})
    with pd.ExcelWriter(output_name) as writer:
        df.to_excel(writer, sheet_name='Rank')                                         # 寫入 Excel 文件
        df1.to_excel(writer, sheet_name='Source')   
        df2.to_excel(writer, sheet_name='Parameter')   

def source(x):                                                             # 用在flexsim上的source輸出成excel檔
    job_set = set()
    Arrival_time = []
    ItemName = []
    ItemType = []
    Quantity = []
    row_names = []   # 創建一個空列表來存放行名稱
    id = 0
    for i in range(len(x)):
        if x[i] not in job_set:
            job_set.add(x[i])
            Arrival_time.append(0)
            ItemName.append('Product_' + str(x[i] + 1))
            ItemType.append(x[i] + 1)
            Quantity.append(1)
            id = id + 1
            row_names.append('Arrival' + str(id))   # 添加行名稱到列表中
            
    data = {
        'ArrivalTime': Arrival_time, 
        'ItemName': ItemName, 
        'ItemType': ItemType, 
        'Quantity': Quantity
    }

    df = pd.DataFrame(data, index=row_names)   # 將行名稱傳遞給 `pd.DataFrame`
    return df

def bestsolution_copy(besttime):                                           # 複製最佳解
    choosen_one = str(-1 * int(besttime))
    for solution_name in iter_files(new_folder_path):
        name = solution_name.split(".")[0]  # 拆分檔名，並取得 `.` 前面的部分
        first_four_chars = name[-4:]  # 取得前四個字元
        if choosen_one == first_four_chars:
            shutil.copy(solution_name, new_folder_path + "best_solution.xlsx")
            break

# ==== GA相關 ====
def initPop():                                                             # 初始化群體
    p = []
    for i in range(NUM_CHROME) :        
        a = []
        for j in range(NUM_JOB):
            for k in range(NUM_MACHINE):
                a.append(j)
        np.random.shuffle(a)
        p.append(a)
    return p

def fitFunc(x):                                                            # 適應度函數
    S = np.zeros((NUM_JOB, NUM_MACHINE))    # S[i][j] = Starting time of job i at machine j
    C = np.zeros((NUM_JOB, NUM_MACHINE))    # C[i][j] = Completion time of job i at machine j    
    B = np.zeros(NUM_MACHINE, dtype=int)    # B[j] = Available time of machine j  
    opJob = np.zeros(NUM_JOB, dtype=int)    # opJob[i] = current operation ID of job i
    
    for i in range(NUM_BIT):
        m = mOrder[x[i]][opJob[x[i]]]
        if opJob[x[i]] != 0:
            S[x[i]][m] = max([B[m], C[x[i]][mOrder[x[i]][opJob[x[i]]-1]]])
        else:
            S[x[i]][m] = B[m]        
        C[x[i]][m] = B[m] = S[x[i]][m] + pTime[x[i]][opJob[x[i]]]
        opJob[x[i]] += 1
    return -max(B)           # 因為是最小化問題

def evaluatePop(p):                                                        # 評估群體之適應度
    return [fitFunc(p[i]) for i in range(len(p))]

def selection(p, p_fit):                                                   # 用二元競爭式選擇法來挑父母
    a = []
    for i in range(NUM_PARENT):
        [j, k] = np.random.choice(NUM_CHROME, 2, replace=False)  # 任選兩個index
        if p_fit[j] > p_fit[k] :                      # 擇優
            a.append(p[j].copy())
        else:
            a.append(p[k].copy())
    return a

def crossover_one_point(p):                                                # 用單點交配來繁衍子代 (new)
    a = []

    for i in range(NUM_CROSSOVER) :
        c = np.random.randint(1, NUM_BIT)      		  # 隨機找出單點(不包含0)
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index

        child1, child2 = p[j].copy(), p[k].copy()
        remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市

        for m in range(NUM_BIT):
            if m < c :
                remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
                remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]

        t = 0
        for m in range(NUM_BIT):
            if m >= c :
                child1[m] = remain2[t]
                child2[m] = remain1[t]
                t += 1

        a.append(child1)
        a.append(child2)
    return a

def crossover_uniform(p):                                                  # 用均勻交配來繁衍子代 (new)
    a = []
    for i in range(NUM_CROSSOVER) :
        mask = np.random.randint(2, size=NUM_BIT)
        [j, k] = np.random.choice(NUM_PARENT, 2, replace=False)  # 任選兩個index
        child1, child2 = p[j].copy(), p[k].copy()
        remain1, remain2 = list(p[j].copy()), list(p[k].copy())     # 存還沒被用掉的城市
        for m in range(NUM_BIT):
            if mask[m] == 1 :
                remain2.remove(child1[m])   # 砍掉 remain2 中的值是 child1[m]
                remain1.remove(child2[m])   # 砍掉 remain1 中的值是 child2[m]
        t = 0
        for m in range(NUM_BIT):
            if mask[m] == 0 :
                child1[m] = remain2[t]
                child2[m] = remain1[t]
                t += 1
        a.append(child1)
        a.append(child2)
    return a

def mutation(p):                                                           # 突變
    for _ in range(NUM_MUTATION) :
        if NUM_CROSSOVER_2 == 0: 
            break
        row = np.random.randint(NUM_CROSSOVER_2)    # 任選一個染色體
        [j, k] = np.random.choice(NUM_BIT, 2, replace=False)  # 任選兩個基因
        p[row][j], p[row][k] = p[row][k], p[row][j]       # 此染色體的兩基因互換

def sortChrome(a, a_fit):	                                               # a的根據a_fit由大排到小
    a_index = range(len(a))                         # 產生 0, 1, 2, ..., |a|-1 的 list    
    a_fit, a_index = zip(*sorted(zip(a_fit,a_index), reverse=True)) # a_index 根據 a_fit 的大小由大到小連動的排序
    return [a[i] for i in a_index], a_fit           # 根據 a_index 的次序來回傳 a，並把對應的 fit 回傳

def replace(p, p_fit, a, a_fit):                                           # 適者生存
    b = np.concatenate((p,a), axis=0)               # 把本代 p 和子代 a 合併成 b
    b_fit = p_fit + a_fit                           # 把上述兩代的 fitness 合併成 b_fit
    b, b_fit = sortChrome(b, b_fit)                 # b 和 b_fit 連動的排序  
    return b[:NUM_CHROME], list(b_fit[:NUM_CHROME]) # 回傳 NUM_CHROME 個為新的一個世代

def GA_solver(group_num, crossover_rate, mutation_rate):                   # GA演算法
    global NUM_CHROME, Pc, Pm, GA_ITERATION, NUM_PARENT, NUM_CROSSOVER, NUM_CROSSOVER_2, NUM_MUTATION, NUM_ITERATION, best_x, best_y
    GA_ITERATION = GA_ITERATION + 1
    NUM_CHROME = int(group_num)
    Pc = crossover_rate
    Pm = mutation_rate
    NUM_PARENT = NUM_CHROME
    NUM_CROSSOVER = int(Pc * NUM_CHROME / 2)          # 交配的次數
    NUM_CROSSOVER_2 = NUM_CROSSOVER * 2               # 上數的兩倍
    NUM_MUTATION = int(Pm * NUM_CHROME * NUM_BIT)     # 突變的次數# === Step 3-2. NUM_BIT 要修改成 3 x 3 ===
    #print( Pc, Pm, GA_ITERATION, NUM_PARENT, NUM_CROSSOVER, NUM_CROSSOVER_2, NUM_MUTATION, NUM_ITERATION)
    # ==== 主程式 ==== 
    pop = initPop()                                 # 初始化 pop
    pop_fit = evaluatePop(pop)                      # 算 pop 的 fit
    best_outputs = []                               # 用此變數來紀錄每一個迴圈的最佳解 (new)
    best_outputs.append(np.max(pop_fit))            # 存下初始群體的最佳解
    mean_outputs = []                               # 用此變數來紀錄每一個迴圈的平均解 (new)
    mean_outputs.append(np.average(pop_fit))        # 存下初始群體的最佳解

    for i in range(NUM_ITERATION) :
        parent = selection(pop, pop_fit)            # 挑父母
        offspring = crossover_one_point(parent)     # 單點交配
        mutation(offspring)                         # 突變
        offspring_fit = evaluatePop(offspring)      # 算子代的 fit
        if NUM_CROSSOVER_2 != 0:
            pop, pop_fit = replace(pop, pop_fit, offspring, offspring_fit)    # 取代
        best_outputs.append(np.max(pop_fit))        # 存下這次的最佳解
        mean_outputs.append(np.average(pop_fit))    # 存下這次的平均解
       #print('iteration %d: y = %d'	%(i, -pop_fit[0]))     # fit 改負的
       #print('iteration %d: x = %s, y = %d'	%(i, pop[0], -pop_fit[0]))     # fit 改負的

    #return best_outputs, mean_outputs, pop[0], -pop_fit[0]
    order_xlsx(pop[0], pop_fit[0])

    if abs(pop_fit[0]) < abs(best_y[0]):
        best_y = pop_fit
        best_x = pop 
    return pop_fit[0]

problem_path = input("請輸入存放問題的資料夾路徑：")                            # 輸入存放問題的資料夾路徑

for file_name in iter_files(problem_path):                                 # 將資料夾中的所有檔案名稱一個一個解出

    # ==== 存放資料處、以及讀檔 ====
    best_y = [100000, 1000000]
    best_x = []
    file_list = readfile() 
    last_part = file_name.rsplit('\\', 1)[-1]           # 這裡先取得 '\\' 之後的所有字符
    file_lname = last_part.split('.')[0]               # 然後取得 '.' 之前的所有字符 

    # ==== JSSP自動讀檔和問題設定，下方為機器排程與所需時間 ====
    pTime = [[file_list[i][j] for j in range(len(file_list[i])) if j % 2 == 1] for i in range(1, len(file_list))]
    mOrder = [[file_list[i][j] for j in range(len(file_list[i])) if j % 2 == 0] for i in range(1, len(file_list))]

    NUM_JOB = file_list[0][0]                         # 工件數量       
    NUM_MACHINE = file_list[0][1]                     # 機器數量
    NUM_BIT = NUM_JOB * NUM_MACHINE                   # 染色體長度 

    # ==== 手動參數設定(與演算法相關) ====
    np.random.seed(0)                                 # 若要每次跑得都不一樣的結果，就把這行註解掉
    NUM_ITERATION = 3000                              # 終止世代數(迴圈數)

    # ==== 貝式參數設定(與演算法相關) ====
    GA_ITERATION = 0                                  # 紀錄GA演算法跑了幾次，下面設定都先設定為0，貝式會自動優化超參數
    NUM_CHROME = 0                                    # 染色體個數
    Pc = 0.0                                          # 交配率 (代表共執行Pc*NUM_CHROME/2次交配)
    Pm = 0.0                                          # 突變率 (代表共要執行Pm*NUM_CHROME*NUM_BIT次突變)
    NUM_PARENT = 0                                    # 父母的個數
    NUM_CROSSOVER = 0                                 # 交配的次數
    NUM_CROSSOVER_2 = 0                               # 上數的兩倍
    NUM_MUTATION = 0                                  # 突變的次數 
    
    # ==== 開始運行 ====
    new_folder_path = mkdir()                         # 創建新資料夾來存放結果
    route_xlsx()                                      # 創建route excel檔
    optimizer = BayesianOptimization(                 # 貝式超參數優化
        f = GA_solver,
        pbounds = {
            'group_num': (4, 300),                                 #約200最好
            'crossover_rate': (0.48, 0.90),             #0.4~0.9最好
            'mutation_rate': (0.00001, 0.01),              #約在0.3最好
        },
        random_state=0,
    )
    optimizer.maximize(                               # 貝式上的參數
        init_points = 10,
        n_iter = 1000,
    )

    # ==== 印出相關 ====
    print(file_name)                                  # 印出檔案名稱
    print(optimizer.max)                              # 印出最佳結果
    print(best_x, best_y)                             # 印出最佳解，以及最佳解的適應度
    bestsolution_copy(optimizer.max['target'])        # 複製最佳解


    # ==== 輸出相關 ====
    Start, Completion, Ava, Sol = fitFunc_2(best_x[0])# 輸出最佳解的各時間資訊
    time_xlsx(Start, Completion)                      # 創建time excel檔
    sequence_xlsx(Start)                              # 創建sequence excel檔
    gantt(Start, Completion)                          # 輸出甘特圖

