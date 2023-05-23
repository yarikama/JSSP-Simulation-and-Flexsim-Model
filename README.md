# Simulation

# **排程方法介紹**

## 我們怎麼解決JSSP？

JSSP為NP問題，適合以元啟發式演算法以較高效的方式近似找出較優解，而我們透過這學期休習的**基因演算法與管理科學**從中獲得靈感，以基因演算法（Genetic Algorithm）來解決這次的問題。

## 演算法

### GA概念介紹

接下來才是重點，因為想必基因演算法為本系選修，大部分的人也會使用，我們這組做出了哪些努力來達到與眾不同呢？畢竟，已經2023年了，如果還手動調參數，還要手動輸出、輸入格式，慢慢填進excel或flexsim中，也太麻煩跟不必要了吧！

我們在此之上的特色分別為：**超參數優化（hyperparameter optimization）**，以及我們將整個GA的結果能透過自動化的方式銜接到Flexsim中來展現我們的成果，並模擬JSSP問題，也就是附帶將**問題自動讀檔、將貝式優化演算法的最優解輸出成Excel至Flexsim的Source、Route、Arrival Order等表來模擬問題。**只差Flexsim若是有API我們就能完美的實現一鍵解決JSSP問題了。

### 超參數優化

我們使用[貝葉斯（Bayesian Optimization）](https://zhuanlan.zhihu.com/p/53826787)優化我們在GA中的參數，優化的超參數有交配率、突變率、以及群體個數。

![Untitled](Simulation%20316f6606e90d47aea70c77c3c5c4e70e/Untitled.png)

## **Flexsim：**

Global Table:

| 名稱 | 功用 |
| --- | --- |
| Time | 紀錄每個Job最後完成時間。 |
| Cycletime | 紀錄每個Job數量、同種類Job總花費時間以及其平均（我覺得這個可刪？） |
| Route | 每個Job的機台加工先後順序。 |
| Rank | 紀錄每個Job中的operator在優先順序中的比重。 |
| time | 紀錄各個job進入各個機台的時間和離開的時間。 |
| EachmachinePtime | 紀錄每個operation在機台中運作的時間，方便統整分析結果。 |

流程概念：

1. 頭尾分別設置Source、Sink，中間設置並聯15個workstation，並在每個workstation前設置一個buffer，準備作為Job同時進入排隊流程的協調站。workstation連回flow control，以作為更改Job在不同operation下rank的站點。
2. 將已知的rank,processing time以及事先做好關於source的相關資料匯入flexsim的對應表格中，並開始模擬。
3. 當同個buffer中有多個job的operation等候時，利用先前已知的rank排序，將等待時間最小化。
4. 每個job在進入機台、離開機台的時間，會分別被記錄到global table *time*中。
5. job在機台處理完回到flow control時，rank會更改為下一個global table ”Rank”中operarion的值。
6. 重複動作直到所有Job都完成，便得到最終Makespan time。

# **排程結果與討論**
