# Scheduling & Simulation

# 我們怎麼解決JSSP？

JSSP為NP問題，適合以元啟發式演算法以較高效的方式近似找出較優解，而我們透過這學期修習的**基因演算法與管理科學**中獲得靈感，決定以基因演算法（Genetic Algorithm）並利用貝式優化（Bayesian Optimization）來解決這次的問題。

[方法說明.pdf](簡報.pdf)

# 演算法

## GA概念介紹

基因演算法是一種優化解的演算法，通過模擬遺傳、進化的過程，在一組隨機解中找到最佳解。並以基因編碼表示解，通過選擇、交叉和變異等操作來進行搜索和改進。

接下來才是重點，因為想必基因演算法為本系選修，大部分的人也會使用，我們這組做出了哪些努力來達到與眾不同呢？畢竟，已經2023年了，如果還手動調參數，還要手動輸出、輸入格式，慢慢填進excel或flexsim中，也太麻煩跟不必要了吧！

我們在此之上的特色分別為：**超參數優化（hyperparameter optimization）**，以及我們將整個GA的結果能透過自動化的方式銜接到Flexsim中來展現我們的成果，並模擬JSSP問題，也就是附帶將**問題自動讀檔、將貝式優化演算法的最優解輸出成Excel至Flexsim的Source、Route、Arrival Order等表來模擬問題。**只差Flexsim若是有API我們就能完美的實現一鍵解決JSSP問題了。

## 超參數優化

我們使用[貝葉斯（Bayesian Optimization）](https://zhuanlan.zhihu.com/p/53826787)優化我們在GA中的參數，優化的超參數有交配率、突變率、以及群體個數。

![Untitled](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/Untitled.png)

# 自動 Flexsim 腳本驗證

我們將整個 GA 的結果透過自動化（Scipts）的⽅式銜接到 Flexsim 中來展現我們的成果，並模擬 JSSP 問題，也就是將問題自動轉成Flexsim適用的檔案格式、將貝式優化演算法的最優解輸出成 Excel，匯⼊⾄ Flexsim 的 Source、Route、Arrival Order 等 Tables 來模擬問題，我們便能⼀鍵驗證在 Genetic Algorithm 中所得出的結果是否能實現了。

備註：預設檔案為 rcmax_20_15_8，如果要使用預設以外的題目，請在 flexsim 的匯⼊ excel 時選擇正確的 solution 檔名。目前我們只實做 20_15 機台作為交叉驗證。

# **Flexsim** 格式  ****

## **表格說**明

| Global Table Type | Utility |
| --- | --- |
| Time | 紀錄每個Job最後完成時間。 |
| Route | 紀錄每個Job的機台加工先後順序。 |
| Rank | 紀錄每個Job在不同Operation的優先順序數值。 |
| time | 紀錄各個Job進入各個機台的時間和離開的時間。 |
| EachmachinePtime | 紀錄每個Operation在機台中運作的時間，方便統整分析結果。 |
| State | 紀錄每個機台各自已經做完幾個Job。 |
| Sequence | 紀錄每個機台各自加工Job的順序。 |

## 流程概念

1. 頭尾分別設置 Source、Sink，中間設置並聯15個機台，並在每個機台前設置一個 Buffer，準備作為 Job 同時進入排隊流程的排序等候站。機台連回 Flow Control，以作為移動 Job 至下一等候站，以及更新 Job在不同 Operation Rank 的站點。
2. 將已知的 Rank, Route 以及 Sequence 的相關資料自動化匯入 Flexsim 對應的表格中。
3. 開始模擬，機台初始設定為全數 Input 都先關閉，避免先進入但還沒輪到順序的 Job 進入機台（因為派工法則不是 FIFO，或 LIFO，是按照演算法得到的 Rank 進行排序）。
4. Job 經過 Flow Control 移動至預定之下一站點，並更新該輪 Rank。
5. 當 Job 進入該輪 Buffer 時，會利用更新後的 Rank 排序。
6. 當 Rank 順位最前面的 Job 編號和 Sequence 內所示之該機台當時輪到的 Job 編號吻合時， Buffer 傳訊息給機台，要求其打開入口讓 Job 進入運作，若當時沒有吻合的 Job 出現，則位於  Buffer 中的所有 Job 必須全數排隊等待，不能因為先到就先進入。
7. 當有 Job 進入機台時，機台入口關上，直到有下一個順位的 Job 進入 Buffer 時，Buffer 才會再次傳訊給機台要求其打開。
8. 如果只做以上設定會少考慮一種情況：當該輪順序2的 Job 進入時，順序1的 Job 正在排隊等待，等到順序1進入機台並完成加工，離開機台的過程，都沒有新進入 Buffer 的 Job，那麼在沒有人檢查是否吻合的情況下，順序2的 Job 會在 Buffer 中乾等，浪費時間。為了解決這類問題，我們在機台的 Exit 中設置了傳訊功能，告訴 Buffer 機台中已經沒有 Job，請 Buffer 再次檢查有沒有符合順位資格的 Job 在 Buffer 等待。
9. 每個 Job 在進入機台、離開機台的時間，會分別被記錄到 Global Table T*ime*中。
10. Job 在機台處理完回到 Flow Control 時，會領取下一輪的 Rank 並重複動作直到所有 Job 都完成，便能得到最終 Makespan Time 進行分析。

# **排程結果與討論**

| Name | Size | LB | UB | Solution | Solution-UB |
| --- | --- | --- | --- | --- | --- |
| Dmu01_rcmax_20_15_8 | 20x15 | 2601 | 2669 | 2944 | 275 |
| Dmu01_rcmax_20_15_5 | 20x15 | 2731 | 2731 | 2972 | 241 |
| Dmu01_rcmax_20_15_4 | 20x15 | 2501 | 2563 | 2966 | 403 |
| Dmu01_rcmax_20_15_1 | 20x15 | 2749 | 2749 | 3178 | 429 |
| Dmu44_cscmax_20_15_7 | 20x15 | 3283 | 3475 | 3917 | 442 |
| Dmu44_cscmax_20_15_6 | 20x20 | 3575 | 4035 | 4517 | 482 |

# 備註

## 自動化備註

預設檔案為rcmax_20_15_8，如果要使用預設以外的題目，請在flexsim的匯入excel時請選擇正確的solution檔名。目前我們只實做20_15機台作為交叉驗證。

## Python備註

code中有註解介紹函式，請確保有安裝numpy, matplotlib, pandas, bayes_opt以及openpyxl。若是有任一函式庫沒有可至cmd或是bash中按照下列方法安裝：

```bash
pip install numpy matplotlib pandas bayesian-optimization
```

## Genetic Algorithm 備註

GA 中設定的 iteration 要跑完才能有 sequence。若尚未跑出 best solution 就想先停⽌演算法，做 flexsim 測試，
可利用已求得現有 solution 以及 route，透過’ADD_SEQUENCE_TO_RANK’求得 sequence 做 flexsim 模擬。

# 成果與範例

[final.fsm](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/final.fsm)

[rcmax_20_15_8_route.xlsx](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/rcmax_20_15_8_route.xlsx)

[rcmax_20_15_8_solution_71_result_2944.xlsx](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/rcmax_20_15_8_solution_71_result_2944.xlsx)

[rcmax_20_15_5_route.xlsx](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/rcmax_20_15_5_route.xlsx)

[rcmax_20_15_5_solution_621_result_2972.xlsx](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/rcmax_20_15_5_solution_621_result_2972.xlsx)

[rcmax_20_15_4_route.xlsx](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/rcmax_20_15_4_route.xlsx)

[rcmax_20_15_4_solution_160_result_2966.xlsx](Scheduling%20&%20Simulation%20316f6606e90d47aea70c77c3c5c4e70e/rcmax_20_15_4_solution_160_result_2966.xlsx)

[模擬學第三組期末書面報告.pdf](書面報告.pdf)