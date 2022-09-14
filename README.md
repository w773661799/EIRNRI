# EPIRNN
experiments for EPIRNN and AIRNN
The funciton PIRNN / AIRNN / PEIRNN are the different methods for solve the Schatten-p regularization problem.  

C-1. 能表明 EPIRNN 有改进的地方
EPIRNN 比 AIRNN 快一些, AIRNN 在 $\epsilon$ 较大时候更稳定? 

!! 一个关键的问题, $\epsilon$ for PIRNN 选取多少合适? 
!! 有没有相关文献说这个事儿?
!! 有没有做过关于 $\epsilon$ 的扰动分析的?

## 修改步骤
### 在随机初始点的情况下
- 为了说明 AIRNN 比 PIRNN 更 robust ? 计算得到的结果更小, rank 更低!, 多次试验取均值
> 怎么取均值?

#### reweighted 方法类

1. IRNN\_Ncvx\_2016: 无 $\epsilon$, 直接用 subgradient  

2. \_NuclearIRWA\_2017\_IEEETIP: 固定 $\epsilon$ 不变化 $\color{red}{没给具体值}$, weights 可以直接求得
   - [4] `Iterative Reweighted Algorithms for Matrix Rank Minimization`[论文地址](https://www.jmlr.org/papers/volume13/mohan12a/mohan12a.pdf) 使用了 $\gamma_{0}/\eta^{k}$ 的更新方式, 有数据支撑
     > 针对 rank 约束, 转化为 IRLS-p 约束问题. 对于约束项做成投影问题. 
     > ${\rm Trace}(WXX^{\top})$, 讨论了 $0\le p \le 1$ 的情况, 提出了 IRLS-GP 投影算法

   - [5] `Improved Iteratively Reweighted Least Squares For Unconstrained Smoothed Minimization_XuYangyang, YinWotao`[论文地址](https://web.archive.org/web/20190302145613id_/http://pdfs.semanticscholar.org/9d9b/c32be385490596bb8d630383df19b5e97573.pdf)
     > 针对rank 约束, 提出了 iteratively 算法, 对于矩阵的情况反复求解一个向量形式的线性方程组 IRucLq-M, 给出了一个可以加速的版本?
     > 缺点: !! 需要先验信息, 知道 rank 的值 $K$, 不然算法就是瞎跑? 文中有提到不知道先验的处理方式,
     > 有 $\epsilon_{k}$ 更新方式, 更新依赖于先验条件 $K$ 

   - [6] `A reweighted nuclear norm minimization algorithm for low rank matrix recovery` 'RNNM\_2014\_form\_2017' $\color{red}{找数值实验, 代码}$ 
     > 给出了 fixed $\epsilon$ 的方法? 证明用fixed $\epsilon$, 实验更新了 $\epsilon_{k}$ 原问题是带线性算子的, 线性算子可以转换为投影
   - 
对比固定 $\epsilon$ 情况下的结果


### 在 Warmstart 情况下的比较

- SCP ADMM 带阈值的 spnorm + ADMM , 关键在  ADMM 的子问题, 结果不是 low rank 的 ? 
  - 初始点在 0 不能 warm start ?

- FGSR 在 p=0.5 的效果非常好? 但是不是满rank的!! 
  - 不需要warm start

- 


1. 比较 PIRNN / AIRNN /EPIRNN 在相同 $p$, $\lambda$ 情况下的计算时间, 计算结果

2. 比较不同 $\lambda$ 情况下(对相同的问题, 不同的方法罚参数因子不同), FGSR, IRSVM, IRNN 的计算效率, 重点是要突出 EPIRNN 计算快, 收敛快的特点.
  - ==做到哪一步了? 跟那一个比不过? 好像是干不过 FGSR 的方法?==

3. 比较 


### 非 Warmstart
- 在随机点初始值的比较 
- 在特殊初始值的比较

主要是为了说明 Reweighted 方法在初始化中光滑化因子 $\epsilon$ 的重要作用.

