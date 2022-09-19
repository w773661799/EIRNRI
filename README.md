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
$\color{blue}{Lu Zhaosong} $ 有两篇没发出去的(两篇结果类似): 局部稳定性, 非零奇异值有界

[第一篇\_2014](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.456.6095&rep=rep1&type=pdf), [第二篇\_2017](https://zhaosong-lu.github.io/ResearchPapers/lp-matrix.pdf) 
1. IRNN\_Ncvx\_2016: 无 $\epsilon$, 直接用 subgradient  

2. \_NuclearIRWA\_2017\_IEEETIP: 固定 $\epsilon$ 不变化 $\color{red}{没给具体值}$, weights 可以直接求得
   > $\epsilon>0$ 时 $g(\sigma(X))$ 一直是 L-连续的 
   - [4] `Iterative Reweighted Algorithms for Matrix Rank Minimization`[论文地址](https://www.jmlr.org/papers/volume13/mohan12a/mohan12a.pdf) 使用了 $\gamma_{0}/\eta^{k}$ 的更新方式, 有数据支撑
     > 针对 rank 约束, 转化为 IRLS-p 约束问题. 对于约束项做成投影问题. 
     > ${\rm Trace}(WXX^{\top})$, 讨论了 $0\le p \le 1$ 的情况, 提出了 IRLS-GP 投影算法
     > 给了一个 $\epsilon$ 的讨论范围, 实验选取 $\epsilon=1e^{-2}$ works well, 

   - [5] `Improved Iteratively Reweighted Least Squares For Unconstrained Smoothed Minimization_XuYangyang, YinWotao`[论文地址](https://web.archive.org/web/20190302145613id_/http://pdfs.semanticscholar.org/9d9b/c32be385490596bb8d630383df19b5e97573.pdf)
     > 针对rank 约束, 提出了 iteratively 算法, 对于矩阵的情况反复求解一个向量形式的线性方程组 IRucLq-M, 给出了一个可以加速的版本?
     > 缺点: !! 需要先验信息, 知道 rank 的值 $K$, 不然算法就是瞎跑? 文中有提到不知道先验的处理方式,
     > 有 $\epsilon_{k}$ 更新方式, 更新依赖于先验条件 $K$ 

   - [6] `A reweighted nuclear norm minimization algorithm for low rank matrix recovery` 'RNNM\_2014\_form\_2017' $\color{red}{找数值实验, 代码}$ 
     > 给出了 fixed $\epsilon$ 的方法? 证明用fixed $\epsilon$, 实验更新了 $\epsilon_{k}$ 原问题是带线性算子的, 线性算子可以转换为投影
   - [8] `Generalized Nonconvex Nonsmooth_LRMM_2014_LuCaiyi`说 KL 有局部线性收敛的结果? ?? 原文没找到
     > LuCanyi 的算法部分会一直更新 $\lambda$ ?  有源码[github](https://github.com/canyilu/Iteratively-Reweighted-Nuclear-Norm-Minimization)
     > 更新 $\lambda$ 的方法 LuZhaosong 好像用过

对比固定 $\epsilon$ 情况下的结果

### 数值实验类

#### 纵项实验对比

##### TIP_2017

PIRNN
$$
\min_{X} \quad \frac{1}{2} \|Y-{\rm P}_{\Omega}(X) \|_{F}^{2} + \sum_{i=1}^{N} g[\sigma_{i}(X)]
$$

实验固定了 $\epsilon$ 但是没有找到数据? 没有源代码

对比的实验:
- APGL: Accelerated Proximal Gradient with Line search
- LMaFit: Low-Rank Matrix Fitting
- TNNR: Truncated Nuclear Norm Regularization 

##### Generalized Nonconvex Nonsmooth Low-Rank Minimization_2014_LuCanyi

IRNN

$$
\min_{X} \quad \frac{1}{2} \|{\rm P}_{\Omega}(X-M) \|_{F}^{2} + \sum_{i=1}^{N} g_{\lambda}[\sigma_{i}(X)]
$$

2014 的一篇实验中会更新 $\lambda$, $g(\cdot)$ 是一般的可分离 penalty function
和 TIP 比多了一个可变参数, TIP 直接固定了 $\lambda$

对比实验:
- ALM: Augmented Lagrange Multiplier
- APGL
- LMaFit
- TNNR, TNNR-ADMM

#### 横向方法: 其他类方法求解 sp-norm

- A novel variational form of the Schatten-p quasi-norm_2020

- Efficient Inexact Proximal Gradient Algorithm for Nonconvex Problems_2017
   > niAPG(Nonconvex inexact APG), nmAPG(Nonmonotone APG)

- Factor Group-Sparse Regularization for Efficient Low-Rank Matrix Recovery_2019NeurIPS
   > FGSR
- A novel variational form of the Schatten-p quasi-norm_2020NeurIPS
   > 也用 FGSR??

- An accelerated IRNN-Iteratively Reweighted Nuclear Norm algorithm for nonconvex nonsmooth low-rank minimization problems_2021JCAM

- Inexact Proximal Gradient Methods for Non-Convex and Non-Smooth Optimization_2018AAAI
   > IPG
   - 对比实验? + niAPG, nmAPG, IRNN, FaNCL(Fast Nonconvex low-rank learning)
   - 和 p=1 时的 convex nuclear-norm-regularized 方法作对比
     - activate subspace selection, ALT-input, ???


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


## 光滑因子的选择
根据model identification 的性质, 可以识别的 feature 应该满足

$$
\sigma(X) \ge (\frac{\lambda p}{\beta C_{1}})^\frac{1}{1-p} 
$$

其中 $\lambda >0$ 是罚因子, $p\in(0,1)$是约束形式参数, $\beta \ge L_{f} >1$ 是与 Lipschitz 有关的常数, $C_{1}$ 是可识别的最大 feature 的上界

则对于可识别的 feature 当其某一步 $\sigma=0$ 时, 光滑因子的选择应当满足:

$$
p(\sigma+\epsilon)^{p-1} < \frac{\beta C_{1}}{\lambda}
$$

即

$$
\epsilon_{0} > (\frac{\lambda p}{\beta C_{1}})^{\frac{1}{1-p}} 
$$

1. 讨论 $\epsilon$ 关于 $p$ 的性质
  $$
  f(x) = [c(1-\frac{1}{x})]^{x},
  $$
  其中 $\frac{1}{1-p} = x\in(1,+\infty), c = \frac{\lambda}{\beta C_{1}}\in(0,+\infty)$
  则:
  $$
  f^{'}(x) = e^{x\ln[c(1-\frac{1}{x})]}(\ln[c(1-\frac{1}{x})] + \frac{1}{x-1})
  $$
  记: 
  $$
  \hat{g}(x) =\ln[c(1-\frac{1}{x})] + \frac{1}{x-1} = g(y) = \ln[\frac{c}{y+1}] + y
  $$ 
  其中 $y=\frac{1}{x-1} = \frac{1}{p}-1\in(0,+\infty)$, 
  得:
   - $c\ge 1$ 时, $g(y) > 0$ 故, $f(x)$ 在$(1,+\infty)$ 严格单调增, 即 $\epsilon$ 关于 $p$ 在 $(0,1)$ 上严格单调增  
   - 当 $c\le 1$ 是存在 $y_{m}$ 满足 $c e^{y_{m}} = y_{m}+1$, 且 $y_{m} = \frac{1}{p_{m}}-1$ 所以:
     -  $y<y_{m}$ 时, 即 $p>p_{m}$, $g(y)<0$, $\epsilon$ 关于 $p$单调减
     -  $y>y_{m}$ 时, 即 $p<p_{m}$, $g(y)>0$, $\epsilon$ 关于 $p$单调增
    
    注意到 $c=\frac{\lambda}{\beta C_{1}}$

2. $\epsilon$ 关于 $\lambda$ 单调增加
3. $\epsilon$ 关于 $\beta$ 单调减少
