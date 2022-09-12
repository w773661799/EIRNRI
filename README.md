# EPIRNN
experiments for EPIRNN and AIRNN
The funciton PIRNN / AIRNN / PEIRNN are the different methods for solve the Schatten-p regularization problem.  

C-1. 能表明 EPIRNN 有改进的地方
EPIRNN 比 AIRNN 快一些, AIRNN 在 $\epsilon$ 较大时候更稳定? 

!! 一个关键的问题, $\epsilon$ for PIRNN 选取多少合适? 
!! 有没有相关文献说这个事儿?
!! 有没有做过关于 $\epsilon$ 的扰动分析的?

## 修改步骤
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

