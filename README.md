# EPIRNN
experiments for EPIRNN and AIRNN
The funciton PIRNN / AIRNN / PEIRNN are the different methods for solve the Schatten-p regularization problem.  

## 修改步骤
### 在 Warmstart 情况下的比较
1. 比较 PIRNN / AIRNN /EPIRNN 在相同 $p$, $\lambda$ 情况下的计算时间, 计算结果

2. 比较不同 $\lambda$ 情况下(对相同的问题, 不同的方法罚参数因子不同), FGSR, IRSVM, IRNN 的计算效率, 重点是要突出 EPIRNN 计算快, 收敛快的特点.
  - ==做到哪一步了? 跟那一个比不过? 好像是干不过 FGSR 的方法?==
3. 比较 

### 非 Warmstart
- 在随机点初始值的比较 
- 在特殊初始值的比较

主要是为了说明 Reweighted 方法在初始化中光滑化因子 $\epsilon$ 的重要作用.
