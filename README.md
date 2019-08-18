# Deep Neural Network from scratch on Image Classification - Dogs or Cats?

## **Phuong T.M. Chu and Hai Nguyen**

**Deep Neural Network from scratch on Image Classification - Dogs or Cats?** is the project that we finished after the 5th week of studying **Machine Learning**.

![](https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg)

## INTRODUCTION
**Dogs vs. Cats** [dataset](https://www.kaggle.com/c/dogs-vs-cats/data) provided by  Microsoft Research contains 25,000 images of dogs and cats with the labels 
* 1 = dog
* 0 = cat 

## BUILDING DEEP NEURAL NETWORK FROM SCRATCH

In this notebook, we implemented all the functions required to build a deep neural network.

![](https://i.imgur.com/ivhZhmx.png)

**Notation**:
- Superscript <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;\fn_phv&space;\small&space;$[l]$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;\fn_phv&space;\small&space;$[l]$" title="\small $[l]$" /></a> denotes a quantity associated with the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$l^{th}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$l^{th}$" title="$l^{th}$" /></a> layer. 
    - Example: <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$a^{[L]}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$a^{[L]}$" title="$a^{[L]}$" /></a> is the <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$L^{th}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$L^{th}$" title="$L^{th}$" /></a> layer activation. $W^{[L]}$ and $b^{[L]}$ are the $L^{th}$ layer parameters.
- Superscript $(i)$ denotes a quantity associated with the $i^{th}$ example. 
    - Example: $x^{(i)}$ is the $i^{th}$ training example.
- Lowerscript $i$ denotes the $i^{th}$ entry of a vector.
    - Example: $a^{[l]}_i$ denotes the $i^{th}$ entry of the $l^{th}$ layer's activations).
    
The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_params`, you should make sure that your dimensions match between each layer. Given $n^{[l]}$ is the number of units in layer $l$. Thus for example if the size of our input $X$ is $(12288, 209)$ (with $m=209$ examples) then:

| |**Shape of W**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; |**Shape of b**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|**Activation**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|**Shape of Activation**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|:-|:-|:-|:-|:-|
|**Layer 1**|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$(n^{[1]},12288)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$(n^{[1]},12288)$" title="$(n^{[1]},12288)$" /></a>|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$(n^{[1]},1)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$(n^{[1]},1)$" title="$(n^{[1]},1)$" /></a>|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$Z^{[1]}&space;=&space;W^{[1]}&space;X&space;&plus;&space;b^{[1]}&space;$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$Z^{[1]}&space;=&space;W^{[1]}&space;X&space;&plus;&space;b^{[1]}&space;$" title="$Z^{[1]} = W^{[1]} X + b^{[1]} $" /></a>|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$(n^{[1]},209)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$(n^{[1]},209)$" title="$(n^{[1]},209)$" /></a>|
| **Layer 2**|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$(n^{[2]},&space;n^{[1]})$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$(n^{[2]},&space;n^{[1]})$" title="$(n^{[2]}, n^{[1]})$" /></a>|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$(n^{[2]},1)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$(n^{[2]},1)$" title="$(n^{[2]},1)$" /></a>|<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$Z^{[2]}&space;=&space;W^{[2]}&space;A^{[1]}&space;&plus;&space;b^{[2]}$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$Z^{[2]}&space;=&space;W^{[2]}&space;A^{[1]}&space;&plus;&space;b^{[2]}$" title="$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$" /></a>|$(n^{[2]}, 209)$|
|$\vdots$| $\vdots$ | $\vdots$|$\vdots$|$\vdots$|
|**Layer L-1** | $(n^{[L-1]}, n^{[L-2]})$ | $(n^{[L-1]}, 1)$ | $Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ | $(n^{[L-1]}, 209)$|
|**Layer L** | $(n^{[L]}, n^{[L-1]})$ | $(n^{[L]}, 1)$|  $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$|$(n^{[L]}, 209)$ |

Remember that when we compute <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$W&space;X&space;&plus;&space;b$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$W&space;X&space;&plus;&space;b$" title="$W X + b$" /></a> in python, it carries out broadcasting. For example, if: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$$&space;W&space;=&space;\begin{bmatrix}&space;j&space;&&space;k&space;&&space;l\\&space;m&space;&&space;n&space;&&space;o&space;\\&space;p&space;&&space;q&space;&&space;r&space;\end{bmatrix}\;\;\;&space;X&space;=&space;\begin{bmatrix}&space;a&space;&&space;b&space;&&space;c\\&space;d&space;&&space;e&space;&&space;f&space;\\&space;g&space;&&space;h&space;&&space;i&space;\end{bmatrix}&space;\;\;\;&space;b&space;=\begin{bmatrix}&space;s&space;\\&space;t&space;\\&space;u&space;\end{bmatrix}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$$&space;W&space;=&space;\begin{bmatrix}&space;j&space;&&space;k&space;&&space;l\\&space;m&space;&&space;n&space;&&space;o&space;\\&space;p&space;&&space;q&space;&&space;r&space;\end{bmatrix}\;\;\;&space;X&space;=&space;\begin{bmatrix}&space;a&space;&&space;b&space;&&space;c\\&space;d&space;&&space;e&space;&&space;f&space;\\&space;g&space;&&space;h&space;&&space;i&space;\end{bmatrix}&space;\;\;\;&space;b&space;=\begin{bmatrix}&space;s&space;\\&space;t&space;\\&space;u&space;\end{bmatrix}$$" title="$$ W = \begin{bmatrix} j & k & l\\ m & n & o \\ p & q & r \end{bmatrix}\;\;\; X = \begin{bmatrix} a & b & c\\ d & e & f \\ g & h & i \end{bmatrix} \;\;\; b =\begin{bmatrix} s \\ t \\ u \end{bmatrix}$$" /></a>

Then <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$WX&space;&plus;&space;b$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$WX&space;&plus;&space;b$" title="$WX + b$" /></a> will be:

<a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\dpi{100}&space;$$&space;WX&space;&plus;&space;b&space;=&space;\begin{bmatrix}&space;(ja&space;&plus;&space;kd&space;&plus;&space;lg)&space;&plus;&space;s&space;&&space;(jb&space;&plus;&space;ke&space;&plus;&space;lh)&space;&plus;&space;s&space;&&space;(jc&space;&plus;&space;kf&space;&plus;&space;li)&plus;&space;s\\&space;(ma&space;&plus;&space;nd&space;&plus;&space;og)&space;&plus;&space;t&space;&&space;(mb&space;&plus;&space;ne&space;&plus;&space;oh)&space;&plus;&space;t&space;&&space;(mc&space;&plus;&space;nf&space;&plus;&space;oi)&space;&plus;&space;t\\&space;(pa&space;&plus;&space;qd&space;&plus;&space;rg)&space;&plus;&space;u&space;&&space;(pb&space;&plus;&space;qe&space;&plus;&space;rh)&space;&plus;&space;u&space;&&space;(pc&space;&plus;&space;qf&space;&plus;&space;ri)&plus;&space;u&space;\end{bmatrix}$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;\dpi{100}&space;$$&space;WX&space;&plus;&space;b&space;=&space;\begin{bmatrix}&space;(ja&space;&plus;&space;kd&space;&plus;&space;lg)&space;&plus;&space;s&space;&&space;(jb&space;&plus;&space;ke&space;&plus;&space;lh)&space;&plus;&space;s&space;&&space;(jc&space;&plus;&space;kf&space;&plus;&space;li)&plus;&space;s\\&space;(ma&space;&plus;&space;nd&space;&plus;&space;og)&space;&plus;&space;t&space;&&space;(mb&space;&plus;&space;ne&space;&plus;&space;oh)&space;&plus;&space;t&space;&&space;(mc&space;&plus;&space;nf&space;&plus;&space;oi)&space;&plus;&space;t\\&space;(pa&space;&plus;&space;qd&space;&plus;&space;rg)&space;&plus;&space;u&space;&&space;(pb&space;&plus;&space;qe&space;&plus;&space;rh)&space;&plus;&space;u&space;&&space;(pc&space;&plus;&space;qf&space;&plus;&space;ri)&plus;&space;u&space;\end{bmatrix}$$" title="$$ WX + b = \begin{bmatrix} (ja + kd + lg) + s & (jb + ke + lh) + s & (jc + kf + li)+ s\\ (ma + nd + og) + t & (mb + ne + oh) + t & (mc + nf + oi) + t\\ (pa + qd + rg) + u & (pb + qe + rh) + u & (pc + qf + ri)+ u \end{bmatrix}$$" /></a>

### *This project explores the applicability of deep neural network by tunning these hyperparameters:*
1. Learning rate
2. Number of hidden layers
3. Number of nodes in each hiden layers
4. Number of iterations



## MODEL PERFORMANCE

## CONCLUSION
The current literature suggests machine classifiers can score above [80% accuracy](chrome-extension://cbnaodkpfinfiipjblikofhlhlcickei/src/pdfviewer/web/viewer.html?file=http://xenon.stanford.edu/~pgolle/papers/dogcat.pdf) on this task.
We successfully achieved the accuracy of % with these detailed hyperparameters:
