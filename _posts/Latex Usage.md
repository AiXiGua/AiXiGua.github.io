---
typora-copy-images-to: images
title: "Latex Usage"
last_modified_at: 2018-03-08T21:28:04-22:00:00
categories:
  - Skill
tags:
  - Skill
  - Latex
---

##Fonts
- Use `\mathbb` or `\Bbb` for "blackboard bold": $$\mathbb CHNQRZ$$.
- Use `\mathbf` for boldface: $$\mathbf ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$$.
- Use `\mathtt` for "typewriter" font: $$\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$$.
- Use `\mathrm` for roman font: $$\mathrm ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$$.
- Use `\mathsf` for sans-serif font:$$\mathsf ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$$.
- Use `\mathcal` for "calligraphic" letters: $$\mathcal ABCDEFGHIJKLMNOPQRSTUVWXYZ$$.
- Use `\mathscr` for script letters:$$\mathscr ABCDEFGHIJKLMNOPQRSTUVWXYZ$$.
- Use `\mathfrak` for "Fraktur" (old German style) letters:$$\mathfrak ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$$.

## symbol

- Use `\geq` and `\leq` for `>= and <=`: $$\geq,  \leq$$ 
- Use `\infty`  for  Infinite: $$\infty$$ 
- Use`\tilde` for tilde: $$\tilde a$$

|    symbol    |  markdown  |
| :----------: | :--------: |
|  $$\alpha$$  |  `\alpha`  |
|  $$\beta$$   |  `\beta`   |
|  $$\gamma$$  |  `\gamma`  |
|  $$\delta$$  |  `\delta`  |
| $$\epsilon$$ | `\epsilon` |
|  $$\zeta$$   |  `\zeta`   |
|   $$\eta$$   |   `\eta`   |
|  $$\theta$$  |  `\theta`  |
|  $$\iota$$   |  `\iota`   |
|  $$\kappa$$  |  `\kappa`  |
| $$\lambda$$  | `\lambda`  |
|   $$\mu$$    |   `\mu`    |
|   $$\nu$$    |   `\nu`    |
|   $$\xi$$    |   `\xi`    |
|   $$\pi$$    |   `\pi`    |
|   $$\rho$$   |   `\rho`   |
|  $$\sigma$$  |  `\sigma`  |
|   $$\tau$$   |   `\tau`   |
| $$\upsilon$$ | `\upsilon` |
|   $$\phi$$   |   `\phi`   |
|   $$\chi$$   |   `\chi`   |
|   $$\psi$$   |   `\psi`   |
|  $$\omega$$  |  `\omega`  |

##bracket
|                 bracket                  |                 markdown                 |
| :--------------------------------------: | :--------------------------------------: |
|      $$\left( \frac{a}{b} \right)$$      |        `\left(\frac{a}{b}\right)`        |
|       $$\left[\frac{a}{b}\right]$$       |        `\left[\frac{a}{b}\right]`        |
|     $$\left\{ \frac{a}{b} \right\}$$     |       `\left\{\frac{a}{b}\right\}`       |
| $$\left\langle\frac{a}{b}\right\rangle$$ |  `\left\langle\frac{a}{b}\right\rangle`  |
|        $\left|\frac{a}{b}\right|$        |        `\left|\frac{a}{b}\right|`        |
|      $\left\|\frac{a}{b} \right\|$       |       `\left\|\frac{a}{b}\right\|`       |
|  $\left\lfloor\frac{a}{b}\right\rfloor$  |  `\left\lfloor\frac{a}{b}\right\rfloor`  |
| $\left \lceil \frac{c}{d} \right \rceil$ |   `\left\lceil\frac{c}{d}\right\rceil`   |
|   $\left/\frac{a}{b}\right\backslash$    |   `\left/\frac{a}{b}\right\backslash`    |
| $\left\uparrow \frac{a}{b}\right \downarrow$ | `\left\uparrow\frac{a}{b}\right\downarrow` |
| $\left\Uparrow\frac{a}{b} \right\Downarrow$ | `\left\Uparrow\frac{a}{b}\right\Downarrow` |
| $\left\updownarrow\frac{a}{b}\right\Updownarrow$ | `\left\updownarrow \frac{a}{b}\right\Updownarrow` |
|         $	\left [ 0,1 \right )$          |           `	\left[0,1\right)`            |
|      $\left \langle \psi \right |$       |        `\left\langle\psi\right|`         |
|     $	\left \{ \frac{a}{b} \right .$     |       `	\left\{\frac{a}{b}\right.`       |
|     $	\left . \frac{a}{b} \right \}$     |       `	\left.\frac{a}{b}\right\}`       |

*备注：* 

- 可以使用`\big, \Big, \bigg, \Bigg`控制括号的大小，比如代码`**\Bigg** ( **\bigg** [ **\Big** \{ **\big** \langle \left | \| \frac{a}{b} \| \right | **\big** \rangle **\Big** \} **\bigg** ] **\Bigg** )`显示：

  $$begin\Bigg( \bigg [ \Big \{ \big \langle \left | \| \frac{a}{b} \| \right | \big \rangle \Big \} \bigg] \Bigg )$$


