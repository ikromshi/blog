---
layout: default
title: "My Notes on Probability Theory"
date:   2025-10-23
excerpt: Discrete/continuous random variables and distributions, conditional probability and independence, expectation, limits, random walks.
---

# Probability Theory

## First Principles

- Sample space $\Omega$: Set of all possible outcomes of a random experiment.
- Outcome $\omega$: The elements of a sample space.
- Event: A subset of the sample space; a collection of outcomes.
- Probability Function: A function $P$ that assigns numbers to the elements $\omega \in \Omega$ such that
    1. $P(\omega) \geq 0$
    2. $\sum_{\omega}{P(\omega)}=1$
    3. For events $A,$ $P(A)=\sum_{\omega \in A}P(\omega)$
- Counting
    1. Multiplication principle: If there are $m$ ways for one thing to happen, and $n$ ways for a second thing to happen, there are $mn$ ways for both things to happen.
    2. Permutations: A permutation of $\{1, \dots, n\}$ is an $n$-element ordering of the $n$ numbers. There are $n!$ permutations of an $n$-element set.
    3. Binomial coefficient: The binomial coefficient $\binom{n}{k}=
    \frac{n!}{k!(n-k)!}$ counts: (i) the number of $k$-element subsets of $\{1, \dots, n\}$ and (ii) the number of n element 0-1 sequences with exactly $k$ ones. Each subset is also referred to as a combination.
- DeMorgan’s Law:
    - $(A \cap B)^c=A^c \cup B^c$
    - $(A \cup B)^c=A^c \cap B^c$
- Problem-solving strategies:
    1. Taking complements: Finding $P(A^c)$, the probability of the complement of the event, might be easier in some cases than finding $P(A)$, the probability of the event. This arises in “at least” problems. For instance, the complement of the event that “at least one of several things occur” is the event that “none of those things occur”. In the former case, the event involves a union. In the latter case, the event involves an intersection.
    2. Principle of Inclusion\Exclusion:
        - $P(A\cup B)=P(A)+P(B)-P(AB)$
        - $(A\cup B \cup C)=P(A)+P(B)+P(C)-P(AB)-P(AC)-P(BC)+P(ABC)$
    

## Conditional Probability and Independence

- Conditional Probability: In conditional probability, some information about the outcome of the random experiment is known - the probability is conditional on that knowledge $P(A\mid B)=\frac{P(AB)}{P(B)}$
- New information: Partial information about the outcome of a random experiment actually changes the set of possible outcomes, that is, it changes the sample space of the original experiment and reduces it based on new information. An example is, what is the probability of getting all heads in three tosses if we know the first toss is heads.
- Finding *P(A* AND *B): $P(AB)=P(A\mid B)P(B)$*
    - By extension, $P(ABC)=P(C\mid AB)P(B\mid A)P(A)$
- Law of total probability: Suppose $B_1, \dots, B_k$ is a partition of the sample space. Then,
    
    $P(A)=\sum_{i=1}^{k}P(A\mid B_i)P(B_i)$
    
    Example: 7% of men and 0.4% of women are colorblind. Picked at random, what’s the probability that a person is colorblind?
    
- Bayes’ formula: for events $A$ and $B$,
    
    $$
    P(B\mid A)=\frac{P(A\mid B)P(B)}{P(A\mid B)P(B) + P(A\mid B^c)P(B^c)}
    $$
    
- Problem solving strategies:
    1. Tree diagrams: Tree diagrams are intuitive and useful tools for finding probabilities of events that can be ordered sequentially.
    2. Conditioning: Given events $B_1,\dots,B_k$ and applying the law of toal probability, whereby the conditional probabilities $P(A\mid B_i)$ are easier and more natural to solve than $P(A)$.
    3. Hypothetical tables: Hypothetical tables can be constructed for many scenarios involving probabilities and can be used in many of the same situations as tree diagrams.
- Independent Events: Events $A$ and $B$ are independent if $P(A\mid B)=P(A)$. Equivalently, $P(AB)=P(A)P(B)$
- Mutual independence: For general collections of events, independence means that for every finite sub-collection $A_1, \dots, A_k,$
    
    $$
    P(A_1, \dots, A_k)=P(A_1), \dots, P(A_k)
    $$
    
    Mutual independence is a synonym for independence.
    
- Pairwise independence: A collection of events is pairwise independent if $P(A_iA_j)=P(A_i)P(A_j)$ for all pairs of events.
- $A$ before $B$: In repeated independent trials, if $A$ and $B$ are mutually exclusive events, the probability that $A$ occurs before $B$ is $\frac{P(A)}{P(A)+P(B)}$

## Discrete Random Variables

- Random variable: A random variable assigns numerical values to the outcomes of a random experiment. We write $\{X=x\}$ for the event that the random variable $X$ takes the value of $x$, where $x$ is a specific number.
- Random variable as a function: a random variable assigns the outcome of the sample space a real number. For example, the probability of getting exactly two heads in coin tosses is written as $P(X=2)$ or $P(\{\omega : X(\omega) = 2\}$ — $\{w : Property\}$ describes the set of all $\omega$ that satisfies some property.
- Uniform Random Variable: Let $S=\{s_1,\dots,s_k\}$ be a finite set. A random variable $X$ is uniformly distributed on set $S$ if
    
    $$
    P(X=s_i)=\frac{1}{k}, \;\;\; for\; i=1,\dots,k
    $$
    
    we write $X\sim Unif(S)$. The tilde stands for “is distributed as”. 
    
- Independent random variables:
    1. Discrete random variables $X$ and $Y$ are said to be independent if $P(X=x\mid Y=y)=P(X=x)$, for all $x, y$.
    2. Equivalently, $O(X=x, Y=y)=P(X=x)P(Y=y)$, for all $x, y$.
- Bernoulli Random Variable: A random variable that takes only two values 0 and 1 is called a *Bernoulli random variable*.
    - Bernoulli distribution: A random variable $X$ has a *Bernoulli distribution* with parameter $p$ if
    
    $$
    P(X=1)=p \;and \; P(X=0)=1-p
    $$
    
    - I.I.D. sequences: a sequence of random variables is said to be *independent and identically distributed* (i.i.d.) if the random variables are independent and have the same probability distribution (including all distribution parameters).
- Binomial Distribution:
    - A random variable $X$ is said to have a binomial distribution with parameters $n$ and $p$ if
    
    $$
    P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}, \; for \; k=0, 1, \dots, n
    $$
    
    - We write $X \sim Binom(n, p)$, or $Bin(n, p)$. The binomial distribution models the probability of obtaining exactly $k$ successes in $n$ Bernoulli trials.
    - Binomial setting: The binomial distribution arises as the number of successes in $n$ i.i.d. Bernoulli trials. The binomial setting requires:
        1. A fixed number of $n$ of independent trials.
        2. Trials take one of the two possible values.
        3. Each trial has a constant probability $p$ of success.
    - Random graphs: let $deg(v)$ be the degree of vertex $v$ in a random graph. There are $n-1$ possible edges incident to $v$, as there are $n-1$ vertices left in the graph, other than $v$. Each of those edges occurs with probability $p$. Thus, for any vertex $v$ in our random graph, the degree $deg(v)$ is a random variable that has a binomial distribution with parameters $n-1$ and $p$:
    
    $$
    P(deg(v)=k)=\binom{n-1}{k}p^k(1-p)^{n-k-1} \; for \; k=0, \dots,n-1
    $$
    
- Poisson Distribution:
    - A random variable $X$ has a Poisson distribution with parameter $\lambda>0$ (average sense) if
        
        $$
        P(X=k)=\frac{e^{-\lambda}\lambda^k}{k!}, \; for \; k=0, 1, \dots
        $$
        
    - Poisson setting: The Poisson setting arises in the context of discrete counts of “events” that occur over space or time with small probability and where successive events are independent. For example, the number of babies born on a maternity ward in one day.
    - Binomial distribution can be modeled with the Poisson distribution as follows:
        
        $$
        \lim_{n\to \infty}\binom{n}{k}(\frac{\lambda}{n})^k(1-\frac{\lambda}{n})^{n-k}=
        
        \frac{e^{-\lambda} \lambda^k}{k!}
        $$
        
    - Poisson distribution can also be modeled using the Binomial distribution with $\lambda=np$

## Expectation and More with Discrete Random Variables

- Probability Mass Function: For a random variable $X$ that takes values in a set $S$, the probability mass function of $X$ is the probability function
    
    $$
    m(x)=P(X=x), \; for \; x \in S
    $$
    
    and implicitly, 0, otherwise.
    
- Discrete probability distributions:
    
    
    | Distribution | Parameters | Probability Mass Function | Expectation | Variance |
    | --- | --- | --- | --- | --- |
    | Bernoulli | $p$ | $P(X=k)=
    \begin{cases}
    p, \; & k=1 \\
    1-p & k=0
    \end{cases}$ | $p$ | $p(1-p)$ |
    | Binomial | $n,p$ | $P(X=k)=
    \binom{n}{k}p^k(1-p)^{n-k}, \;k=0, 1, \dots$ | $np$ | $np(1-p)$ |
    | Poisson | $\lambda$ | $P(X=k)=\frac{e^{-\lambda} \lambda^k}{k!}, \; k=0,1,\dots$  | $\lambda$ | $\lambda$ |
    | Uniform |  | $P(X=x_k)=\frac{1}{n}, \; k=1,\dots,n$ | $(n+1)/2$ | $(n^2-1)/12$ |
    | Indicator Variable | $p$ |  | $P(A)$ | $P(A)P(A^C)$ |
    | Geometric | $p$ | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ |
- Expectation: If $X$ is a discrete random variable that takes values in a set $S$, its expectation, $E[X]$, is defined as:
    
    $$
    E[X]=\sum_{x\in S}xP(X=x)
    $$
    
    - The expectation is a numerical measure that summarizes the typical, or average, behavior of a random variable. Expectation is a weighted average of the values of $X$, where the weights are the corresponding probabilities of those values. The expectation places more weight on values that have greater probability.
    - The expectation of a discrete uniform distribution. Let $X\sim Unif\{1, \dots,n\}$. The expectation of $X$ is:
    
    $$
    E[X]=
    \sum_{x=1}^{n}xP(X=x)=
    \sum_{x=1}^{n}\frac{x}{n}=
    \frac{1}{n}\frac{(n+1)n}{2}=
    \frac{n+1}{2}
    $$
    
    - The expectation of a Poisson distribution. Let $X\sim Pois(\lambda)$. The expectation of $X$ is:
    
    $$
    E[X]=
    \sum_{k=0}^{\infty}kP(X=k)=
    \sum_{k=0}^{\infty}\frac{e^{-\lambda} \lambda^k}{k!}
    \newline
    =e^{-\lambda}\sum_{k=1}^{\infty}\frac{\lambda^k}{(k-1)!}=
    \lambda e^{-\lambda}\sum_{k=1}^{\infty}
    \frac{\lambda^{k-1}}{(k-1)!}
    \newline
    =\lambda e^{-\lambda}\sum_{k=0}^{\infty} \frac{\lambda^k}{k!} = \lambda e^{-\lambda}e^{\lambda} = \lambda
    $$
    
- Functions of random variables. Suppose $X$ is a random variable and $g$ is some function. Then, $Y=g(X)$ is a random variable that is a function of $X$. The values of this new random variable are found as follows. If $X=x$, then $Y=g(x)$.
    - Expectation of function of a random variable. Let $X$ be a random variable that takes values in a set $S$. Let $g$ be a function. Then,
    
    $$
    E[g(X)]=
    \sum_{x\in S}g(x)P(X=x)
    $$
    
    - Expectation of a linear function of $X$. For constants $a$ and $b$, and a random variable $X$ with expectation $E[X]$,
    
    $$
    E[aX+b]=aE[X]+b
    $$
    
    Let $g(X)=ax+b$. By the law of the unconscious statistician,
    
    $$
    E[aX+b]=\sum_x (ax+b)P(X=x)
    \newline
    = a\sum_xxP(X=x)+b\sum_x P(X=x)
    \newline
    =aE[X]+b
    $$
    
    - Except for the case above, $E[g(X)]\neq g(E[X])$
- Joint Distributions:
    - In the case of two random variables $X$ and $Y$, a joint distribution specifies the values and probabilities for all pairs of outcomes. For the two discrete variables joing pmf of $X$ and $Y$ is the function of two variables $P(X=x, Y=y)$
        
        $$
        \sum_{x\in S} \sum_{y\in T}P(X=x, Y=y)=1
        $$
        
    - Marginal Distributions: If $X$ takes values in a set $S$, and $Y$ takes values in a set $T$, then the marginal distribution of $X$ is
        
        $$
        P(X=x)=\sum_{y\in T}P(X=x, Y=y)
        $$
        
        and the marginal distribution of $Y$ is
        
        $$
        P(Y=y)=\sum_{x\in S}P(X=x, Y=y)
        $$
        
    - Expectation of function of two random variables:
        
        $$
        E[g(X,Y)]=\sum_{x \in S} \sum_{y\in T}
        g(x, y) P(X=x, Y=y)
        $$
        
- Independent Random Variables: If $X$ and $Y$ are independent random variables, then the joint pmf of $X$ and $Y$ has a particularly simple form. In the case of independence,
    
    $$
    P(X=x, Y=y)=P(X=x)P(Y=y), \; for \; all \; x \; and \; y.
    $$
    
    - Expectation of a product of independent random variables: Let $X$ and $Y$ be independent random variables. Then for any functions $g$ and $h$,
        
        $$
        E[g(X)h(Y)]=E[g(X)]E[h(Y)]
        $$
        
        Letting $g$ and $h$ be the identity function gives the useful result that 
        
        $$
        E[XY]=E[X]E[Y]
        $$
        
    - Sums of independent random variables: To find probabilities of the for $P(X+Y=k)$, observe that $X+Y=k$ if and only if $X=i$ and $Y=k-i$ for some $i$. This gives:
        
        $$
        P(X+Y=k)=\sum_i P(X=i, Y=k-i)=\sum_i P(X=i)P(Y=k-i)
        $$
        
- Linearity of Expectation: For random variables $X$ and $Y$,
    
    $$
    E[X+Y]=E[X]+E[Y]
    $$
    
- Indicator Variables: Given an event $A$, define a random variable $I_A$, such that
    
    $$
    I_A=
    \begin{cases}
    1, & if \;A \;occurs, \\
    0, & if \; A \;doesn't \;occur
    \end{cases}
    $$
    
    Therefore, $I_A$ equals 1, with probability $P(A)$, and 0, wit hprobabilit y$P(A^c)$. Such a random variable is known as an indicator variable. An indicator is a Bernoulli random variable with $p=P(A)$.
    
- Variance and Standard Deviation: Variance and SD are measures of variability or spread. They describe how near or far typical outcomes are to the expected value (mean).
    
    Let $X$ be a random variable with mean $E[X]=\mu<\infty$. The variance of $X$ is:
    
    $$
    V[X]=E[(X-\mu)^2]=\sum_x (x-\mu)^2P(X=x)
    $$
    
    The standard deviation of $X$ is
    
    $$
    SD[X]=\sqrt{V[X]}
    $$
    
    - Computational formula for variance:
        
        $$
        V[X]=E[X^2]-E[X]^2
        $$
        
    - Properties of Expectation, Variance, and Standard Deviation: Let $X$ be a random variable, where $E[X]$ and $V[X]$ exist. For constants $a$ and $b$,
        
        $$
        E[aX+b]=aE[X]+b, 
        \newline
        V[aX+b]=a^2V[X], \; 
        \newline
        SD[aX+b]=|a|SD[X]
        $$
        
    - Variance of the sum of independent variables: If $X$ and $Y$ are independent, then
        
        $$
        V[X+Y]=V[X]+V[Y]
        $$
        
    - General formula for variance of a sum: For random variables $X$ and $Y$ with finite variance,
        
        $$
        V[X+Y]=V[X]+V[Y]+2Cov(X,Y)
        \newline
        V[X-Y]=V[X]+V[Y]-2Cov(X,Y)
        $$
        
- Covariance and Correlation: Having looked at measures of variability for individual and independent random variables, we now consider measures of variability between dependent random variables. The covariance is a measure of the association between two random variables.
    - Covariance: For random variables $X$ and $Y$, with respective means $\mu_X$ and $\mu_Y$, the covariance between $X$ and $Y$ is
        
        $$
        Cov(X, Y)=E[(X-\mu_X)(Y-\mu_Y)]
        $$
        
        Equivalently, an often more usable computational formula is
        
        $$
        Cov(X, Y)=E[XY]-\mu_X\mu_Y=
        E[XY]-E[X]E[Y]
        $$
        
        Covariance will be positive when large values of $X$ are associated with large values of $Y$ and small values of $X$ are associated with small values of  $Y$. On the other hand, if $X$ are $Y$ are inversely related, most product terms will be negative, as when $X$ takes values above the mean, $Y$ will tend to fall below the mean, and vice versa.
        
        Covariance is a measure of linear association between two variables. In a sense, the “less linear” the relationship, the closer the covariance is to 0.
        
        The sign of the covariance indicates whether two random variables are positively or negatively associated. But the magnitude of the covariance can be difficult to interpret due to the scales of the original variables. The correlation is an alternative measure which is easier to interpret.
        
    - Correlation: The Correlation between $X$ and $Y$ is
        
        $$
        Corr(X, Y)=\frac{Cov(X, Y)}{SD[X]SD[Y]}
        $$
        
        Properties of correlation:
        
        1. $-1 \leq Corr(X,Y) \leq 1$
        2. If $Y=aX+b$ is a linear function of $X$ for constants $a$ and $b$, then $Corr(X,Y)= \pm1$, depending on the sign of $a$.
        
        Dividing the covariance by the standard deviation creates a “standardized” covariance, which is a unitless measure that takes values between -1 and 1.
        
    - Uncorrelated random variables: We say random variables $X$ and $Y$ are uncorrelated if
        
        $$
        E[XY]=E[X]E[Y]
        $$
        
        that is, if $Cov(X,Y)=0$ (i.e. $X$ and $Y$ are independent).
        
- Conditional Distribution: If $X$ and $Y$ are jointly distributed discrete random variables, then the conditional probability mass function of $Y$ given $X=x$ is
    
    $$
    P(Y=y|X=x)=\frac{P(X=x,Y=y)}{P(X=x)}
    $$
    
- Conditional Expectation: For discrete random variables $X$ and $Y$, the conditional expectation of $Y$ given $X=x$ is
    
    $$
    E[Y|X=x]=\sum_y yP(Y=y|X=x)
    $$
    
- Properties of Covariance and Correlation:
    
    <aside>
    💡
    
    Some cool-ass properties and proofs shown here
    
    </aside>
    
    - Covariance Property - Linearity: For random variables $X$, $Y$, and $Z$, and constants $a,b, c$,
        
        $$
        Cov(aX+bY+c,Z)=aCov(X,Z)+bCov(Y,Z)
        \newline
        Cov(X,aY+bZ+c)=aCov(X,Y)+bCov(X,Z)
        $$
        
        Given a random variable with mean $\mu$ and variance $\sigma^2$, the standardized variable $X^*$ is defined as
        
        $$
        X^*=\frac{X-\mu}{\sigma}
        $$
        
        Observe that
        
        $$
        E[X^*]=E[\frac{X-\mu}{\sigma}]=
        \frac{1}{\sigma}(E[X]-\mu)=
        \frac{1}{\sigma}(\mu-\mu)=0
        $$
        
        $$
        V[X^*]=V[\frac{X-\mu}{\sigma}]=
        \frac{1}{\sigma^2}(V[X-\mu])=
        \frac{\sigma^2}{\sigma^2}=1
        $$
        
    - Correlation Results: For random variables $X$ and $Y$,
        
        $$
        -1\leq Corr(X,Y)\leq 1
        $$
        
        If $Corr(X,Y)= \pm 1$, then there exists constants $a \neq 0$ and $b$ such that $Y=aX+b$.
        
        *Proof:* Given $X$ and $Y$, let $X^*$ and $Y^*$ be the standardized variables. Observe that
        
        $$
        Cov(X^*,Y^*)
        =Cov(\frac{X-\mu_X}{\sigma_X},\frac{Y-\mu_Y}{\sigma_Y})=
        \frac{1}{\sigma_X \sigma_Y}Cov(X,Y)=
        Corr(X,Y)
        $$
        
        Consider the variance of $X^*\pm Y^*$:
        
        $$
        V[X^*+Y^*]=V(X^*)+V(Y^*)+2Cov(X^*,Y^*)
        \newline
        =2+2Corr(X,Y)
        $$
        
        similarly,
        
        $$
        V[X^*-Y^*]=V(X^*)+V(Y^*)-2Cov(X^*,Y^*)
        \newline
        =2-2Corr(X,Y)
        $$
        
        This gives,
        
        $$
        Corr(X,Y)=\frac{V(X^*+Y^*)}{2}-1\geq-1
        $$
        
        and
        
        $$
        Corr(X,Y)=-\frac{V(X^*-Y^*)}{2}+1\leq-1
        $$
        
        because the variance is nonnegative. That is, $-1\leq Corr(X,Y)\leq1$
        

## More Discrete Distributions

- Geometric Distribution: There are many questions one can ask about an underlying sequence of Bernoulli trials. The binomial distribution describes the number of successes in $n$ trials. The geometric distribution describes the number of trials until the first success occurs. To find the pmf of $X$, observe that $X=k$ if success occurs on the $k$th trial and the first $k-1$ trials are failures. This occurs with probability
 $(1-p)^{k-1}p$.
    - The random variable $X$ has a geometric distribution with parameter $p$ if
        
        $$
        P(X=k)=(1-p)^{k-1}p, \; for\; k=1,2,\dots
        $$
        
    - Tail Probability: If $X \sim Geom(p)$, then for $k>0$,
        
        $$
        P(X>k)=(1-p)^k
        $$
        
- Memorylessness: The geometric distribution has a unique property among discrete distributions - it is what is called memoryless.
    
    A random variable $X$ has the memorylessness property if for all $0<s<t$,
    
    $$
    P(X>t|X>s)=P(X>t-s)
    $$
    
    See example on page 188 for context about traffic violations
    
- Moment-Generating Functions: Some expectations have special names. For $k=1,2,\dots,$ the *kth moment* of a random variable $X$ is $E[X^k]$. For instance, the first moment of $X$ is the expectation $E[X]$. The moment-generating function (mgf), as the name suggests, can be used to generate the moments of a random variable. Mgfs are also useful for demonstrating some relationships between random variables.
    - MGF: Let $X$ be a random variable. The *mgf*  of $X$ is the real-valued function
        
        $$
        m(t)=E[e^{tX}]
        $$
        
        defined for all real $t$ when this expectation exists. Also written as $m_X(t)$.
        
    - Geometric distribution: Let $X\sim Geom(p)$. The mgf of $X$ is
        
        $$
        m(t)=E[e^{tX}]=\sum_{k=1}^\infty e^{tk}(1-p)^{k-1}p
        \newline
        =pe^t\sum_{k=1}^\infty(e^t(1-p))^{k-1}
        =\frac{pe^t}{1-e^t(1-p)}
        $$
        
        How do we get moments from the mfg? Moments of $X$ are obtained from the mgf by successively differentiating $m(t)$ and evaluating at $t=0$. We have
        
        $$
        m\prime(t)=\frac{d}{dt}E[e^{tX}]=E[\frac{d}{dt}e^{tX}]=E[Xe^{tX}]
        $$
        
        and $m\prime(0)=E[X]$.
        
        Taking the second derivative gives
        
        $$
        m\prime \prime(t)=\frac{d}{dt}m\prime (t)
        = \frac{d}{dt}E[Xe^{tX}]=E[\frac{d}{dt}Xe^{tX}]=E[X^2e^{tX}]
        $$
        
        and $m\prime \prime(0)=E[X^2]$
        
        In general the *kth* derivative of the mgf evaluated at $t=0$ gives the *kth* moment as 
        
        $$
        m^{(k)}(0)=E[X^k],\; for\; k=1,2,\dots
        $$
        
- Properties of MGFs:
    1. If $X$ and $Y$ are independent random variables, then the mgf of their sum is the product of their mgfs. That is
        
        $$
        M_{X+Y}(t)=E[e^{t(X+Y)}]
        \newline
        =E[e^{tX}e^{tY}]=E[e^{tX}]E[e^{tY}]
        \newline
        =m_X(t)m_Y(t)
        $$
        
    2. Let $X$ be a random variable with mgf $m_X(t)$ and constants $a\neq0$ and $b$. Then
        
        $$
        m_{aX+b}(t)=E[e^{t(aX+b)}]=e^{bt}E[e^{(ta)X}]=e^{bt}m_X(at)
        $$
        
    3. MGFs uniquely determine the underlying probability distribution. That is, if two random variables have the same MGF, then they have the same probability distribution.
    
    <aside>
    💡
    
    Moments and Moment-Generating Functions are defined here.
    
    </aside>
    
- Negative Binomial: The geometric distribution counts the number of trials until the first success occurs in i.i.d. Bernoulli trials. The negative binomial distribution extends this, counting the number of trials until the *r*th success occurs.
    - Negative Binomial Distribution: A random variable $X$ has the negative binomial distribution parameter $r$ and $p$ if
        
        $$
        P(X=k)= \binom{k-1}{r-1}p^r(1-p)^{k-r}
        \newline 
        \;r=1,2,\dots, \;k=r,r+1,\dots
        $$
        
        We erite $X\sim NegBin(r,p)$
        
- Hypergeometric - Sampling without Replacement: Whereas the binomial distribution arises from sampling with replacement, the hypergeometric distribution often arises when sampling is without replacement from a finite population.
    - Hypergeometric Distrbution: A random varable $X$ has a hypergeometric distribution with parameters $r$, $N$, and $n$ if
        
        $$
        P(X=k)=\frac
        {\binom{r}{k} \binom{N-r}{n-k}}
        {\binom{N}{n}}
        $$
        
        for $max(0, n-(N-r))\leq k\leq min(n,r)$. The values of $k$ are restricted by the domain of the binomial coefficients as $0\leq k \leq r$ and $0\leq n-k \leq N-r$.
        
        We write $X \sim HyperGeo(R, N, n).$
        
- From Binomial to Multinomial: In a binomial setting, successive trials take one of two possible values (e.g., success or failure). The multinomial distribution is a generalization of the binomial distribution which arises when successive independent trials can take more than two values. The multinomial distribution is used to model such things as follows:
    1. The number of ones, twos, threes, fours, fives, and sizes in 25 dice rolls.
    2. The frequencies of $r$ different alleles among $n$ individuals.
    3. The number of outcoes of an experiment that has $m$ possible results when repeated $n$ times.
    4. The frequencies of size different colors in a sample of 10 candies.
    
    - Multinomial Distribution: Suppose $p_1,\dots,p_r$ are nonnegative numbers such that $p_1+\cdots +p_r=1$. Random variables $X_1,\dots,X_r$ have a multinomial distribution with parameters $n,p_1,\dots,p_r$ if
        
        $$
        P(X_1=x_1,\dots,X_r=x_r)=
        \frac{n!}{x_1!\cdots x_r!}{p_1}^{x_1}\cdots {p_r}^{x_r}
        $$
        
        for nonnegative interegers $x_1,\dots,x_r$ such that $x_1+\cdots +x_r=n$.
        
        We write $(X_1\dots,X_r) \sim Multin(n,p_1,\dots ,p_r)$
        
- Benford’s Law: Pick a random book in your backpack. Open up a random page.Let your eyes fall on a random number in the middle of the page. Write down the number and circle the first digit, ignoring zeros. You’ll notice that ones are the most common (~30%), twos are less common, threes and even less common, etc. The formula is described as
    
    $$
    P(d)=log_{10}(\frac{d+1}{d})
    $$
    

## Continuous Probability

- Continuous Random Variable: A random variable which takes values in a continuous set.
- Probability Density Function: A function $f$ is the density function of a continuous random variable if
    1. $f(x) \ge 0$ for all $x$
    2. $\int_{-\infty}^\infty f(x)dx=1$.
    3. For all $S \subseteq ℝ $, $P(X\in S)=\int_S f(x)dx$
- Cumulative Distribution Function: The CDF of $X$ is $F(x)=P(X\leq x)$, defined for all $x$.
- PDF and CDF: $F\prime(x)=f(x)$
- Properties of CDF:
    1. $\lim_{x\to \infty} F(X)=1$.
    2. $\lim_{x\to -\infty} F(X)=0$.
    3. $F(X)$  is right-continuous at all $x$.
    4. $F(X)$ is an increasing function of $x$.
- Expectation: $E[X]=\int_{-\infty}^\infty xf(x)dx$.
- Variance: $V[X]=\int_{-\infty}^\infty (x-E[X])^2f(x)dx$.
- Law of the Unconscious Statistician: If $g$ is a function, then
    
    $$
    E[g(X)]=\int_{-\infty}^\infty g(x)f(x)dx
    $$
    
- Uniform Distribution: The uniform distribution arises as a model for equally likely outcomes. Properties of the continuous uniform distribution include:
    1. $E[X]=(b+a)/2$
    2. $V[X]=(b-a)^2/12$
    3. $F(x)=P(X\leq x)=(x-a)/(b-a),$ if $a< x < b, \; 0$, if $x\leq a$, and $1$, if  $x \ge b$.
- Exponential Distribution: The distribution of $X$ is exponential with parameter $\lambda > 0$ if the density of $X$ is $f(x)=\lambda e^{-\lambda e}$, for all $x>0$.
- Exponential Setting: The exponential distribution is often used to model arrival times - the time until some event occurs, such as phone calls, traffic accidents, component failures, etc. Properties of the exponential distribution include:
    1. $E[X]=1/\lambda$
    2. $V[X]=1/\lambda^2$
    3. $F(x)=P(X\leq x)=1-e^{-\lambda x}$
    4. The exponential distribution is the only continuous distribution which is memoryless.
- Joint Probability Density Function: For jointly continuous random variables the joint density $f(x, y)$ has similar properties as the univariate density function:
    1. $f(x,y) \geq 0,$ for all $x$ and $y$.
    2. $\int_{-\infty}^\infty  \int_{-\infty}^\infty  f(x,y)  = 1$.
    3. For all $S \subseteq ℝ^2, \; P((X,Y) \in S)= {\int \int}_S f(x,y)dxdy$
- Joint Cumulative Distribution Function: $F(x,y)=P(X \leq x, Y\leq y)$, defined for all real $x$ and $y$.
- Joint CDF and Joint PDF: $\frac{d^2}{dx dy}F(x,y)=f(x,y)$
- Expectation of function of two random variables: If $g(x,y)$ is a function of two variables, then $E[g(X,Y)]=\int_{-\infty}^\infty \int_{-\infty}^\infty  g(x,y)f(x,y)dx\;dy$
- Independence: If $X$ and $Y$ are jointly continuous and independent, with marginal densities $f_X$ and $f_Y$, respectively, then th ejoint density of $X$ and $Y$ is $f(x,y)=f_X(x)f_Y(y)$.
- Accept-Reject Method: Suppose $S$ is a bounded set in the plane. The method gives a way to simulate from the unform distribution on $S$. Enclose $S$ ina. rectangle $R$. Generate a point unformly distributed in $R$. If the point is in $S$, ‘accept’; if the point is not in $S$, ‘reject’ and try again. The first accepted point will be uniformly distributed on $S$.
- Covariance:
    
    $$
    Cov(X,Y)=E[(X-E[X])(Y-E[Y])]
    \newline
    =\int_{-\infty}^\infty \int_{-\infty}^\infty (ex-E[X])(y-E[Y])f(x,y)dx\;dy
    $$
    

## Continuous Distribution

- Normal Distribution: A random variable $X$ has the normal distribution with parameters $\mu$ and $\sigma^2$, if the density function of $X$ is
    
    $$
    f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}},\;-\infty<x<\infty
    $$
    
    We write $X \sim Norm(\mu, \sigma^2)$ or $N(\mu,\sigma^2)$
    
- MGF of the Normal Distribution: If $X\sim Norm(\mu, \sigma^2)$, then the MGF of $X$ is
    
    $$
    m(t)=E[e^{tX}]=e^{\mu t+\sigma^2 t^2/2}
    $$
    
- Gamma Distribution: The gamma distribution is a family of positive, continuous distributions with two parameters. The density curve can take a wide variety of shapes, which allows the distribution to be used to model variables that exhibit skewed and non-symmetric behavior.
    - A random variable $X$ has a gama distribution with parameters $a>0$ and $\lambda >0$ if the density function $X$ is
        
        $$
        f(x)=\frac{\lambda^a x^{a-1} e^{-\lambda x}}{\Gamma(a)}, \; for \; x>0
        $$
        
        where 
        
        $$
        \Gamma(a)=\int_0^\infty t^{a-1}e^{-t}dt
        $$
        
        We write $X\sim Gamma(a, \lambda)$. The function $\Gamma$ is the gamma function, which is continuous, defined by an integral and arises in many applied settings.
        
    - In many applications, exponential random variables are used to model inter-arrival times between events, such as the times between successive highway accidents, component failures, telephone calls, or bus arrivals. The nth occurrence, or time of the nth arrival, is the sum of n inter-arrival times. It turns out that the sum of n i.i.d. exponential random variables has a gamma distribution.
    - Sum of i.i.d. Exponentials: Let $E_1,\dots,E_n$ be an i.i.d. sequence of exponential random variables with parameter $\lambda$. Let $S=E_1+\dots+E_n$. Then $S$ has a gamma distribution with parameters $n$ and $\lambda$.
- Poisson Distribution: Consider a process whereby “events” — also called “points” or “arrivals” — occur randomly in time or space. Examples include phone calls throughout the day, car accidents along a stretch of highway, component failures, service times, and radioactive particle emissions.
    
    For many applications, it is reasonable to model the times between successive events as memoryless (e.g., a phone call doesn’t “remember” when the last phone call took place). Model these interarrival times as an independent sequence $E_1,E_2,\dots$ of exponential random variables with parameter $\lambda$, where $E_k$, is the time between the $(k-1)$st and kth arrival. Set $S_0=0$ and let 
    
    $$
    S_n=E_1+\cdots + E_n
    $$
    
    for $n=1,2,\dots$ Then, $S_n$ is the time of the nth arrival and $S_n \sim Gamma(n, \lambda)$. The sequence $S_0,S_1,S_2,\dots$ is the arrival sequence, i.e., the sequence of arrival times.
    
    For each time $t\geq 0$, ket $N_t$ be the number of arrivals that occur up through time $t.$ Then for each $t,N_t$ is a discrete random variable. We show that $N_t$ has a Poisson distribution. The collection of $N_t$s forms a random process called a Poisson Process with parameter $\lambda$. It is an example of what is called a stochastic process. Formally, a stochastic process is a collection of random variables defined on a common sample space.
    
- Notation:
    - $E_k$ - inter-arrival time ($E_1,E_2,\dots \sim Exp(\lambda)$)
    - $S_n$ - arrival time ($S_n \sim Gamma(n, \lambda)$)
    - $N_t$ number of arrivals ($N_t \sim Pois(\lambda t)$)
- Distribution of $N_t$: Let $(N_t)_{t\geq0}$ be a Poisson process with parameter $\lambda$. Then
    
    $$
    P(N_t=k)=\frac{e^{-\lambda t} (\lambda t)^k}{k!},\; for \; k=0,1,\dots
    $$
    

## Limits

- Law of Large Numbers — Let $X_1,X_2,\dots$ be an independent and identically distributed sequence of random variables with finite expectation $\mu$. For $n=1,2,\dots,$ let
    
    $$
    S_n=X_1+\cdots+X_n
    $$
    
     The law says that $S_n/n \to \mu = P(A)$, as $n\to \infty$. That is, the proportion of $n$ trials in which $A$ occurs converges to $P(A)$.
    
- Weak Law of Large Numbers: WLLN says that for any $\epsilon > 0$ the sequence of probabilities
    
    $$
    P(|\frac{S_n}{n} -\mu| < \epsilon) \to 1, \;as\; n\to \infty
    $$
    
    That is, the probability that $S_n/n$ is arbitrarily close to $\mu$ converges to $1$.
    
    Similarly,
    
    $$
    P(|\frac{S_n}{n} -\mu| \geq \epsilon) \to 0, \; as\; n\to \infty
    $$
    
    “As you increase the number of trials to infinity, an estimate approaches its parameter.”
    
- Markov and Chebyshev Inequalities — Bernoulli’s original proof of the weak law of large numbers is fairly complicated and technical. A much simpler proof was discovered mid-1800s based on what is now called Chebyshev’s inequality by way of Markov’s.
    - Markov’s Inequality — Let $X$ be a nonnegative random variable with finite expectation. Then for all $\epsilon > 0$,
        
        $$
        P(X \geq \epsilon) \leq \frac{E[X]}{\epsilon}
        $$
        
        *Proof*: When $X$ is continuous with density function $f$, we have:
        
        $$
        E[X]=\int_0^{\infty} xf(x)dx \geq \int_\epsilon^\infty xf(x)dx \geq 
        \int_\epsilon^\infty \epsilon f(x)dx= \epsilon P(X\geq \epsilon)
        $$
        
        *Example*: Let $\epsilon = kE[X]=k\mu$ in Markov’s inequality for positive integer $k$. Then
        
        $$
        P(X \geq k\mu)\leq \frac{\mu}{k\mu}= \frac{1}{k}
        $$
        
        For instance, the probability that a nonnegative random variable is at least twice its mean is at most $1/2$.
        
        **Corollary**: If $g$ is an increasing positive function, then
        
        $$
        P(X \geq \epsilon)= P(g(X) \geq g(\epsilon)) \leq 
        \frac{E[g(X)]}{g(\epsilon)}
        $$
        
        By careful choice of the function $g$, one can often improve teh Markov inequality upper bound.
        
    - Chebyshev’s Inequality — Let $X$ be a random variable (not necessarily positive) with finite mean $\mu$ and variance $\sigma^2$. Then for all $\epsilon > 0$,
        
        $$
        P(|X-\mu|\geq \epsilon)\leq \frac{\sigma^2}{\epsilon^2}
        $$
        
        *Proof*: Let $g(x)=x^2$ on $(0, \infty)$. By our Corollary, applied to the nonnegative random variable $\mid X-\mu\mid,$
        
        $$
        P(|X-\mu|\geq \epsilon)=P(|X-\mu|^2 \geq \epsilon^2) \leq 
        \frac{E[(X-\mu)^2]}{\epsilon^2}=\frac{\sigma^2}{\epsilon^2}
        $$
        
        At times, it may make sense to consider an equivalent expression using our understanding of complements, giving
        
        $$
        P(|X-\mu| < \epsilon)>1 - \frac{\sigma^2}{\epsilon^2}
        $$
        
        *Example*: Let $X$ be an exponential random variable with mean and variance equal to $1$. Consider $P(X \geq 4)$. By Markov’s inequalitty,
        
        $$
        P(X \geq 4) \leq \frac{1}{4}=0.25
        $$
        
        To bound $P(X\geq4)$ using Chebyshev’s inequality, we have 
        
        $$
        P(X\ge 4)=P(X-1 \ge 3)=P(|X-1| \ge 3) \le \frac{1}{9}=0.111
        $$
        
        We see the improvement of Chebyshev’s bound over Markov’s bound.
        
        In fact, $P(X\ge4)=e^{-4}=0.0183$. So both bounds are fairly crude. However, the power of Markov’s an dChebyshev’s inequalities is that they apply without regard to the distribution of the random variable, so long as their requirements are satisfied. 
        
- Strong Law of Large Numbers — Let $X_1,X_2,\dots$ be an i.i.d. sequence of random variables with finite mean $\mu$. For $n=1,2,\dots,$ let $S_n=X_1+\cdots+X_n$. Then
    
    $$
    P(\lim_{n \to \infty} \frac{S_n}{n} = \mu ) =1
    $$
    
    We say that $S_n/n$ converges to $\mu$ with probability $1$.
    
- Method of Moments — The method of moments is a statistical technique for using data to estimate the unknown parameters of a probability distribution (similar to the maximum likelihood approach).
    
    Recall that the kth moment of a random variable $X$ is $E[X^k]$. We will also call this the kth theoretical moment.
    
    Let $X_1,\dots,X_n$ be an i.i.d. sample from a probability distribution with finite moments. Think if $X_1,\dots,X_n$ as representing data from a random sample. The kth sample moment is defined as
    
    $$
    \frac{1}{n} \sum_{i=1}^{n}X^{k}_{i}
    $$
    
    In a typical statistical context, the values of the $X_i$s are known (they are observation values in the data set), and the parameters of the underlying probability distribution are unknown.
    
    In the method of moments, one sets up equations that equate sample moments with corresponding theoretical moments. The equations are solved for the unknown parameters of interest. The method is reasonable because if $X$ is a random variable from the probability distribution of interest then by the SSLN, with probability 1, 
    
    $$
    \frac{1}{n}\sum_{i=1}^{n}X^{k}_{i} \to E[X^k], \; as \; n\to \infty
    $$
    
    and thus for large $n$,
    
    $$
    \frac{1}{n}\sum_{i=1}^{n}X^{k}_{i} \approx E[X^k]
    $$
    
- Central Limit Theorem — This theorem rivals the Law of large Numbers in importance. It geives insight into the behavior of sums of random variables, it is fundamental to much of statistical inference, and it quantifies the size of the error in using Monte Carlo methods to approximate integrals, expectations, and probabilities.
    - Let $X_1,X_2,\dots$ be an i.i.d sequence of random variables with finite mean $\mu$ and variance $\sigma^2$. For $n=1,2,\dots,$ let $S_n=X_1+\cdots+X_n$. Then the distribution of the standardized random variabeles $(S_n/n -\mu)/(\sigma /\sqrt{n})$ converges to a standard normal distribution in the following sense. For all $t$,
        
        $$
        P(\frac{S_n/n-\mu}{\sigma/\sqrt{n}} \leq t) \to P(Z \leq t), \; as \; n \to \infty
        $$
        
        Where $Z \sim Norm(0,1)$.
        
        The specialness of the CLT is that it applies to any distribution of the $X_i$’s with finite mean and variance.
        
    - Equivalent Expressions for the CLT — If the sequence of random variables $X_1,X_2,\dots$  satisfies the assumption of the CLT, then for large $n$,
        
        $$
        X_1+\cdots+X_n \approx Norm(n\mu,n\sigma^2)
        $$
        
        $$
        \overline{X}_n=\frac{X_1+\cdots+X_n}{n} \approx Norm(\mu,\sigma^2/n)
        $$
        
    - Random Walks — A particle starts at the origin on the integer number line. At each step, the particle moves left or right with probability $1/2$. Find the expectation and standard deviation of the distance of the walk from the origin after $n$ steps.
        
        A random walk process is constructed as follows. Let $X_1,X_2,\dots$ be an independent sequence of random variables taking values $\pm1$ with probability $1/2$ each. The $X_i$’s represent the individual steps of the random walk. For $n=1,2,\dots$, let $S_n=X_1+\cdots X_n$ be the position of the walk after $n$ steps. The random walk process is the sequence $(S_1,S_2,S_3,\dots)$. 
        
        The $X_i$’s have mean $0$ and variance $1$. Thus, $E[S_n]=0$ and $V[S_n]=1$. By the CLT, the for large $n$ the distribution of $S_n$ is approximately normal with mean $0$ and variance $n$.
        
        After $n$ steps, the random walk’s distance from the origin is $\mid S_n\mid $. Using the normal approximation, the expected distrance from the origin is
        
        $$
        E[\mid S_n\mid] \approx \int_{-\infty}^\infty |t| \frac{1}{\sqrt{2\pi n}}e^{t^2 /2n}dt=
        \frac{2}{\sqrt{2\pi n}} \int_0 ^\infty te^{-t^2/2n}dt= 
        \frac{2}{\sqrt{2\pi n}} =
        \sqrt{\frac{2}{\pi}}\sqrt{n} \approx (0.80)\sqrt{n}
        $$
        
        For the standard deviation of distance, $E[\mid S_n\mid^2]=E[S_{n}^2]=n$. Thus,
        
        $$
        V[|S_n|]=E[|S_{n}^2|]-E[|S_n|]^2 \approx n- \frac{2n}{\pi}=n(\frac{\pi-2}{\pi})
        $$
        
        giving 
        
        $$
        SD[|S_n|] \approx \sqrt{\frac{\pi-2}{\pi}}=
        n(\frac{\pi-2}{\pi})
        $$
        
    - In broad strokes, the law of large numbers asserts that $S_n/n \approx \mu$, when $n$ is large. The CLT states that for large $n$,
     $(S_n/n-\mu)/(\sigma/\sqrt{n})\approx Z$, where $Z$ is a standard normal random variable.

<aside>
💡

Stuff on Markov/Chebyshev inequalities, Law of Large Numbers, CLT and Random Walks.

</aside>

## Random Walks and Markov Chains

Random walks on graphs are a special case of Markov chains. A Markov chain is a sequence of random variables $X_0,X_1,X_2,\dots$, with the property that for all $n$, the conditional distribution of $X_{n+1}$ given the past history $X_0,\dots,X_n$ is equal to the conditional distribution of $X_{n+1}$ given $X_n$. This is sometimes stated as the *distribution of the future given the past only depends on the present*. The set of values of the Markov chain is called the *state space*.

A simple random walk on a graph is a Markov chain because the distribution of the walk’s position at any fixed time only depends on the last vertex visited and not on the previous locations of the walk. The state space is the vertex set of the graph, leading to a discrete state space. Extensions exist to continuous state spaces.

Markov chains are remarkably useful models. They are used extensively in virtually every applied field to model random processes that exhibit some dependency structure between successive outcomes.

- Page Rank — When you make an inquiry using Google, it returns an ordered list of sites by assigning a rank to each page. The rank it assigns is essentially the limiting distribution of a Markov chain.
    
    This Markov chain can be described as a random walk on the web graph. In the web graph, vertices represent web pages, edges represent hyperlinks. A directed edge joins page $i$ to page $j$. Imagine a random walker that visits web pages according to this model moving from a page with probability proportional to the number of “out-links” from that page. The long-term probability that the random walker is at page $i$ is precisely the PageRank of page $i$.
    
    We write $i{^\to}\sim j$ if there is a directed edge from $i$ to $j$. Let $link(i)$ be the number of directed edges from $i$. Transition probabilities are defined as follows:
    
    $$
    T_{ij}=
    \begin{cases}
    P(X_1=j|X_0=i)=1/link(i), && if \; i^\to \sim j \\
    0, && otherwise,
    \end{cases}
    $$