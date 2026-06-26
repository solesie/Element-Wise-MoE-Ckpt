# Mixed-Age Rollback Ideal Objective

이 문서는 mixed-age checkpoint 상태에서 fault 후 rollback loss를 정확히 정의하고,
현재 구현의 2차 score가 어디에서 나오는지 수학적으로 고정한다.

수식은 Markdown math renderer 기준으로 inline math는 `$...$`, block math는 `$$...$$`
형식을 사용한다.

## 1. State And Rollback Error

전체 파라미터 수를 $d$라 하고, 현재 학습 중인 모델 파라미터를 다음 열벡터로 둔다.

$$
\theta \in \mathbb{R}^{d}
$$

SSD에 저장된 mixed-age checkpoint 값을 다음과 같이 둔다.

$$
c \in \mathbb{R}^{d}
$$

원소 $i$의 현재값을 이번 checkpoint에서 저장했는지를 나타내는 이진 선택 벡터를
다음과 같이 둔다.

$$
z \in \{0,1\}^{d}
$$

그리고 선택 행렬을 다음처럼 정의한다.

$$
Z = \operatorname{Diag}(z)
$$

여기서 $z_i=1$이면 원소 $i$는 현재값 $\theta_i$로 저장되어 fault 후 rollback error가
없고, $z_i=0$이면 SSD의 오래된 값 $c_i$로 돌아간다.

따라서 fault 후 롤백되는 양은 다음이다.

$$
e(z) = (I-Z)(\theta-c)
$$

복구된 모델은 다음과 같다.

$$
\theta_{\mathrm{rb}} = \theta - e(z)
$$

동치로 다음처럼 쓸 수도 있다.

$$
\theta_{\mathrm{rb}} = Z\theta + (I-Z)c
$$

정확한 rollback loss 증가는 다음이다.

$$
J(z) = L(\theta - e(z)) - L(\theta)
$$

여기까지는 근사가 없다.

## 2. Fixing One Selection

특정 저장 선택 $z$를 하나 고정하면 $e=e(z)$도 고정된다. 이때 1변수 함수
$\phi$를 다음처럼 정의한다.

$$
\phi(u) = L(\theta - ue), \qquad u \in [0,1]
$$

그러면 다음이 성립한다.

$$
\phi(0)=L(\theta), \qquad \phi(1)=L(\theta-e)
$$

따라서 rollback loss는 다음과 같다.

$$
J(z)=\phi(1)-\phi(0)
$$

## 3. Derivatives Along The Rollback Path

현재 gradient를 다음과 같이 둔다.

$$
g = \nabla L(\theta) \in \mathbb{R}^{d}
$$

연쇄법칙에 의해

$$
\phi'(u) = -\nabla L(\theta-ue)^{T}e
$$

이고, 따라서

$$
\phi'(0) = -g^{T}e
$$

이다.

현재 Hessian을 다음과 같이 둔다.

$$
H(\theta)=\nabla^{2}L(\theta) \in \mathbb{R}^{d\times d}
$$

그러면

$$
\phi''(u) = e^{T}H(\theta-ue)e
$$

이고,

$$
\phi''(0) = e^{T}H(\theta)e
$$

이다.

손실의 3차 미분 텐서를 $\nabla^{3}L(x)$라고 쓰면, 세 벡터 $e,e,e$를 넣은 값은
다음 스칼라다.

$$
\nabla^{3}L(x)[e,e,e]
=
\sum_{i,j,k=1}^{d}
\frac{\partial^{3}L(x)}
{\partial \theta_i \, \partial \theta_j \, \partial \theta_k}
e_i e_j e_k
$$

경로가 $\theta-ue$ 방향으로 움직이므로 3차 미분은 다음과 같다.

$$
\phi'''(u)
=
-\nabla^{3}L(\theta-ue)[e,e,e]
$$

## 4. Exact Taylor Formula

$\phi$가 세 번 연속 미분 가능하면, 어떤 $\xi_z \in (0,1)$가 존재하여
Lagrange 나머지항 형태의 Taylor 정리가 정확히 성립한다.

$$
\phi(1)
=
\phi(0)
+\phi'(0)
+\frac{1}{2}\phi''(0)
+\frac{1}{6}\phi'''(\xi_z)
$$

따라서 다음 식이 정확히 성립한다.

$$
J(z)
=
-g^{T}e(z)
+\frac{1}{2}e(z)^{T}H(\theta)e(z)
-\frac{1}{6}
\nabla^{3}L(\theta-\xi_z e(z))[e(z),e(z),e(z)]
$$

즉,

$$
\boxed{
J(z)
=
-g^{T}e(z)
+\frac{1}{2}e(z)^{T}H(\theta)e(z)
+R_3(e(z))
}
$$

여기서 나머지항은 다음이다.

$$
R_3(e(z))
=
-\frac{1}{6}
\nabla^{3}L(\theta-\xi_z e(z))[e(z),e(z),e(z)]
$$

이 식은 근사가 아니다.

## 5. Meaning Of The Remainder

$R_3(e)$는 단순히 현재점 $\theta$에서의 3차항 하나만 뜻하지 않는다.
3차 미분이 중간점 $\theta-\xi_z e$에서 평가되기 때문에, 현재점 기준으로 다시 전개하면
4차, 5차 이상의 영향까지 포함한다.

손실함수가 해석적이고 Taylor 급수가 수렴한다고 가정하면 익숙한 무한급수 형태는 다음이다.

$$
\begin{aligned}
J(z)
=&
-\nabla L(\theta)^{T}e
+\frac{1}{2!}\nabla^{2}L(\theta)[e,e] \\
&-\frac{1}{3!}\nabla^{3}L(\theta)[e,e,e]
+\frac{1}{4!}\nabla^{4}L(\theta)[e,e,e,e]
-\cdots
\end{aligned}
$$

하지만 일반적인 엄밀한 표현은 위의 Lagrange 나머지항이다.

## 6. The Quadratic Ideal Score

2차 근사로 넘어가는 정확한 순간은 $R_3(e)$를 버리는 지점이다.

$$
J(z)
=
-g^{T}e(z)
+\frac{1}{2}e(z)^{T}H(\theta)e(z)
+O(\|e(z)\|^{3})
$$

따라서 2차 ideal score는 다음이다.

$$
\boxed{
\operatorname{score}_{\mathrm{ideal}}(z)
=
-g^{T}e(z)
+\frac{1}{2}e(z)^{T}H(\theta)e(z)
}
$$

이 근사는 다음 판단을 포함한다.

$$
R_3(e(z)) \text{가 저장 선택의 ranking을 바꿀 만큼 크지 않다.}
$$

이 판단이 성립할 때만 다음처럼 볼 수 있다.

$$
J(z) \approx \operatorname{score}_{\mathrm{ideal}}(z)
$$

## 7. Optimization Problem

저장 예산이 $K$개 원소라고 하면 정확한 저장 선택 문제는 다음이다.

$$
\min_{z \in \{0,1\}^{d}}
J(z)
\quad
\text{s.t.}
\quad
\mathbf{1}^{T}z \le K
$$

동치로, 현재값을 저장해서 rollback loss를 가장 많이 줄이는 subset을 고르는 문제다.

일반적인 $L$에 대해서는 $J(z)$ 자체가 비선형 binary objective다. 2차 근사만 보더라도
다음과 같다.

$$
-g^{T}(I-Z)(\theta-c)
+\frac{1}{2}
(\theta-c)^{T}(I-Z)H(\theta)(I-Z)(\theta-c)
$$

Hessian의 off-diagonal coupling 때문에 선택 변수들 사이의 상호작용이 생긴다.
즉 일반형은 binary quadratic optimization/subset selection을 포함하고, 예산 제약까지
붙으면 NP-hard다.

중요한 예외는 Hessian을 diagonal 또는 element-wise proxy로 근사하는 경우다. 이때는
objective가 원소별 additive score로 분해되고, top-$K$ 또는 single-threshold selection이
가능해진다. 현재 proposal score는 이 일반 NP-hard objective를 직접 푸는 것이 아니라,
element-wise Taylor proxy로 바꾼 뒤 단일 $K$-budget 경쟁을 수행하는 근사다.
