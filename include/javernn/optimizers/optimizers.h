#ifndef __JAVERNN_OPTIMIZER_H__
#define __JAVERNN_OPTIMIZER_H__

namespace javernn
{
/**
 * base class of Optimizer
 * usesHessian : true if an Optimizer uses hessian (2nd order derivative of loss
 *function)
 **/
class Optimizer {
  Optimizer()                  = default;
  Optimizer(const Optimizer &) = default;
  Optimizer(Optimizer &&)      = default;
  Optimizer &operator=(const Optimizer &) = default;
  Optimizer &operator=(Optimizer &&) = default;
  virtual ~Optimizer()               = default;
  virtual void update(const vec_t &dW, vec_t &W, bool parallelize) = 0;
  virtual void reset() {}  // override to implement pre-learning action
};

// helper class to hold N values for each weight
template <int N>
class StatefulOptimizer : public Optimizer {
  void reset() override {
    for (auto &e : E_) e.clear();
  }

 protected:
  template <int Index>
  vec_t &get(const vec_t &key) {
    static_assert(Index < N, "index out of range");
    if (E_[Index][&key].empty()) E_[Index][&key].resize(key.size(), float_t());
    return E_[Index][&key];
  }
  std::unordered_map<const vec_t *, vec_t> E_[N];
};

/**
 * adaptive gradient method
 *
 * J Duchi, E Hazan and Y Singer,
 * Adaptive subgradient methods for online learning and stochastic optimization
 * The Journal of Machine Learning Research, pages 2121-2159, 2011.
 **/
class Adagrad : public StatefulOptimizer<1> {
  Adagrad() : alpha(float_t(0.01)), eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &g = get<0>(W);
    for_i(parallelize, W.size(), [&](size_t i) {
      g[i] += dW[i] * dW[i];
      W[i] -= alpha * dW[i] / (std::sqrt(g[i]) + eps);
    });
  }

  float_t alpha;  // learning rate
 private:
  float_t eps;
};

/**
 * RMSprop
 *
 * T Tieleman, and G E Hinton,
 * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
 **/
class RMSprop : public StatefulOptimizer<1> {
  RMSprop() : alpha(float_t(0.0001)), mu(float_t(0.99)), eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &g = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
      W[i] -= alpha * dW[i] / std::sqrt(g[i] + eps);
    });
  }

  float_t alpha;  // learning rate
  float_t mu;     // decay term
 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * @brief [a new Optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
class Adam : public StatefulOptimizer<2> {
  adam()
    : alpha(float_t(0.001)),
      b1(float_t(0.9)),
      b2(float_t(0.999)),
      b1_t(float_t(0.9)),
      b2_t(float_t(0.999)),
      eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &mt = get<0>(W);
    vec_t &vt = get<1>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      vt[i] = b2 * vt[i] + (float_t(1) - b2) * dW[i] * dW[i];

      // L2 norm based update rule
      W[i] -= alpha * (mt[i] / (float_t(1) - b1_t)) /
              std::sqrt((vt[i] / (float_t(1) - b2_t)) + eps);
    });

    b1_t *= b1;
    b2_t *= b2;
  }

  float_t alpha;  // learning rate
  float_t b1;     // decay term
  float_t b2;     // decay term
  float_t b1_t;   // decay term power t
  float_t b2_t;   // decay term power t

 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * @brief [a new Optimizer (2015)]
 * @details [see Adam: A Method for Stochastic Optimization (Algorithm 2)
 *               http://arxiv.org/abs/1412.6980]
 *
 */
class Adamax : public StatefulOptimizer<2> {
  Adamax()
    : alpha(float_t(0.002)),
      b1(float_t(0.9)),
      b2(float_t(0.999)),
      b1_t(b1),
      eps(float_t(1e-8)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &mt = get<0>(W);
    vec_t &ut = get<1>(W);

    for_i(parallelize, W.size(), [&](int i) {
      mt[i] = b1 * mt[i] + (float_t(1) - b1) * dW[i];
      ut[i] = std::max(b2 * ut[i], std::abs(dW[i]));

      // Lp norm based update rule
      W[i] -= (alpha / (1.0 - b1_t)) * (mt[i] / (ut[i] + eps));
    });

    b1_t *= b1;
  }

  float_t alpha;  // learning rate
  float_t b1;     // decay term
  float_t b2;     // decay term
  float_t b1_t;   // decay term power t

 private:
  float_t eps;  // constant value to avoid zero-division
};

/**
 * SGD without momentum
 *
 * slightly faster than tiny_dnn::momentum
 **/
class GradientGescent : public Optimizer {
  GradientGescent() : alpha(float_t(0.01)), lambda(float_t(0)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    for_i(parallelize, W.size(),
          [&](size_t i) { W[i] = W[i] - alpha * (dW[i] + lambda * W[i]); });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
};

/**
 * SGD with momentum
 *
 * B T Polyak,
 * Some methods of speeding up the convergence of iteration methods
 * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
 **/
class Momentum : public StatefulOptimizer<1> {
 public:
  Momentum() : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += V;
      dWprev[i] = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

/**
 * SGD with Nesterov momentum
 *
 * Y Nesterov,
 * A method for unconstrained convex minimization problem with the rate of
 * convergence o(1/k2), Doklady ANSSSR, vol.269, pp.543-547, 1983.
 **/
class NesterovMomentum : public StatefulOptimizer<1> {
 public:
  NesterovMomentum()
    : alpha(float_t(0.01)), lambda(float_t(0)), mu(float_t(0.9)) {}

  void update(const vec_t &dW, vec_t &W, bool parallelize) {
    vec_t &dWprev = get<0>(W);

    for_i(parallelize, W.size(), [&](size_t i) {
      float_t V = mu * dWprev[i] - alpha * (dW[i] + W[i] * lambda);
      W[i] += (-mu) * dWprev[i] + (1 + mu) * V;
      dWprev[i] = V;
    });
  }

  float_t alpha;   // learning rate
  float_t lambda;  // weight decay
  float_t mu;      // momentum
};

}

#endif