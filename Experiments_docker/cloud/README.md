# Documentation

1. `ComputePrivacy.py`

   The provided Python script is used to compute differential privacy bounds, also known as privacy epsilon. Differential privacy is a system for publicly sharing information about a dataset by describing the patterns of groups within the dataset while withholding information about individuals in the dataset. The term "epsilon" indicates the amount of privacy budget in the data. A lower epsilon implies more privacy and a higher epsilon implies less privacy.

   Let's break down the script.

   Import necessary libraries:

   - math: provides mathematical functions.
   - mpmath: provides arbitrary precision arithmetic.
   - numpy: provides support for large, multi-dimensional arrays and matrices, along with mathematical functions.
   - sys: provides access to some variables used or maintained by the Python interpreter and to functions that interact strongly with the interpreter.
   - The function ComputePrivacy() is the main function which takes several arguments: sample_ration, variance, iterations, delta, and max_order. This function calculates the log moments using the helper function compute_log_moment(), and then it uses these moments to compute the minimum epsilon, which is returned as the output.

   The function compute_log_moment() calculates the log moment for the given order. It defines Gaussian distributions, then defines a third lambda function (mu) as a weighted combination of the other two Gaussians. The function a_lambda_fn is used to calculate the integral of the distribution, then it is converted to a NumPy float, and the log of this moment is multiplied by the number of iterations.

   The function pdf_gauss_mp() calculates the Gaussian probability density function (pdf) given an input value x, standard deviation sigma, and mean mean. It uses the mpmath library for high precision calculations.

   The function `_to_np_float64()` is a utility function that converts its input to a NumPy float64. It checks whether the value is not a number (NaN) or infinity, and if it is, it returns infinity; otherwise, it converts the value to a NumPy float64.

   The function `_compute_eps()` takes as input the computed log moments and the delta, and calculates the minimum epsilon value, which represents the privacy loss parameter. It iterates over the log moments, and for each log moment, it computes an epsilon value, and keeps track of the smallest epsilon value seen.

   It's worth noting that a couple of commented lines in the function `_compute_eps()` seem to provide error checking functionality for specific corner cases, but these lines have been commented out in the provided code.

   The script seems to be part of a larger privacy-preserving data analysis system. It appears to be designed to work with Gaussian differential privacy, a particular method of differential privacy that provides stronger protection for outliers.

   ### **A further explaination about the `compute_log_moment` method:**

   The compute_log_moment function calculates a specific mathematical quantity called the "log moment" for a privacy-preserving mechanism in the context of differential privacy.

   Here is a more detailed explanation of each step in the function:

   Two Gaussian probability density functions (pdf) are defined - ${\mu_0}$ and ${\mu_1}$:
   ${\mu_0}$ is a Gaussian pdf centered around 0 with a standard deviation of variance.
   ${\mu_1}$ is a Gaussian pdf centered around 1 with a standard deviation of variance.
   These two functions define two different Gaussian distributions. The specific choice of these distributions (centered at 0 and 1) might be related to the specific privacy-preserving mechanism that this function is being used to analyze.

   A third lambda function $\mu$ is defined as a weighted combination of ${\mu_0}$ and ${\mu_1}$:
   This lambda function represents a mixed Gaussian distribution. The sample_ration parameter determines the weight given to ${\mu_1}$ relative to ${\mu_0}$. This is effectively creating a distribution that is somewhere between the two original Gaussian distributions, with the exact position determined by sample_ration.

   The function a_lambda_fn is defined:
   This lambda function is the one used to calculate the moment. It multiplies the mixed Gaussian pdf $\mu(z)$ by $(\mu(z) / {\mu_0}(z))^{order}$

   This value is used in calculating the moments of the distribution. Note that order specifies which moment is being calculated. The integral of this function over the entire real line is the desired moment of the mixed distribution.

   The moment is then calculated by integrating `a_lambda_fn` over the entire real line (from negative infinity to positive infinity). This is done using mp.quad, which is a function from the mpmath library that can compute numerical integrals.

   The moment is then converted to a numpy float using the `_to_np_float64` function, and the log of this moment is taken.

   The function finally returns the log of the moment multiplied by the number of iterations.

   In essence, this function is calculating the log of a specific moment of a mixed Gaussian distribution that is defined by two other Gaussian distributions. This information is crucial in the analysis of privacy-preserving mechanisms. The order of the moment being calculated, the variance of the two original Gaussian distributions, and the proportion in which these two distributions are mixed together can all affect the privacy properties of the mechanism.
