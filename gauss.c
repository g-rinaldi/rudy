/**
 * @file gauss.c
 * @brief Implements a function to generate standard normally distributed random numbers.
 * @author Giovanni Rinaldi
 *
 * Description:
 * This file provides the `gauss()` function, which generates pseudo-random numbers
 * following a standard normal (Gaussian) distribution (mean 0, variance 1).
 * It uses the Box-Muller transform, specifically the polar method, for efficiency,
 * avoiding trigonometric calculations.
 *
 * The function relies on the `gb_next_rand()` function from the Stanford GraphBase's
 * gb_flip module (`gb_flip.h`) as its source of uniform random integers in the
 * range [0, 0x7fffffff].
 *
 * It maintains static state (`saved_deviate` and `has_saved_deviate`) to generate
 * pairs of Gaussian deviates per Box-Muller iteration and return one per call,
 * improving performance by amortizing the generation cost over two calls.
 */

#include <math.h>      // For sqrt() and log()
#include "gb_flip.h"   // For gb_next_rand()

// Define the maximum value returned by gb_next_rand() as a double for normalization.
// gb_next_rand() returns values in [0, 0x7fffffff].
#define MAX_INT_AS_DOUBLE ((double)0x7fffffff)

/**
 * @brief Generates a pseudo-random number from a standard normal distribution.
 * @details Uses the polar Box-Muller transform method. It generates pairs of
 *          independent standard normal random numbers, returning one and saving
 *          the other for the next call. This function depends on `gb_next_rand()`
 *          from SGB's gb_flip module having been initialized (e.g., via `gb_init_rand()`).
 * @return A double-precision floating-point number sampled from N(0, 1).
 */
double gauss() {
  // Static variables to store the second deviate and the state flag.
  // Note: Static variables are initialized only once.
  static double saved_deviate = 0.0;
  static int has_saved_deviate = 0; // Flag: 0 = need new pair, 1 = return saved deviate

  // If we have a saved deviate from the previous call, return it.
  if (has_saved_deviate) {
    has_saved_deviate = 0; // Clear the flag for the next call
    return saved_deviate;
  } else {
    // Generate a new pair of standard normal deviates using the polar Box-Muller method.
    double u1, u2, v1, v2, s;
    double multiplier; // To store the result of sqrt/log calculation

    // Generate two uniform random numbers in (0, 1] and map them to v1, v2 in [-1, 1].
    // Continue generating until we find a point (v1, v2) inside the unit circle
    // (but not exactly at the origin, though probability is negligible).
    do {
      // Generate uniform random numbers in [0, 1] using gb_next_rand().
      // Note: Division by MAX_INT_AS_DOUBLE maps [0, 0x7fffffff] to [0, 1].
      u1 = (double)gb_next_rand() / MAX_INT_AS_DOUBLE;
      u2 = (double)gb_next_rand() / MAX_INT_AS_DOUBLE;

      // Transform u1, u2 from [0, 1] to v1, v2 in [-1, 1]
      v1 = 2.0 * u1 - 1.0;
      v2 = 2.0 * u2 - 1.0;

      // Calculate the squared radius
      s = v1 * v1 + v2 * v2;

      // Reject points outside or exactly on the unit circle boundary, and the origin.
      // s == 0.0 check added for robustness against potential floating point issues
      // or the extremely unlikely event of u1=u2=0.5 exactly.
    } while (s >= 1.0 || s == 0.0);

    // Calculate the Box-Muller transformation factor once.
    multiplier = sqrt((-2.0 * log(s)) / s);

    // Generate the two standard normal deviates.
    // Save the second deviate (v2 * multiplier) for the next call.
    saved_deviate = v2 * multiplier;
    has_saved_deviate = 1; // Set the flag

    // Return the first deviate (v1 * multiplier).
    return v1 * multiplier;
  }
}
