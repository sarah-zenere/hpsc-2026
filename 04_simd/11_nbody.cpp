#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>
//editing raw again because github won't push
//i love my computer
//at least i can access the lab server and TSUBAME now

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for (int i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0.0f;
  }

  //forces w AVX-512
  for (int i = 0; i < N; i++) {

    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 fx_sum = _mm512_setzero_ps();
    __m512 fy_sum = _mm512_setzero_ps();

    for (int j = 0; j < N; j += 16) {

      //particles
      __m512 xj = _mm512_loadu_ps(x + j);
      __m512 yj = _mm512_loadu_ps(y + j);
      __m512 mj = _mm512_loadu_ps(m + j);

      //displacement
      __m512 rx = _mm512_sub_ps(xi, xj);
      __m512 ry = _mm512_sub_ps(yi, yj);

      //ehhhh
      __m512 r2 = _mm512_add_ps(
        _mm512_mul_ps(rx, rx),
        _mm512_mul_ps(ry, ry)
      );
      __mmask16 mask = 0xFFFF;
      if (i >= j && i < j + 16) {
        mask &= ~(1 << (i - j));
      }

      __m512 safe_r2 = _mm512_mask_blend_ps(
        mask,
        _mm512_set1_ps(1.0f),  //dummy
        r2
      );

      __m512 r = _mm512_sqrt_ps(safe_r2);
      __m512 r3 = _mm512_mul_ps(r, safe_r2);


      __m512 inv = _mm512_div_ps(mj, r3);

      __m512 fx_contrib = _mm512_mul_ps(rx, inv);
      __m512 fy_contrib = _mm512_mul_ps(ry, inv);

      fx_contrib = _mm512_maskz_mov_ps(mask, fx_contrib);
      fy_contrib = _mm512_maskz_mov_ps(mask, fy_contrib);

      fx_sum = _mm512_sub_ps(fx_sum, fx_contrib);
      fy_sum = _mm512_sub_ps(fy_sum, fy_contrib);
    }

    //vector to scalar
    float fx_tmp[16], fy_tmp[16];
    _mm512_storeu_ps(fx_tmp, fx_sum);
    _mm512_storeu_ps(fy_tmp, fy_sum);

    float fx_total = 0.0f, fy_total = 0.0f;
    for (int k = 0; k < 16; k++) {
      fx_total += fx_tmp[k];
      fy_total += fy_tmp[k];
    }

    printf("%d %g %g\n", i, fx_total, fy_total);
  }

  return 0;
}
