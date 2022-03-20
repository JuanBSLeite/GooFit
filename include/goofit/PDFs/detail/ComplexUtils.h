#pragma once

#include <goofit/Error.h>
#include <goofit/detail/Complex.h>
#include <vector>

namespace GooFit {


/// Flatten a complex array into a standard one (1r, 1i, 2r, 2i, ...)
template <typename T>
std::vector<T> flatten(const std::vector<thrust::complex<T>> &input) {
    std::vector<T> output;
    for(auto val : input) {
        output.push_back(val.real());
        output.push_back(val.imag());
    }
    return output;
}

std::vector<fpcomplex> complex_derivative(const std::vector<fptype> &x, const std::vector<fpcomplex> &y) {
    if(x.size() != y.size()) // Must be a valid pointer
        throw GeneralError("x and y must have the same diminsions!");

    int i, k;
    unsigned int n = x.size();
    std::vector<fpcomplex> u(n);
    std::vector<fpcomplex> y2(n);

    fptype sig, p, qn, un;
    fpcomplex yp1 = 2. * (y[1] - y[0]) / (x[1] - x[0]);
    fpcomplex ypn = 2. * (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);

    /* The lower boundary condition is set either to be "natural" or else to have specified first derivative*/
    if(yp1.real() > 0.99e30) {
        y2[0].real(0.);
        u[0].real(0.);
    } else {
        y2[0].real(-0.5);
        u[0].real(3.0 / (x[1] - x[0]) * ((y[1].real() - y[0].real()) / (x[1] - x[0]) - yp1.real()));
    }
    if(yp1.imag() > 0.99e30) {
        y2[0].imag(0.);
        u[0].imag(0.);
    } else {
        y2[0].imag(-0.5);
        u[0].imag(3.0 / (x[1] - x[0]) * ((y[1].imag() - y[0].imag()) / (x[1] - x[0]) - yp1.imag()));
    }

    /* This is the decomposition loop of the tridiagonal algorithm. y2 and u are used for temporary storage of the
     * decomposed factors*/

    for(i = 1; i < n - 1; i++) {
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
        p   = sig * y2[i - 1].real() + 2.0;
        y2[i].real((sig - 1.0) / p);
        u[i].real((y[i + 1].real() - y[i].real()) / (x[i + 1] - x[i])
                  - (y[i].real() - y[i - 1].real()) / (x[i] - x[i - 1]));
        u[i].real((6.0 * u[i].real() / (x[i + 1] - x[i - 1]) - sig * u[i - 1].real()) / p);
        p = sig * y2[i - 1].imag() + 2.0;
        y2[i].imag((sig - 1.0) / p);
        u[i].imag((y[i + 1].imag() - y[i].imag()) / (x[i + 1] - x[i])
                  - (y[i].imag() - y[i - 1].imag()) / (x[i] - x[i - 1]));
        u[i].imag((6.0 * u[i].imag() / (x[i + 1] - x[i - 1]) - sig * u[i - 1].imag()) / p);
    }

    /* The upper boundary condition is set either to be "natural" or else to have specified first derivative*/

    if(ypn.real() > 0.99e30) {
        qn = 0.;
        un = 0.;
    } else {
        qn = 0.5;
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn.real() - (y[n - 1].real() - y[n - 2].real()) / (x[n - 1] - x[n - 2]));
    }
    y2[n - 1].real((un - qn * u[n - 2].real()) / (qn * y2[n - 2].real() + 1.0));
    if(ypn.imag() > 0.99e30) {
        qn = 0.;
        un = 0.;
    } else {
        qn = 0.5;
        un = (3.0 / (x[n - 1] - x[n - 2])) * (ypn.imag() - (y[n - 1].imag() - y[n - 2].imag()) / (x[n - 1] - x[n - 2]));
    }
    y2[n - 1].imag((un - qn * u[n - 2].imag()) / (qn * y2[n - 2].imag() + 1.0));

    /* This is the backsubstitution loop of the tridiagonal algorithm */

    for(k = n - 2; k >= 0; k--) {
        y2[k].real(y2[k].real() * y2[k + 1].real() + u[k].real());
        y2[k].imag(y2[k].imag() * y2[k + 1].imag() + u[k].imag());
    }

    return y2;
}

#define M_2PI 6.28318530717958
//#define ROOT2 1.41421356

// tables for Pade approximation
__constant__ fptype _C[7]
    = {65536.0, -2885792.0, 69973904.0, -791494704.0, 8962513560.0, -32794651890.0, 175685635125.0};
__constant__ fptype _D[7]
    = {192192.0, 8648640.0, 183783600.0, 2329725600.0, 18332414100.0, 84329104860.0, 175685635125.0};

//#define UNROLL_LOOP 1

#ifndef UNROLL_LOOP
__constant__ fptype _n1[12] = {0.25, 1.0, 2.25, 4.0, 6.25, 9.0, 12.25, 16.0, 20.25, 25.0, 30.25, 36.0};
__constant__ fptype _e1[12] = {0.7788007830714049,
                              0.3678794411714423,
                              1.053992245618643e-1,
                              1.831563888873418e-2,
                              1.930454136227709e-3,
                              1.234098040866795e-4,
                              4.785117392129009e-6,
                              1.125351747192591e-7,
                              1.605228055185612e-9,
                              1.388794386496402e-11,
                              7.287724095819692e-14,
                              2.319522830243569e-16};

// table 2: coefficients for h = 0.53
__constant__ fptype _n2[12]
    = {0.2809, 1.1236, 2.5281, 4.4944, 7.0225, 10.1124, 13.7641, 17.9776, 22.7529, 28.09, 33.9889, 40.4496};
__constant__ fptype _e2[12] = {0.7551038420890235,
                              0.3251072991205958,
                              7.981051630007964e-2,
                              1.117138143353082e-2,
                              0.891593719995219e-3,
                              4.057331392320188e-5,
                              1.052755021528803e-6,
                              1.557498087816203e-8,
                              1.313835773243312e-10,
                              6.319285885175346e-13,
                              1.733038792213266e-15,
                              2.709954036083074e-18};

__device__ fpcomplex device_Faddeeva(const fpcomplex &z) {
    fptype *n, *e, t, u, r, s, d, f, g, h;
    fpcomplex c, d2, v;
    int i;

    s = thrust::norm(z); // NB: norm2 is correct although CPU version calls the function 'norm'.

    if(s < 1e-7) {
        // use Pade approximation
        fpcomplex zz = z * z;
        v = exp(zz); // Note lower-case! This is our own already-templated exp function for thrust::complex, no need for
                     // float/double define.
        c  = _C[0];
        d2 = _D[0];

        for(i = 1; i <= 6; i++) {
            c  = c * zz + _C[i];
            d2 = d2 * zz + _D[i];
        }

        return fptype(1.0) / v + fpcomplex(0.0, M_2_SQRTPI) * c / d2 * z * v;
    }

    // use trapezoid rule
    // select default table 1
    n = _n1;
    e = _e1;
    r = M_1_PI * 0.5;

    // if z is too close to a pole select table 2
    if(fabs(z.imag()) < 0.01 && fabs(z.real()) < 6.01) {
        // h = modf(2*fabs(z.real),&g);
        // Equivalent to above. Do this way because nvcc only knows about double version of modf.
        h = fabs(z.real()) * 2;
        g = floor(h);
        h -= g;

        if(h < 0.02 || h > 0.98) {
            n = _n2;
            e = _e2;
            r = M_1_PI * 0.53;
        }
    }

    d = (z.imag() - z.real()) * (z.imag() + z.real());
    f = 4 * z.real() * z.real() * z.imag() * z.imag();

    g = h = 0.0;

    for(i = 0; i < 12; i++) {
        t = d + n[i];
        u = e[i] / (t * t + f);
        g += (s + n[i]) * u;
        h += (s - n[i]) * u;
    }

    u = 1 / s;

    c = r * fpcomplex(z.imag() * (u + 2.0 * g), z.real() * (u + 2.0 * h));

    if(z.imag() < M_2PI) {
        s = 2.0 / r;
        t = s * z.real();
        u = s * z.imag();
        s = sin(t);
        h = cos(t);
        f = exp(-u) - h;
        g = 2.0 * exp(d - u) / (s * s + f * f);
        u = 2.0 * z.real() * z.imag();
        h = cos(u);
        t = sin(u);
        c += g * fpcomplex((h * f - t * s), -(h * s + t * f));
    }

    return c;
}

#else
__device__ fpcomplex device_Faddeeva(const fpcomplex &z) {
    fptype u, s, d, f, g, h;
    fpcomplex c, d2, v;

    s = norm2(z); // NB: norm2 is correct although CPU version calls the function 'norm'.

    if(s < 1e-7) {
        // use Pade approximation
        fpcomplex zz = z * z;
        v = exp(zz); // Note lower-case! This is our own already-templated exp function for thrust::complex, no need for
                     // float/double define.
        c  = _C[0];
        d2 = _D[0];

        for(int i = 1; i < 7; ++i) {
            c  = c * zz + _C[i];
            d2 = d2 * zz + _D[i];
        }

        return fptype(1.0) / v + fpcomplex(0.0, M_2_SQRTPI) * c / d2 * z * v;
    }

    // use trapezoid rule
    fptype r        = M_1_PI * 0.5;
    bool useDefault = true;

    // if z is too close to a pole select table 2
    if(fabs(z.imag) < 0.01 && fabs(z.real) < 6.01) {
        // h = modf(2*fabs(z.real),&g);
        // Equivalent to above. Do this way because nvcc only knows about double version of modf.
        h = fabs(z.real) * 2;
        g = floor(h);
        h -= g;

        if(h < 0.02 || h > 0.98) {
            useDefault = false;
            r          = M_1_PI * 0.53;
        }
    }

    d = (z.imag - z.real) * (z.imag + z.real);
    f = 4 * z.real * z.real * z.imag * z.imag;

    g = h           = 0.0;
    fptype currentN = (useDefault ? 0.25 : 0.2809);
    fptype currentE = (useDefault ? 0.7788007830714049 : 0.7551038420890235);
    fptype t        = d + currentN;
    u               = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 1.0 : 1.1236);
    currentE = (useDefault ? 0.3678794411714423 : 0.3251072991205958);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 2.25 : 2.5281);
    currentE = (useDefault ? 1.053992245618643e-1 : 7.981051630007964e-2);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 4.0 : 4.4944);
    currentE = (useDefault ? 1.930454136227709e-3 : 0.891593719995219e-3);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 6.25 : 7.0225);
    currentE = (useDefault ? 4.785117392129009e-6 : 1.052755021528803e-6);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 9.0 : 10.1124);
    currentE = (useDefault ? 1.605228055185612e-9 : 1.313835773243312e-10);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 12.25 : 13.7641);
    currentE = (useDefault ? 7.287724095819692e-14 : 1.733038792213266e-15);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 16.0 : 17.9776);
    currentE = (useDefault ? 1.831563888873418e-2 : 1.117138143353082e-2);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 20.25 : 22.7529);
    currentE = (useDefault ? 1.234098040866795e-4 : 4.057331392320188e-5);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 25.0 : 28.09);
    currentE = (useDefault ? 1.125351747192591e-7 : 1.557498087816203e-8);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 30.25 : 33.9889);
    currentE = (useDefault ? 1.388794386496402e-11 : 6.319285885175346e-13);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    currentN = (useDefault ? 36.0 : 40.4496);
    currentE = (useDefault ? 2.319522830243569e-16 : 2.709954036083074e-18);
    t        = d + currentN;
    u        = currentE / (t * t + f);
    g += (s + currentN) * u;
    h += (s - currentN) * u;

    u = 1 / s;
    c = r * fpcomplex(z.imag * (u + 2.0 * g), z.real * (u + 2.0 * h));

    if(z.imag < M_2PI) {
        s = 2.0 / r;
        t = s * z.real;
        u = s * z.imag;
        s = sin(t);
        h = cos(t);
        f = exp(-u) - h;
        g = 2.0 * exp(d - u) / (s * s + f * f);
        u = 2.0 * z.real * z.imag;
        h = cos(u);
        t = sin(u);
        c += g * fpcomplex((h * f - t * s), -(h * s + t * f));
    }

    return c;
}
#endif


} // namespace GooFit