"""Generated numerical kernel; regenerate with scripts/build_fast_solver.py.
Derived from starmodel.py (STATSTAR; Schiminovich/Nidever). No fastmath.
"""
import numpy as np
from numba import njit

@njit(cache=True, error_model='numpy')
def startmdl(deltar, X, Z, mu, Rs, r_i, M_ri, L_ri, tog_bf, irc):
    """
    Computes values of M_r, L_r, P, and T, near the surface of the
    star using expansions of the stellar structure equations (M_r and
    L_r are assumed to be constant).
  
    X = hydrogen mass fraction
    Z = metal mass fraction
    mu = mean molecular weight
    Rs = radius of star
    r_i = radius of shell
    M_ri = mass enclosed within shell at this radius 
          (assumed constant at start of model)
    L_ri = enclosed luminosity, assumed constant
    """
    r = r_i + deltar
    M_rip1 = M_ri
    L_rip1 = L_ri
    if irc == 0:
        T_ip1 = 6.67259e-08 * M_rip1 * mu * 1.673534e-24 / (4.25 * 1.380658e-16) * (1.0 / r - 1.0 / Rs)
        A_bf = 4.34e+25 * Z * (1.0 + X) / tog_bf
        A_ff = 3.68e+22 * 1.0 * (1.0 - Z) * (1.0 + X)
        Afac = A_bf + A_ff
        P_ip1 = np.sqrt(1.0 / 4.25 * (16.0 / 3.0 * np.pi * 7.56591e-15 * 29979245800.0) * (6.67259e-08 * M_rip1 / L_rip1) * (1.380658e-16 / (Afac * mu * 1.673534e-24))) * T_ip1 ** 4.25
    else:
        T_ip1 = 6.67259e-08 * M_rip1 * mu * 1.673534e-24 / 1.380658e-16 * (1.0 / r - 1.0 / Rs) / 2.5
        P_ip1 = 0.3 * T_ip1 ** 2.5
    return (r, P_ip1, M_rip1, L_rip1, T_ip1)

@njit(cache=True, error_model='numpy')
def eos(X, Z, XCNO, mu, P, T, izone):
    """
    Equation-of-State eos() calculates the values of density, opacity, the 
    guillotine-to-gaunt factor ratio, and the energy generation rate at
    the radius r.
    """
    if T < 0.0 or P < 0.0:
        return (0.0, 0.0, 0.0, 0.0, 1)
    Prad = 7.56591e-15 * T ** 4 / 3.0
    Pgas = P - Prad
    rho = mu * 1.673534e-24 / 1.380658e-16 * (Pgas / T)
    if rho < 0.0:
        return (rho, 0.0, 0.0, 0.0, 1)
    tog_bf = 2.82 * (rho * (1.0 + X)) ** 0.2
    k_bf = 4.34e+25 / tog_bf * Z * (1.0 + X) * rho / T ** 3.5
    k_ff = 3.68e+22 * 1.0 * (1.0 - Z) * (1.0 + X) * rho / T ** 3.5
    k_e = 0.2 * (1.0 + X)
    kappa = k_bf + k_ff + k_e
    oneo3 = 0.333333333
    twoo3 = 0.666666667
    T6 = T * 1e-06
    fx = 0.133 * X * np.sqrt((3.0 + X) * rho) / T6 ** 1.5
    fpp = 1.0 + fx * X
    psipp = 1.0 + 141200000.0 * (1.0 / X - 1.0) * np.exp(-49.98 * T6 ** (-1.0 * oneo3))
    Cpp = 1.0 + 0.0123 * T6 ** oneo3 + 0.0109 * T6 ** twoo3 + 0.000938 * T6
    epspp = 2380000.0 * rho * X * X * fpp * psipp * Cpp * T6 ** (-twoo3) * np.exp(-33.8 * T6 ** (-oneo3))
    CCNO = 1.0 + 0.0027 * T6 ** oneo3 - 0.00778 * T6 ** twoo3 - 0.000149 * T6
    epsCNO = 8.67e+27 * rho * X * XCNO * CCNO * T6 ** (-twoo3) * np.exp(-152.28 * T6 ** (-oneo3))
    epslon = epspp + epsCNO
    return (rho, kappa, epslon, tog_bf, 0)

@njit(cache=True, error_model='numpy')
def dPdr(r, M_r, rho):
    return -6.67259e-08 * rho * M_r / r ** 2

@njit(cache=True, error_model='numpy')
def dMdr(r, rho):
    return 4 * np.pi * rho * r ** 2

@njit(cache=True, error_model='numpy')
def dLdr(r, rho, epslon):
    return 4 * np.pi * rho * epslon * r ** 2

@njit(cache=True, error_model='numpy')
def dTdr(r, M_r, L_r, T, rho, kappa, mu, irc):
    if irc == 0:
        return -(3 / (16 * np.pi * 7.56591e-15 * 29979245800.0)) * kappa * rho / T ** 3 * L_r / r ** 2
    else:
        return -1 / 2.5 * 6.67259e-08 * M_r / r ** 2 * mu * 1.673534e-24 / 1.380658e-16

@njit(cache=True, error_model='numpy')
def runge(f_im1, dfdr, r_im1, deltar, irc, X, Z, XCNO, mu, izone):
    """
    Runge-kutta algorithm
    """
    f_temp = np.zeros(4)
    f_i = np.zeros(4)
    dr12 = deltar / 2.0
    dr16 = deltar / 6.0
    r12 = r_im1 + dr12
    r_i = r_im1 + deltar
    for i in range(4):
        f_temp[i] = f_im1[i] + dr12 * dfdr[i]
    df1, ierr = fundeq(r12, f_temp, irc, X, Z, XCNO, mu, izone)
    if ierr != 0:
        return (f_i, ierr)
    for i in range(4):
        f_temp[i] = f_im1[i] + dr12 * df1[i]
    df2, ierr = fundeq(r12, f_temp, irc, X, Z, XCNO, mu, izone)
    if ierr != 0:
        return (f_i, ierr)
    for i in range(4):
        f_temp[i] = f_im1[i] + deltar * df2[i]
    df3, ierr = fundeq(r_i, f_temp, irc, X, Z, XCNO, mu, izone)
    if ierr != 0:
        return (f_i, ierr)
    for i in range(4):
        f_i[i] = f_im1[i] + dr16 * (dfdr[i] + 2.0 * df1[i] + 2.0 * df2[i] + df3[i])
    return (f_i, 0)

@njit(cache=True, error_model='numpy')
def fundeq(r, f, irc, X, Z, XCNO, mu, izone):
    """
    This function returns the required derivatives for runge(), the
    Runge-Kutta integration routine.

    fundeg(r, f, irc, X, Z, XCNO, mu, izone)
    """
    dfdr = np.zeros(4)
    P = f[0]
    M_r = f[1]
    L_r = f[2]
    T = f[3]
    rho, kappa, epslon, tog_bf, ierr = eos(X, Z, XCNO, mu, P, T, izone)
    dfdr[0] = dPdr(r, M_r, rho)
    dfdr[1] = dMdr(r, rho)
    dfdr[2] = dLdr(r, rho, epslon)
    dfdr[3] = dTdr(r, M_r, L_r, T, rho, kappa, mu, irc)
    return (dfdr, ierr)

@njit(cache=True, error_model='numpy')
def integrate(Msolar, Lsolar, Te, X, Z):
    istop = 0
    ip1 = 0
    '\n    Main program for calculating stellar structure\n\n    Variables, run-time parameters and settings:\n\n    deltar = radius integration step\n    idrflg = set size flag\n           = 0 (initial surface step size of Rs/1000.)\n           = 1 (standard step size of Rs/1000.)\n           = 2 (core step size of Rs/5000.)\n  \n    Nstart = number of steps for which starting equations are to be used\n             (the outermost zone is assumed to be radiative)\n    Nstop = maximum number of allowed zones in the star\n    Igoof = final model condition flag\n          = -1 (number of zones exceeded; also the initial value)\n          =  0 (good model)\n          =  1 (core density was extreme)\n          =  2 (core luminosity was extreme)\n          =  3 (extrapolated core temperature is too low)\n          =  4 (mass became negative before center was reached)\n          =  5 (luminosity became negative before center was reached)\n    X, Y, Z = mass fractions of hydrogen, helium, and metals\n    T0, P0 = surface temperature and pressure (T0 = P0 = 0 is assumed)\n    Ms, Ls, Rs = mass, luminosity, and radius of the star (cgs units)\n    '
    nsh = 5000
    r = np.zeros(nsh, float)
    P = np.zeros(nsh, float)
    M_r = np.zeros(nsh, float)
    L_r = np.zeros(nsh, float)
    T = np.zeros(nsh, float)
    rho = np.zeros(nsh, float)
    kappa = np.zeros(nsh, float)
    epslon = np.zeros(nsh, float)
    tog_bf = np.zeros(nsh, float)
    dlPdlT = np.zeros(nsh, float)
    deltar = 0.0
    XCNO = 0.0
    mu = 0.0
    Ms = 0.0
    Ls = 0.0
    Rs = 0.0
    T0 = 0.0
    P0 = 0.0
    Pcore = 0.0
    Tcore = 0.0
    rhocor = 0.0
    epscor = 0.0
    rhomax = 0.0
    Rsolar = 0.0
    Y = 1.0 - (X + Z)
    tog_bf0 = 0.01
    f_im1 = np.zeros(4, float)
    dfdr = np.zeros(4, float)
    f_i = np.zeros(4, float)
    Nstart = 10
    Nstop = 5000
    Igoof = -1
    ierr = 0
    P0 = 0.0
    T0 = 0.0
    dlPlim = 99.9
    debug = 0
    XCNO = Z / 2.0
    Ms = Msolar * 1.989e+33
    Ls = Lsolar * 3.826e+33
    Rs = np.sqrt(Ls / (4.0 * np.pi * 5.67051e-05)) / Te ** 2
    Rsolar = Rs / 69599000000.0
    deltar = -Rs / 1000.0
    idrflg = 0
    mu = 1 / (2 * X + 0.75 * Y + 0.5 * Z)
    initsh = 0
    r[initsh] = Rs
    M_r[initsh] = Ms
    L_r[initsh] = Ls
    T[initsh] = T0
    P[initsh] = P0
    tog_bf[initsh] = tog_bf0
    if P0 <= 0.0 or T0 <= 0.0:
        rho[initsh] = 0.0
        kappa[initsh] = 0.0
        epslon[initsh] = 0.0
        tog_bf[initsh] = 0.01
    else:
        rho[initsh], kappa[initsh], epslon[initsh], tog_bf[initsh], ierr = eos(X, Z, XCNO, mu, P[initsh], T[initsh], 0)
    if ierr != 0:
        istop = 0
    irc = 0
    dlPdlT[initsh] = 4.25
    for i in range(Nstart):
        ip1 = i + 1
        r[ip1], P[ip1], M_r[ip1], L_r[ip1], T[ip1] = startmdl(deltar, X, Z, mu, Rs, r[i], M_r[i], L_r[i], tog_bf[i], irc)
        rho[ip1], kappa[ip1], epslon[ip1], tog_bf[ip1], ierr = eos(X, Z, XCNO, mu, P[ip1], T[ip1], ip1)
        if ierr != 0:
            break
        if i > initsh:
            dlPdlT[ip1] = np.log(P[ip1] / P[i]) / np.log(T[ip1] / T[i])
        else:
            dlPdlT[ip1] = dlPdlT[i]
        if dlPdlT[ip1] < 2.5:
            irc = 1
        else:
            irc = 0
        deltaM = deltar * dMdr(r[ip1], rho[ip1])
        M_r[ip1] = M_r[i] + deltaM
        if np.abs(deltaM) > 0.001 * Ms:
            if ip1 > 1:
                ip1 -= 1
                break
    Nsrtp1 = ip1 + 1
    if ierr != 0:
        Nstop = Nsrtp1 - 1
        istop = Nstop
    for i in range(Nsrtp1, Nstop):
        im1 = i - 1
        f_im1[0] = P[im1]
        f_im1[1] = M_r[im1]
        f_im1[2] = L_r[im1]
        f_im1[3] = T[im1]
        dfdr[0] = dPdr(r[im1], M_r[im1], rho[im1])
        dfdr[1] = dMdr(r[im1], rho[im1])
        dfdr[2] = dLdr(r[im1], rho[im1], epslon[im1])
        dfdr[3] = dTdr(r[im1], M_r[im1], L_r[im1], T[im1], rho[im1], kappa[im1], mu, irc)
        f_i, ierr = runge(f_im1, dfdr, r[im1], deltar, irc, X, Z, XCNO, mu, i)
        if ierr != 0:
            break
        r[i] = r[im1] + deltar
        P[i] = f_i[0]
        M_r[i] = f_i[1]
        L_r[i] = f_i[2]
        T[i] = f_i[3]
        rho[i], kappa[i], epslon[i], tog_bf[i], ierr = eos(X, Z, XCNO, mu, P[i], T[i], i)
        if ierr != 0:
            istop = i
            break
        dlPdlT[i] = np.log(P[i] / P[im1]) / np.log(T[i] / T[im1])
        if dlPdlT[i] < 2.5:
            irc = 1
        else:
            irc = 0
        if r[i] <= np.abs(deltar) and (L_r[i] >= 0.1 * Ls or M_r[i] >= 0.01 * Ms):
            Igoof = 6
        elif L_r[i] <= 0.0:
            Igoof = 5
            rhocor = M_r[i] / (4.0 / 3.0 * np.pi * r[i] ** 3)
            if M_r[i] != 0.0:
                epscor = L_r[i] / M_r[i]
            else:
                epscor = 0.0
            Pcore = P[i] + 2.0 / 3.0 * np.pi * 6.67259e-08 * rhocor ** 2 * r[i] ** 2
            Tcore = Pcore * mu * 1.673534e-24 / (rhocor * 1.380658e-16)
        elif M_r[i] <= 0.0:
            Igoof = 4
            Rhocor = 0.0
            epscor = 0.0
            Pcore = 0.0
            Tcore = 0.0
        elif r[i] < 0.02 * Rs and (M_r[i] < 0.01 * Ms and L_r[i] < 0.1 * Ls):
            rhocor = M_r[i] / (4.0 / 3.0 * np.pi * r[i] ** 3)
            rhomax = 10.0 * (rho[i] / rho[im1]) * rho[i]
            epscor = L_r[i] / M_r[i]
            Pcore = P[i] + 2.0 / 3.0 * np.pi * 6.67259e-08 * rhocor ** 2 * r[i] ** 2
            Tcore = Pcore * mu * 1.673534e-24 / (rhocor * 1.380658e-16)
            if rhocor < rho[i] or rhocor > rhomax:
                Igoof = 1
            elif epscor < epslon[i]:
                Igoof = 2
            elif Tcore < T[i]:
                Igoof = 3
            else:
                Igoof = 0
        if Igoof != -1:
            istop = i
            break
        if idrflg == 0 and M_r[i] < 0.99 * Ms:
            deltar = -1.0 * Rs / 1000.0
            idrflg = 1
        if idrflg == 1 and abs(deltar) >= 0.5 * r[i]:
            deltar = -1.0 * Rs / 5000.0
            idrflg = 2
        istop = i
    rhocor = M_r[istop] / (4.0 / 3.0 * np.pi * r[istop] ** 3)
    epscor = L_r[istop] / M_r[istop]
    Pcore = P[istop] + 2.0 / 3.0 * np.pi * 6.67259e-08 * rhocor ** 2 * r[istop] ** 2
    Tcore = Pcore * mu * 1.673534e-24 / (rhocor * 1.380658e-16)
    return (Igoof, ierr, istop, r, P, M_r, L_r, T, rho, kappa, epslon, dlPdlT, rhocor, epscor, Pcore, Tcore)
