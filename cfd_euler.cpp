#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>

using namespace std;

// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

#pragma omp declare target
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

void fluxX(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho  = rhou;
    frhou = rhou*u + p;
    frhov = rhov*u;
    fE    = (E + p)*u;
}

void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho  = rhov;
    frhou = rhou*v;
    frhov = rhov*v + p;
    fE    = (E + p)*v;
}
#pragma omp end declare target

int main(int argc, char* argv[]) {
    int mult = 1;
    string label = "GPU";
    if (argc > 1) mult  = atoi(argv[1]);
    if (argc > 2) label = argv[2];

    const int Nx = 200 * mult;
    const int Ny = 100 * mult;
    const double Lx = 2.0, Ly = 1.0;
    const double dx = Lx / Nx, dy = Ly / Ny;
    const int N = (Nx+2) * (Ny+2);
    const int nSteps = 2000;

    double* rho      = new double[N];
    double* rhou     = new double[N];
    double* rhov     = new double[N];
    double* E        = new double[N];
    double* rho_new  = new double[N];
    double* rhou_new = new double[N];
    double* rhov_new = new double[N];
    double* E_new    = new double[N];
    int*    solid    = new int[N];

    const double rho0 = 1.0, u0 = 1.0, v0 = 0.0, p0 = 1.0;
    const double E0 = p0/(gamma_val-1.0) + 0.5*rho0*(u0*u0 + v0*v0);
    const double cx = 0.5, cy = 0.5, radius = 0.1;

    for (int i = 0; i < Nx+2; i++) {
        for (int j = 0; j < Ny+2; j++) {
            int idx = i*(Ny+2)+j;
            double x = (i - 0.5)*dx;
            double y = (j - 0.5)*dy;
            if ((x-cx)*(x-cx) + (y-cy)*(y-cy) <= radius*radius) {
                solid[idx] = 1;
                rho[idx]  = rho0;
                rhou[idx] = 0.0;
                rhov[idx] = 0.0;
                E[idx]    = p0/(gamma_val-1.0);
            } else {
                solid[idx] = 0;
                rho[idx]  = rho0;
                rhou[idx] = rho0*u0;
                rhov[idx] = rho0*v0;
                E[idx]    = E0;
            }
            rho_new[idx] = 0.0; rhou_new[idx] = 0.0;
            rhov_new[idx] = 0.0; E_new[idx] = 0.0;
        }
    }

    double c0 = sqrt(gamma_val*p0/rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0) / 2.0;

    auto t_start = chrono::high_resolution_clock::now();

    #pragma omp target data \
        map(tofrom: rho[0:N], rhou[0:N], rhov[0:N], E[0:N]) \
        map(alloc:  rho_new[0:N], rhou_new[0:N], rhov_new[0:N], E_new[0:N]) \
        map(to:     solid[0:N])
    {
        for (int n = 0; n < nSteps; n++) {

            // left inflow
            #pragma omp target teams distribute parallel for
            for (int j = 0; j < Ny+2; j++) {
                rho[j]  = rho0;
                rhou[j] = rho0*u0;
                rhov[j] = rho0*v0;
                E[j]    = E0;
            }

            // right outflow
            #pragma omp target teams distribute parallel for
            for (int j = 0; j < Ny+2; j++) {
                rho[(Nx+1)*(Ny+2)+j]  = rho[Nx*(Ny+2)+j];
                rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j];
                rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j];
                E[(Nx+1)*(Ny+2)+j]    = E[Nx*(Ny+2)+j];
            }

            // bottom reflective
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < Nx+2; i++) {
                rho[i*(Ny+2)]  = rho[i*(Ny+2)+1];
                rhou[i*(Ny+2)] = rhou[i*(Ny+2)+1];
                rhov[i*(Ny+2)] = -rhov[i*(Ny+2)+1];
                E[i*(Ny+2)]    = E[i*(Ny+2)+1];
            }

            // top reflective
            #pragma omp target teams distribute parallel for
            for (int i = 0; i < Nx+2; i++) {
                rho[i*(Ny+2)+(Ny+1)]  = rho[i*(Ny+2)+Ny];
                rhou[i*(Ny+2)+(Ny+1)] = rhou[i*(Ny+2)+Ny];
                rhov[i*(Ny+2)+(Ny+1)] = -rhov[i*(Ny+2)+Ny];
                E[i*(Ny+2)+(Ny+1)]    = E[i*(Ny+2)+Ny];
            }

            // Lax-Friedrichs interior update
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    int idx = i*(Ny+2)+j;
                    if (solid[idx]) {
                        rho_new[idx]  = rho[idx];
                        rhou_new[idx] = rhou[idx];
                        rhov_new[idx] = rhov[idx];
                        E_new[idx]    = E[idx];
                    } else {
                        rho_new[idx]  = 0.25*(rho[(i+1)*(Ny+2)+j]  + rho[(i-1)*(Ny+2)+j]  + rho[i*(Ny+2)+j+1]  + rho[i*(Ny+2)+j-1]);
                        rhou_new[idx] = 0.25*(rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + rhou[i*(Ny+2)+j+1] + rhou[i*(Ny+2)+j-1]);
                        rhov_new[idx] = 0.25*(rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + rhov[i*(Ny+2)+j+1] + rhov[i*(Ny+2)+j-1]);
                        E_new[idx]    = 0.25*(E[(i+1)*(Ny+2)+j]    + E[(i-1)*(Ny+2)+j]    + E[i*(Ny+2)+j+1]    + E[i*(Ny+2)+j-1]);

                        double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                        double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                        double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                        double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;

                        fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                              fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                        fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                              fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                        fluxY(rho[i*(Ny+2)+j+1], rhou[i*(Ny+2)+j+1], rhov[i*(Ny+2)+j+1], E[i*(Ny+2)+j+1],
                              fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                        fluxY(rho[i*(Ny+2)+j-1], rhou[i*(Ny+2)+j-1], rhov[i*(Ny+2)+j-1], E[i*(Ny+2)+j-1],
                              fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                        double dtdx = dt / (2.0*dx);
                        double dtdy = dt / (2.0*dy);

                        rho_new[idx]  -= dtdx*(fx_rho1  - fx_rho2)  + dtdy*(fy_rho1  - fy_rho2);
                        rhou_new[idx] -= dtdx*(fx_rhou1 - fx_rhou2) + dtdy*(fy_rhou1 - fy_rhou2);
                        rhov_new[idx] -= dtdx*(fx_rhov1 - fx_rhov2) + dtdy*(fy_rhov1 - fy_rhov2);
                        E_new[idx]    -= dtdx*(fx_E1    - fx_E2)    + dtdy*(fy_E1    - fy_E2);
                    }
                }
            }

            // copy updated values back
            #pragma omp target teams distribute parallel for collapse(2)
            for (int i = 1; i <= Nx; i++) {
                for (int j = 1; j <= Ny; j++) {
                    int idx = i*(Ny+2)+j;
                    rho[idx]  = rho_new[idx];
                    rhou[idx] = rhou_new[idx];
                    rhov[idx] = rhov_new[idx];
                    E[idx]    = E_new[idx];
                }
            }
        }
    }

    auto t_end = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration<double>(t_end - t_start).count();

    cout << label << "  Nx=" << Nx << "  Ny=" << Ny << "  Steps=" << nSteps << "  Time=" << elapsed << "s" << endl;

    delete[] rho; delete[] rhou; delete[] rhov; delete[] E;
    delete[] rho_new; delete[] rhou_new; delete[] rhov_new; delete[] E_new;
    delete[] solid;

    return 0;
}
