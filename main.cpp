#include <stdio.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <utility>
#include <matplot/matplot.h>
#include <unistd.h>


// Enzyme definitions
int enzyme_dup;
int enzyme_dupnoneed;
int enzyme_out;
int enzyme_const;

template < typename return_type, typename ... T >
return_type __enzyme_fwddiff(void*, T ... );

template < typename return_type, typename ... T >
return_type __enzyme_autodiff(void*, T ... );

template < typename T, typename ... arg_types >
auto wrapper(T obj, arg_types && ... args) {
    return obj(args ... );
}

template < typename T, typename ... arg_types >
auto LogObj(T obj, arg_types && ... args) {
    return std::log(obj(args ... ));
}

double Gaussian2(double x, double y, double m, double s) {
    double coef = 1 / (2 * std::numbers::pi * s * s);
    double xm = (x - m) / s;
    xm *= xm;
    double ym = (y - m) / s;
    ym *= ym;
    double e = std::exp(-0.5 * (xm + ym));
    return coef * e;
}

double Gaussian(double x, double m, double s) {
    double coef = 1 / std::sqrt(2 * std::numbers::pi * s * s);
    double xm = (x - m) / s;
    xm *= xm;
    double e = std::exp(-0.5 * xm);
    return coef * e;
}

double Gaussian2s(double x, double y) {
    return Gaussian2(x, y, 0, .5);
}

double logGaussian2s(double x, double y) {
    return std::log(Gaussian2s(x, y));
}

double Custom2(double x, double y) {
    x = std::abs(x);
    y = std::abs(y);
    if (x * y == 0) {
        return 10;
    }
    return std::min(1 / (x * x + y * y), 10.);
}

double logCustom2(double x, double y) {
    return std::log(Custom2(x, y));
}

struct double2 { double x, y; };
struct bounds2 { double2 a, b; };

struct Samples2 {
    bounds2 bounds;
    double left, right;
    double top, bottom;
    int n, m;
    std::vector<std::vector<double>> data;
    double weight;
    int n_samples;
    Samples2(bounds2 bounds, int n, int m, int n_samples = 1, double norm = 1.0) : 
        bounds(bounds), n(n), m(m), data(n, std::vector<double>(m, 0)) {
            double hor_step = (bounds.a.y - bounds.a.x) / n;
            double ver_step = (bounds.b.y - bounds.b.x) / m;
            weight = norm / (hor_step * ver_step * n_samples);
        }
    void add(double2 sample) {
        double x = (sample.x - bounds.a.x) / (bounds.a.y - bounds.a.x) * n;
        double y = (sample.y - bounds.b.x) / (bounds.b.y - bounds.b.x) * m;

        if (x < 0) { return; }
        if (y < 0) { return; }

        int i = x;
        if (i >= n) { return; }

        int j = y;
        if (j >= m) { return; }

        data[i][j] += weight;
    }
    void print() {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                std::cout << data[i][j] << " ";
            }   
            std::cout << std::endl;
        }
    }
};

void mala(double (*logf)(double2 p), double2 (*grad)(double2 p), double n_samples, std::vector<double2> &samples) {
    int n_accepted = 0;
    double h = .5;

    std::random_device rd;
    std::mt19937 gen1(rd());
    std::mt19937 gen2(rd());
    std::uniform_real_distribution<> udis(0.0f, 1.0f);

    double2 cur{.0, .0};
    samples.push_back(cur);

    double2 gradlog = grad(cur);
    for (int i = 0; i < n_samples; ++i) {
        std::normal_distribution<> disx(cur.x + 0.5 * h * gradlog.x, h);
        std::normal_distribution<> disy(cur.y + 0.5 * h * gradlog.y, h);
        double2 next = {disx(gen1), disy(gen1)};

        double2 gradlognext = grad(next);

        double transxy = Gaussian(next.x, cur.x + 0.5 * h * gradlog.x, h) * Gaussian(next.y, cur.y + 0.5 * h * gradlog.y, h);
        double transyx = Gaussian(cur.x, next.x + 0.5 * h * gradlognext.x, h) * Gaussian(cur.y, next.y + 0.5 * h * gradlognext.y, h);

        double A = std::exp(logf(next) - logf(cur)) * transyx / transxy;

        if (udis(gen2) < A) {            
            cur = next;
            gradlog = gradlognext;
            ++n_accepted;
        }

        samples.push_back(cur);
    }

    std::cout << "Acceptance prob: " << (double) n_accepted / n_samples << std::endl;
}

double volume(double (*f)(double2 p), bounds2 bounds, int n_samples = 100000);

double dis1(double2 p) {
    return Gaussian2(p.x, p.y, 0, 0.5);
}

double dis2(double2 p) {
    return std::min(10.0, 3 / (p.x * p.x + p.y * p.y));
}

double logdis1(double2 p) {
    return std::log(dis1(p));
}

double logdis2(double2 p) {
    return std::log(dis2(p));
}

int main() {
    auto grad1 = [](double2 p){
        return __enzyme_autodiff<double2>((void*)logdis1,
            enzyme_out, p.x,
            enzyme_out, p.y);
    };
    auto grad2 = [](double2 p){
        if (logdis2(p) == 10) {
            return double2(0, 0);
        }
        return double2(-2 * p.x * logdis2(p), -2 * p.y * logdis2(p));
    };

    auto dis = dis2;
    auto logdis = logdis2;
    auto grad = grad2;
    double volume = 45;

    int n_samples = 1000000;

    std::vector<double2> samples;
    mala(logdis, grad, n_samples, samples);

    int n_bins = 30;
    double x = 2;
    bounds2 bounds {{-x, x}, {-x, x}};
    Samples2 samples_grid(bounds, n_bins, n_bins, n_samples, volume);
    for (auto i : samples) {
        samples_grid.add(i);
    }

    {
        using namespace matplot;

        auto fig = figure(true);
        hold(on);

        auto [X, Y] = meshgrid(linspace(bounds.a.x, bounds.a.y, n_bins), linspace(bounds.b.x, bounds.b.y, n_bins));
        auto Z = transform(X, Y, [&](double x, double y){ return dis({x, y}); });
        surf(X, Y, Z);

        fig->save("plot.png");
    }
    {
        using namespace matplot;

        auto fig = figure(true);
        hold(on);

        auto [X, Y] = meshgrid(linspace(bounds.a.x, bounds.a.y, n_bins), linspace(bounds.b.x, bounds.b.y, n_bins));
        auto Z = transform(X, Y, [&](double x, double y) { 
            x = (x - bounds.a.x) / (bounds.a.y - bounds.a.x) * n_bins;
            y = (y - bounds.b.x) / (bounds.b.y - bounds.b.x) * n_bins;

            if (x < 0) { return 0.; }
            if (y < 0) { return 0.; }

            int i = x;
            if (i >= n_bins) { return 0.; }

            int j = y;
            if (j >= n_bins) { return 0.; }

            return samples_grid.data[i][j];
        });
        surf(X, Y, Z);

        int n_plot = 40;
        std::vector<double> x, y, z;
        for (int i = 0; i < n_plot; ++i) {
            double2 point = samples[i];
            x.push_back(point.x);
            y.push_back(point.y);
            z.push_back(dis({point.x, point.y}));
        }
        auto p = plot3(x, y, z);
        p->line_width(1).color("red");

        fig->save("plot2.png");

        show();
    }
}

double volume(double (*f)(double2 p), bounds2 bounds, int n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis_x(bounds.a.x, bounds.a.y);
    std::uniform_real_distribution<double> dis_y(bounds.b.x, bounds.b.y);

    double sum = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        double x = dis_x(gen);
        double y = dis_y(gen);
        sum += f({x, y});
    }

    double average = sum / n_samples;
    double area = (bounds.a.y - bounds.a.x) * (bounds.b.y - bounds.b.x);
    double integral = average * area;

    return integral;
}