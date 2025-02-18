
function coulomb_potiential_zero_fixed(;Rco::Float64 = 25.0)
    return r -> begin
        if r < Rco
            return -1.0 / r
        elseif r < 2 * Rco
            return -(2*Rco - r) / Rco^2
        else
            return 0.0
        end
    end
end

function coulomb_potiential_zero_fixed_plus(;Rco::Float64 = 25.0, a::Float64 = 3.0, b::Float64 = 2.5)
    return r -> begin
        if r <= Rco
            return -1.0 / r
        elseif r > Rco
            return -1.0 / r * exp(-((r - Rco) / (a * Rco)) ^ b)
        end
    end
end

function coulomb_potiential_zero_fixed_COS(Rco1, Rco2)
    return r -> begin
        if r <= Rco1
            return -1.0 / sqrt(r^2 + 1.0)
        elseif r <= Rco2
            return -1.0 / sqrt(r^2 + 1.0) * cos(((r - Rco1) / (Rco2 - Rco1)) * (pi/2)) ^ 2
        else
            return 0.0
        end
    end
end

function coulomb_potiential_helium_zero_fixed_plus(;Rco::Float64 = 25.0, a::Float64 = 3.0, b::Float64 = 2.5)
    return r -> begin
        if r <= Rco
            return (-1.0 / r - exp(-2.0329 * r) / r - 0.3953 * exp(-6.1805 * r))
        elseif r > Rco
            return (-1.0 / r - exp(-2.0329 * r) / r - 0.3953 * exp(-6.1805 * r)) * exp(-((r - Rco) / (a * Rco)) ^ b)
        end
    end
end


function flap_top_windows_f(t, t_min, t_max, ratio; left_flag=true, right_flag=true)
    T = t_max - t_min
    if t >= t_min && t < t_min + T * ratio
        if left_flag == true
            return sin(pi * (t - t_min) / (T * ratio * 2)) ^ 2
        else
            return 1.0
        end
    elseif t >= t_min + T * ratio && t < t_min + T * (1 - ratio)
        return 1.0
    elseif t >= t_min + T * (1 - ratio) && t < t_max
        if right_flag == true   
            return sin(pi * ((t - t_min) - T * (1 - ratio * 2)) / (T * ratio * 2)) ^ 2
        else
            return 1.0
        end
    elseif t < t_min
        if left_flag == false
            return 1.0
        else
            return 0.0
        end
    elseif t >= t_max
        if right_flag == false   
            return 1.0
        else
            return 0.0
        end
    end
end


function coulomb_potiential_zero_fixed_windows(Ri_tsurf; slop_ratio=1/4)
    return r -> -1.0 * (r ^ 2) ^ (-0.5) * flap_top_windows_f(r, 0, Ri_tsurf, slop_ratio, left_flag=false)
end


# function absorb_boundary_r(rmax; ratio::Float64 = 0.85, pow_value::Float64 = 8.0, max_value::Float64 = 100.0)
#     return r -> -im * max_value * ((r - rmax * ratio) / (rmax - rmax * ratio)) ^ pow_value * (r - rmax * ratio > 0.0)
# end

function absorb_boundary_r(rmax, rbound; pow_value::Float64 = 8.0, max_value::Float64 = 100.0)
    return r -> -im * max_value * ((r - rbound) / (rmax - rbound)) ^ pow_value * (r - rbound > 0.0)
end

function create_linspace(steps, delta; zero_start::Bool = true)
    if zero_start == true
        return [(i - 1) * delta for i = 1: steps]
    end
    return [i * delta for i = 1: steps]
end


# laser helper function
function light_pulse(omega, amplitude, cycle_num, time_delay; pulse_shape="sin2", ellipticity=0.0, phase1=0.0, phase2=pi/2, mode="E")
    E1 = amplitude * 1 / sqrt(ellipticity^2 + 1)
    E2 = amplitude * ellipticity / sqrt(ellipticity^2 + 1)
    Tp = 2pi * cycle_num / omega
    fixed_shift = Tp / 2
    C = 1.0
    tmax = time_delay + Tp

    if pulse_shape == "sin2" || pulse_shape == "cos2"
        pulse_shape_f = t -> cos(t * pi / Tp) ^ 2 * (t > -Tp/2 && t < Tp/2)
    elseif pulse_shape == "sin8" || pulse_shape == "cos8"
        pulse_shape_f = t -> cos(t * pi / Tp) ^ 8 * (t > -Tp/2 && t < Tp/2)
    elseif pulse_shape == "gauss" || pulse_shape == "normal"
        pulse_shape_f = t -> exp(-(t / (Tp / 2)) ^ 2)
    elseif pulse_shape == "none"
        pulse_shape_f = t -> 1 * (t > -Tp/2 && t < Tp/2)
    end

    if mode == "E"
        C = 1.0
    elseif mode == "A"
        C = 1.0 / omega
    end

    f_1(t) = C * E1 * pulse_shape_f(t - fixed_shift - time_delay) * cos(omega * (t - fixed_shift - time_delay) + phase1) + MIN_ERROR
    f_2(t) = C * E2 * pulse_shape_f(t - fixed_shift - time_delay) * cos(omega * (t - fixed_shift - time_delay) + phase2) + MIN_ERROR
    f_3(t) = MIN_ERROR

    return f_1, f_2, f_3, tmax
end

function dc_bias(amplitude, start_time, peak_time_1, peak_time_2, end_time; slop_ratio_left = 1/2, slop_ratio_right = 1/2)
    T1 = (peak_time_1 - start_time) * 2
    T2 = (end_time - peak_time_2) * 2
    f(t) = amplitude * flap_top_windows_f(t, start_time, start_time + T1, slop_ratio_left, right_flag = false) *
        flap_top_windows_f(t, end_time - T2, end_time, slop_ratio_right, left_flag = false)
    return f
end

no_light(t) = 0.0



# tau_fs = induce_time + 500
# tau_lst = [tau_fs - 2pi/ω2; range(tau_fs - 2pi/ω2 + tau_fs, tau_fs + nc1*2pi/ω1 - tau_fs, 14); tau_fs + nc1*2pi/ω1]
# # tau_lst = [tau_fs - 2pi/ω2, tau_fs + nc1*pi/ω1 - 1.5pi/ω2,
# #     tau_fs + nc1*pi/ω1 - pi/ω2, tau_fs + nc1*pi/ω1 - 0.5pi/ω2, tau_fs + nc1*2pi/ω1, 0.0]
# tau_thz = induce_time + tau_lst[tau_id]
# mid_point = tau_fs + nc1*pi/ω1 - pi/ω2

function get_1c_thz_delay_list(omega_pump, tau_pump, nc_pump, omega_thz; samples_num=16)
    T_thz = 2pi / omega_thz
    T_pump = nc_pump * 2pi / omega_pump
    tau_list = [tau_pump - T_thz; range(tau_pump - T_thz + tau_pump, tau_pump + T_pump - tau_pump, samples_num - 2); tau_pump + T_pump]
    return tau_list
end

function get_1c_thz_delay_list_ok(omega_pump, tau_pump, nc_pump, omega_thz; samples_num=16)
    T_thz = 2pi / omega_thz
    T_pump = nc_pump * 2pi / omega_pump
    tau_list = [tau_pump - T_thz; range(tau_pump - T_thz + T_pump * 0.3, tau_pump + T_pump - T_pump * 0.3, samples_num - 2); tau_pump + T_pump]
    return tau_list
end


# only for 5 samples
function get_1c_thz_delay_list_selected(omega_pump, tau_pump, nc_pump, omega_thz)
    tau_list = [
        tau_pump - 2pi/omega_thz,
        tau_pump + nc_pump*pi/omega_pump - 1.5pi/omega_thz,
        tau_pump + nc_pump*pi/omega_pump - pi/omega_thz,
        tau_pump + nc_pump*pi/omega_pump - 0.5pi/omega_thz,
        tau_pump + nc_pump*2pi/omega_pump]
    return tau_list
end

function get_exactly_coincided_delay(omega_pump, tau_pump, nc_pump, omega_thz)
    return tau_pump + nc_pump*pi/omega_pump - pi/omega_thz
end


function plot_fs_thz_figure(Ex_fs, Ey_fs, E_thz, ts; thz_ratio = 500)
    etp = plot(ts, [Ex_fs.(ts) Ey_fs.(ts) (E_thz.(ts)) .* thz_ratio],
    labels = ["E_fs_x" "E_fs_y" "E_thz"],
    xlabel="Time t",
    ylabel="E (a.u.)",
    title="fs Laser & THz Field Visualization (τ3)",
    guidefont=Plots.font(14, "Times"),
    tickfont=Plots.font(14, "Times"),
    titlefont=Plots.font(18, "Times"),
    legendfont=Plots.font(10, "Times"),
    margin = 5 * Plots.mm)
    return etp
end


function get_hhg_spectrum_xy(hhg_integral_t, Et_data_x, Et_data_y, pulse_start, pulse_end, ω0, ts, delta_t; max_display_rate=15, min_log_limit=1e-7, max_log_limit=1e3)
    
    hhg_xy_t = -hhg_integral_t .- (Et_data_x .+ im .* Et_data_y)

    start_id = Int64(floor(pulse_start ÷ delta_t)) + 1
    end_id = Int64(floor((pulse_end ÷ delta_t))) + 1
    steps = Int64(floor((pulse_end - pulse_start) ÷ delta_t)) + 1
    hhg_delta_k = 2pi / steps / delta_t
    hhg_k_linspace = [hhg_delta_k * (i - 1) for i = 1: steps]
    base_id = Int64(floor((ω0 ÷ hhg_delta_k))) + 1

    hhg_windows_f(t, tmin, tmax) = (1 - cos(2 * pi * (t - tmin) / (tmax - tmin))) / 2 * (t >= tmin && t <= tmax)
    hhg_windows_data = hhg_windows_f.(ts, pulse_start, pulse_end)

    hhg_spectrum_x = fft((real.(hhg_xy_t) .* hhg_windows_data)[start_id: end_id])
    hhg_spectrum_y = fft((imag.(hhg_xy_t) .* hhg_windows_data)[start_id: end_id])
    hhg_spectrum = norm.(hhg_spectrum_x) .^ 2 + norm.(hhg_spectrum_y) .^ 2

    spectrum_range = 1: Int64((max_display_rate * ω0) ÷ hhg_delta_k + 1)

    # plot(hhg_k_linspace[spectrum_range] ./ ω0, hhg_spectrum[spectrum_range], yscale=:log10)
    p = plot(hhg_k_linspace[spectrum_range] ./ ω0,
        [norm.(hhg_spectrum_x[spectrum_range]) norm.(hhg_spectrum_y[spectrum_range])],
        yscale=:log10, ylimit=(min_log_limit, max_log_limit))
    
    ll = Int64((ω0) ÷ hhg_delta_k) * hhg_delta_k / ω0
    plot!(p, [ll, ll], [min_log_limit, max_log_limit])
    ll = Int64((2ω0) ÷ hhg_delta_k) * hhg_delta_k / ω0
    plot!(p, [ll, ll], [min_log_limit, max_log_limit])
    ll = Int64((3ω0) ÷ hhg_delta_k) * hhg_delta_k / ω0
    plot!(p, [ll, ll], [min_log_limit, max_log_limit])
    ll = Int64((5ω0) ÷ hhg_delta_k) * hhg_delta_k / ω0
    plot!(p, [ll, ll], [min_log_limit, max_log_limit])
    return p, hhg_spectrum_x, hhg_spectrum_y, base_id, hhg_k_linspace[spectrum_range] ./ ω0, hhg_k_linspace[spectrum_range]
end


function create_tdata(tmax, tmin, Δt, Ex_f, Ex_y, Ex_z; appendix_steps::Int64 = 1)
    steps = Int64((tmax - tmin) ÷ Δt) + appendix_steps
    ts = [tmin + (i - 1) * Δt for i = 1: steps]
    Et_data_x = Ex_f.(ts)
    Et_data_y = Ex_y.(ts)
    Et_data_z = Ex_z.(ts)
    At_data_x = -get_integral(Et_data_x, Δt)
    At_data_y = -get_integral(Et_data_y, Δt)
    At_data_z = -get_integral(Et_data_z, Δt)
    return [At_data_x, At_data_y, At_data_z], [Et_data_x, Et_data_y, Et_data_z], ts, steps
end