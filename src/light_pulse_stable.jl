# A separate implementation to preserve the existing light_pulse interface.
function _stable_pulse_components(envelope::F, omega, E1, E2, shift, delay,
    phase1, phase2, scale, tmax) where {F}
    # Arguments are not reassigned and are captured directly, without Core.Box.
    f1 = t -> scale * E1 * envelope(t-shift-delay) * cos(omega*(t-shift-delay)+phase1) + MIN_ERROR
    f2 = t -> scale * E2 * envelope(t-shift-delay) * cos(omega*(t-shift-delay)+phase2) + MIN_ERROR
    f3 = t -> MIN_ERROR
    return f1, f2, f3, tmax
end

"""
    light_pulse_stable(omega, amplitude, cycle_num, time_delay;
        pulse_shape="sin2", ellipticity=0.0, phase1=0.0, phase2=pi/2, mode="E")

类型稳定的连续场生成接口，参数、返回值和支持的脉冲形状与 light_pulse 相同。
采用函数屏障固定包络和缩放因子的类型，不在闭包中使用 Core.Box。
组合电场时仍应在函数内部或 let 中捕获返回的场函数。
"""
function light_pulse_stable(omega, amplitude, cycle_num, time_delay;
    pulse_shape="sin2", ellipticity=0.0, phase1=0.0, phase2=pi/2, mode="E")
    E1 = amplitude * 1 / sqrt(ellipticity^2 + 1)
    E2 = amplitude * ellipticity / sqrt(ellipticity^2 + 1)
    Tp = 2pi * cycle_num / omega
    shift = Tp / 2
    tmax = time_delay + Tp
    envelope = if pulse_shape == "sin2" || pulse_shape == "cos2"
        t -> cos(t*pi/Tp)^2 * (t > -Tp/2 && t < Tp/2)
    elseif pulse_shape == "sin8" || pulse_shape == "cos8"
        t -> cos(t*pi/Tp)^8 * (t > -Tp/2 && t < Tp/2)
    elseif pulse_shape == "gauss" || pulse_shape == "normal"
        t -> exp(-(t/(Tp/2))^2)
    elseif pulse_shape == "none"
        t -> 1 * (t > -Tp/2 && t < Tp/2)
    else
        throw(ArgumentError("unsupported pulse_shape: $pulse_shape"))
    end
    scale = mode == "A" ? 1.0/omega : 1.0
    return _stable_pulse_components(envelope, omega, E1, E2, shift, time_delay,
        phase1, phase2, scale, tmax)
end
