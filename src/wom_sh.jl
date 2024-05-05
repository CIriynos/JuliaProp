
const WOM_MODE_PL = 1
const WOM_MODE_ELLI = 2

function window_operator_method_sh(shwave, gamma, n, Ev_list, rt::tdse_sh_rt, pw::physics_world_sh_t, mode)
    q_nk(n, k) = (2k - 1) * pi / 2 ^ n
    Plist = similar(Ev_list)    # Pγn(E)
    foo = gamma ^ (2^n)
    id_list = 1: ((mode == WOM_MODE_PL) ? pw.shgrid.l_num : (pw.shgrid.l_num ^ 2))

    for j in eachindex(Ev_list)
        Ev = Ev_list[j]
        # init phi
        copy_shwave(rt.phi, shwave)
        for i in id_list
            rt.phi[i] .*= foo
        end
        # apply window opeartor (i.e. solve linear equations on each lm opponent.)
        Threads.@threads for id in id_list
            l = rt.lmap[id]
            m = rt.mmap[id]
            for k = reverse(1: 2^(n-1))     # from outer to inner
                alpha_pos = -Ev + exp(im * q_nk(n, k)) * gamma
                alpha_neg = -Ev - exp(im * q_nk(n, k)) * gamma
                
                @. rt.Htmp_list[id] = rt.Hl_right_list_boost[l+1] + alpha_pos * rt.M2_boost
                mul!(rt.phi_tmp[id], rt.M2_boost, rt.phi[id])
                rt.phi[id] .= rt.phi_tmp[id]
                trimat_elimination(rt.phi_tmp[id], rt.Htmp_list[id], rt.phi[id], rt.A_add_list[id], rt.B_add_list[id])
                rt.phi[id] .= rt.phi_tmp[id]

                @. rt.Htmp_list[id] = rt.Hl_right_list_boost[l+1] + alpha_neg * rt.M2_boost
                mul!(rt.phi_tmp[id], rt.M2_boost, rt.phi[id])
                rt.phi[id] .= rt.phi_tmp[id]
                trimat_elimination(rt.phi_tmp[id], rt.Htmp_list[id], rt.phi[id], rt.A_add_list[id], rt.B_add_list[id])
                rt.phi[id] .= rt.phi_tmp[id]
            end
        end
        # get Pγn(E)
        res = real(sum(map(dot, @view(rt.phi[id_list]), @view(rt.phi[id_list]))))
        Plist[j] = res
        println("Pγn($Ev) = $res")
    end
    return Plist
end