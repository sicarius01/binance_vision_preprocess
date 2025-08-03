


function ft_gen_ofi(df; symbol="")
    of_b = df[!, "orderflow_v2_bid_place_volume_0_bp"] .- df[!, "orderflow_v2_bid_cancel_volume_0_bp"]
    of_a = df[!, "orderflow_v2_ask_place_volume_0_bp"] .- df[!, "orderflow_v2_ask_cancel_volume_0_bp"]
    ofi = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        of_b_norm = norm_by_before_n_days(of_b, 7, 1)
        of_a_norm = norm_by_before_n_days(of_a, 7, 1)
        ofi = of_b_norm .- of_a_norm
        ofi
    end
    return ofi
end

function ft_gen_aq(df; symbol="")
    aq_a = (df[!, "AggQtyByBP_Ask_0_bp"] .+ df[!, "AggQtyByBP_Ask_1_bp"])
    aq_b = (df[!, "AggQtyByBP_Bid_0_bp"] .+ df[!, "AggQtyByBP_Bid_1_bp"])
    aq = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        aq_a_norm = norm_by_before_n_days(aq_a, 7, 1)
        aq_b_norm = norm_by_before_n_days(aq_b, 7, 1)
        aq = aq_b_norm .- aq_a_norm
        aq
    end
    return aq
end

function ft_gen_tor(df; symbol="")
    tor_a = df[!, "TickFlow_buy_volume"] ./ (1.0 .+ df[!, "AggQtyByBP_Ask_0_bp"])
    tor_b = df[!, "TickFlow_sell_volume"] ./ (1.0 .+ df[!, "AggQtyByBP_Bid_0_bp"])
    tor_ema_norm = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        # tor_a_ema_norm = ema_norm(tor_a, 15, 1800)
        # tor_b_ema_norm = ema_norm(tor_b, 15, 1800)
        # tor_ema_norm = tor_b_ema_norm .- tor_a_ema_norm
        tor_a_ema_norm = ema_norm(tor_a, 2, 30)
        tor_b_ema_norm = ema_norm(tor_b, 2, 30)
        tor_ema_norm = tor_a_ema_norm .- tor_b_ema_norm
        tor_ema_norm
    end
    return tor_ema_norm
end

function ft_gen_tv(df; symbol="")
    tv_a = df[!, "TickFlow_buy_volume"]
    tv_b = df[!, "TickFlow_sell_volume"]
    tv = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        # tv_a_norm = norm_by_before_n_days(tv_a, 1, 1)
        # tv_b_norm = norm_by_before_n_days(tv_b, 1, 1)
        tv_a_norm = norm_by_before_n_days(tv_a, 2, 15)
        tv_b_norm = norm_by_before_n_days(tv_b, 2, 15)
        tv = tv_b_norm .- tv_a_norm
        tv
    end
    return tv
end

function ft_gen_rand(df; symbol="")
    return randn(size(df, 1))
end

function ft_gen_index_dist(df; symbol="")
    index_dist = 10_000 .* (df[!, "index_price"] .- df[!, "WAP_Lag_0ms"]) ./ df[!, "WAP_Lag_0ms"]
    index_dist_ema_norm = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        # ema_norm(index_dist, 60, 7200)
        index_dist_ema_norm = ema_norm(index_dist, 2, 10)
    end
    return index_dist_ema_norm
end

function ft_gen_mark_dist(df; symbol="")
    mark_dist = 10_000 .* (df[!, "mark_price"] .- df[!, "WAP_Lag_0ms"]) ./ df[!, "WAP_Lag_0ms"]
    mark_dist_ema_norm = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        # ema_norm(mark_dist, 60, 7200)
        ema_norm(mark_dist, 2, 10)
    end
    return mark_dist_ema_norm
end

function ft_gen_trcr(df; symbol="")
    trcr_b = df[!, "TickFlow_buy_count"] ./ (df[!, "TickFlow_total_count"])
    trcr_s = df[!, "TickFlow_sell_count"] ./ (df[!, "TickFlow_total_count"])
    trcr = if symbol in ["BTCUSDT", "ETHUSDT", ""]
        # trcr_b_norm = ema_norm(trcr_b, 1, 300)
        # trcr_s_norm = ema_norm(trcr_s, 1, 300)
        trcr_b_norm = ema_norm(trcr_b, 3, 200)
        trcr_s_norm = ema_norm(trcr_s, 3, 200)
        trcr = trcr_b_norm .- trcr_s_norm
        trcr
    end
    return trcr
end

function ft_gen_trv(df; symbol="")
    trv_b = df[!, "TickFlow_buy_volume"]
    trv_s = df[!, "TickFlow_sell_volume"]
    trv = if symbol in ["BTCUSDT", ""]
        trv_b_norm = ema_norm(trv_b, 1, 300)
        trv_s_norm = ema_norm(trv_s, 1, 300)
        trv = trv_b_norm .- trv_s_norm
        trv
    end
    return trv
end

function ft_gen_liqu(df; symbol="")
    liqu_side = fill(-1, size(df, 1))
    liqu_side[df[!, "side"] .== "buy"] .= 1
    liqu = liqu_side .* df.price .* df.amount
    liqu_ema_norm = if symbol in ["BTCUSDT", ""]
        liqu_ema_norm = ema_norm(liqu, 30, 3600)
        liqu_ema_norm
    end
    return liqu_ema_norm
end

function ft_gen_vwap(df; symbol="")
    vwap_b = 10_000 .* (df[!, "TickFlow_buy_money_volume"] ./ df[!, "TickFlow_buy_volume"]) .- df[!, "WAP_Lag_0ms"]
    vwap_s = 10_000 .* (df[!, "TickFlow_sell_money_volume"] ./ df[!, "TickFlow_sell_volume"]) .- df[!, "WAP_Lag_0ms"]
    vwap_ema_norm = if symbol in ["BTCUSDT", ""]
        # vwap_b_ema_norm = ema_norm(vwap_b, 3, 600)        
        # vwap_s_ema_norm = ema_norm(vwap_s, 3, 600)
        vwap_b_ema_norm = ema_norm(vwap_b, 2, 30)
        vwap_s_ema_norm = ema_norm(vwap_s, 2, 30)
        vwap_ema_norm = vwap_b_ema_norm .- vwap_s_ema_norm
        vwap_ema_norm
    end
    return vwap_ema_norm
end





function ft_gen_ttc(df; symbol="")
    return df[!, "TickFlow_total_count"]
end

function ft_gen_doi(df; symbol="")
    doi = [NaN; abs.(diff(df[!, "open_interest"]))]
    ft = if symbol in ["BTCUSDT", ""]
        ft = ema_norm(doi, 10, 60)
        ft
    end
    return ft
end





