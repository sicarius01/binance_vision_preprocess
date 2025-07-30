


function ft_gen_ofi(df)
    of_b = df[!, "orderflow_v2_bid_place_volume_0_bp"] .- df[!, "orderflow_v2_bid_cancel_volume_0_bp"]
    of_a = df[!, "orderflow_v2_ask_place_volume_0_bp"] .- df[!, "orderflow_v2_ask_cancel_volume_0_bp"]
    of_b_norm = norm_by_before_n_days(of_b, 7, 1)
    of_a_norm = norm_by_before_n_days(of_a, 7, 1)
    ofi = of_b_norm .- of_a_norm
    return ofi
end

function ft_gen_aq(df)
    aq_a = (df[!, "AggQtyByBP_Ask_0_bp"] .+ df[!, "AggQtyByBP_Ask_1_bp"])
    aq_b = (df[!, "AggQtyByBP_Bid_0_bp"] .+ df[!, "AggQtyByBP_Bid_1_bp"])
    aq_a_norm = norm_by_before_n_days(aq_a, 7, 1)
    aq_b_norm = norm_by_before_n_days(aq_b, 7, 1)
    aq = aq_b_norm .- aq_a_norm
    return aq
end








