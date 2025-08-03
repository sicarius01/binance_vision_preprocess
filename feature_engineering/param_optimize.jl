


include("./../const.jl")
include("./../utils.jl")
include("./../feature_generator/feature_generator.jl")

begin
    df_train_map::Dict{String, DataFrame} = Dict()
    @threads for sb in symbols
        date_list_train = [
            # [string(di) for di in 20230313:20230331]; 
            [string(di) for di in 20230401:20230430];
            [string(di) for di in 20230501:20230531]
            ]
        # df_list = [get_df_oneday_full(tardis_dir, s7_dir, sb, date_one_day, features) for date_one_day in date_list_train]
        df_list = [Threads.@spawn get_df_oneday_full(tardis_dir, s7_dir, sb, date, features) for date in date_list_train]
        df_list = fetch.(df_list)

        df_train = vcat(df_list...)
        set_ret_bp(df_train, ret_interval)
        df_train_map[sb] = df_train
    end
end

begin
    df_test_map::Dict{String, DataFrame} = Dict()
    @threads for sb in symbols
        date_list_test = [
            [string(di) for di in 20230525:20230531];
            [string(di) for di in 20230601:20230630]
            ]

        # df_list2 = [get_df_oneday_full(tardis_dir, s7_dir, sb, date_one_day, features) for date_one_day in date_list_test]
        df_list2 = [Threads.@spawn get_df_oneday_full(tardis_dir, s7_dir, sb, date, features) for date in date_list_test]
        df_list2 = fetch.(df_list2)

        df_test = vcat(df_list2...)
        set_ret_bp(df_test, ret_interval)
        df_test_map[sb] = df_test
    end
end


######################################
# ETH 

sb = "ETHUSDT"
df = df_train_map[sb]

of_b = df[!, "orderflow_v2_bid_place_volume_0_bp"] .- df[!, "orderflow_v2_bid_cancel_volume_0_bp"]
of_a = df[!, "orderflow_v2_ask_place_volume_0_bp"] .- df[!, "orderflow_v2_ask_cancel_volume_0_bp"]
of_b_norm = norm_by_before_n_days(of_b, 7, 1)
of_a_norm = norm_by_before_n_days(of_a, 7, 1)
ofi = of_b_norm .- of_a_norm

histogram(of_b_norm, xlims=(-10, 10))
histogram(of_a_norm, xlims=(-10, 10))
histogram(ofi, xlims=(-10, 10))

simple_view_feature_power(of_b_norm, df.ret_bp)
simple_view_feature_power(of_a_norm, df.ret_bp)
simple_view_feature_power(ofi, df.ret_bp)



aq_a = (df[!, "AggQtyByBP_Ask_0_bp"] .+ df[!, "AggQtyByBP_Ask_1_bp"])
aq_b = (df[!, "AggQtyByBP_Bid_0_bp"] .+ df[!, "AggQtyByBP_Bid_1_bp"])
aq_a_norm = norm_by_before_n_days(aq_a, 7, 1)
aq_b_norm = norm_by_before_n_days(aq_b, 7, 1)
aq = aq_b_norm .- aq_a_norm

histogram(aq, xlims=(-10, 10))

simple_view_feature_power(aq, df.ret_bp)



tor_a = df[!, "TickFlow_buy_volume"] ./ (1.0 .+ df[!, "AggQtyByBP_Ask_0_bp"])
tor_b = df[!, "TickFlow_sell_volume"] ./ (1.0 .+ df[!, "AggQtyByBP_Bid_0_bp"])
tor_a_ema_norm = ema_norm(tor_a, 2, 30)
tor_b_ema_norm = ema_norm(tor_b, 2, 30)
tor_ema_norm = tor_a_ema_norm .- tor_b_ema_norm

# histogram(tor_ema_norm, xlims=(-10, 10))

simple_view_feature_power(tor_ema_norm, df.ret_bp)

tor_a_ema_norm = ema_norm(tor_a, 15, 1800)
tor_b_ema_norm = ema_norm(tor_b, 15, 1800)
tor_ema_norm = tor_b_ema_norm .- tor_a_ema_norm
simple_view_feature_power(tor_ema_norm, df.ret_bp)





tv_a = df[!, "TickFlow_buy_volume"]
tv_b = df[!, "TickFlow_sell_volume"]

tv_a_norm = norm_by_before_n_days(tv_a, 7, 1)
tv_b_norm = norm_by_before_n_days(tv_b, 7, 1)

tv_a_norm = norm_by_before_n_days(tv_a, 2, 15)
tv_b_norm = norm_by_before_n_days(tv_b, 2, 15)

tv = tv_a_norm .- tv_b_norm
simple_view_feature_power(tv, df.ret_bp)




index_dist = 10_000 .* ( df2[!, "WAP_Lag_0ms"] .- df2[!, "index_price"]) ./ df2[!, "WAP_Lag_0ms"]
index_dist = 10_000 .* ( df[!, "WAP_Lag_0ms"] .- df[!, "index_price"]) ./ df[!, "WAP_Lag_0ms"]

index_dist_ema_norm = ema_norm(index_dist, 2, 10)

simple_view_feature_power(index_dist_ema_norm, df.ret_bp)



df2 = df_train_map["BTCUSDT"]



mark_dist = 10_000 .* (df2[!, "WAP_Lag_0ms"] .- df2[!, "mark_price"]) ./ df2[!, "WAP_Lag_0ms"]
mark_dist = 10_000 .* (df[!, "WAP_Lag_0ms"] .- df[!, "mark_price"]) ./ df[!, "WAP_Lag_0ms"]
mark_dist_ema_norm = ema_norm(mark_dist, 60, 7200)

mark_dist_ema_norm = ema_norm(mark_dist, 2, 10)
simple_view_feature_power(mark_dist_ema_norm, df.ret_bp)



trcr_b = df2[!, "TickFlow_buy_count"] ./ (df2[!, "TickFlow_total_count"])
trcr_s = df2[!, "TickFlow_sell_count"] ./ (df2[!, "TickFlow_total_count"])

trcr_b = df[!, "TickFlow_buy_count"] ./ (df[!, "TickFlow_total_count"])
trcr_s = df[!, "TickFlow_sell_count"] ./ (df[!, "TickFlow_total_count"])
trcr_b_norm = ema_norm(trcr_b, 1, 300)
trcr_s_norm = ema_norm(trcr_s, 1, 300)

trcr_b_norm = ema_norm(trcr_b, 3, 200)
trcr_s_norm = ema_norm(trcr_s, 3, 200)
trcr = trcr_b_norm .- trcr_s_norm

simple_view_feature_power(trcr, df.ret_bp)



liqu_side[df2[!, "side"] .== "buy"] .= 1

liqu_side = fill(-1, size(df, 1))
liqu_side[df[!, "side"] .== "buy"] .= 1
liqu = liqu_side .* df.price .* df.amount

liqu_ema_norm = ema_norm(liqu, 30, 3600)

liqu_ema_norm = ema_norm(liqu, 18, 8100)

liqu_ema_norm = ema_norm(norm_by_before_n_days(liqu, 1, 1), 30, 3600)

simple_view_feature_power(liqu_ema_norm, df.ret_bp)

# histogram(liqu)
histogram(liqu_ema_norm)





vwap_b = 10_000 .* (df2[!, "TickFlow_buy_money_volume"] ./ df2[!, "TickFlow_buy_volume"]) .- df2[!, "WAP_Lag_0ms"]
vwap_s = 10_000 .* (df2[!, "TickFlow_sell_money_volume"] ./ df2[!, "TickFlow_sell_volume"]) .- df2[!, "WAP_Lag_0ms"]


vwap_b = 10_000 .* (df[!, "TickFlow_buy_money_volume"] ./ df[!, "TickFlow_buy_volume"]) .- df[!, "WAP_Lag_0ms"]
vwap_s = 10_000 .* (df[!, "TickFlow_sell_money_volume"] ./ df[!, "TickFlow_sell_volume"]) .- df[!, "WAP_Lag_0ms"]

vwap_b_ema_norm = ema_norm(vwap_b, 3, 600)
vwap_s_ema_norm = ema_norm(vwap_s, 3, 600)


vwap_b_ema_norm = ema_norm(vwap_b, 2, 30)
vwap_s_ema_norm = ema_norm(vwap_s, 2, 30)
vwap_ema_norm = vwap_b_ema_norm .- vwap_s_ema_norm

simple_view_feature_power(vwap_ema_norm, df.ret_bp)

simple_view_feature_power(vwap_ema_norm, df2.ret_bp)










