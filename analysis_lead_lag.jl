

include("const.jl")
include("utils.jl")


ym_start = 202101
ym_end = 202503


# df_map = get_df_map(bar_dir, ym_start, ym_end, symbols)
df_map = get_df_map(bar_dir, ym_start, ym_end, [sb1, sb2, sb5])


past_win = 600 ÷ interval_sec

for (sb, df_sb) in df_map
    past_c = [fill(NaN, past_win); df_sb.c[1:end-past_win]]
    past_ret_bp = 10_000 .* (df_sb.c .- past_c) ./ past_c
    df_sb[!, :prb] = past_ret_bp

    tg_sec = 20 * 60
    tg_interval = Int(round(tg_sec / interval_sec))
    fut_c = [df_sb.c[tg_interval + 1 : end]; fill(NaN, tg_interval)]
    ret_bp = (fut_c .- df_sb.c) .* 10_000 ./ df_sb.c
    ret_bp[isnan.(ret_bp)] .= 0.0
    df_sb.ret_bp = ret_bp

    ud = fill(0, size(df_sb, 1))
    ud[df_sb.o .< df_sb.c] .= 1
    ud[df_sb.o .> df_sb.c] .=- -1
    df_sb.ud = ud
end

se1 = rolling_ema(df_map[sb1].ud, 12*10)
le1 = rolling_ema(df_map[sb1].ud, 12 * 60 * 4)
ell1 = se1 ./ (le1 .+ 1.0)
histogram(ell1)

simple_view_feature_power(ell1, df_map[sb1].ret_bp)
simple_view_feature_power(ell1, df_map[sb2].ret_bp)
simple_view_feature_power(ell1, df_map[sb4].ret_bp)
simple_view_feature_power(ell1, df_map[sb5].ret_bp)


size(df_map[sb1])
size(df_map[sb2])
size(df_map[sb4])
size(df_map[sb5])


signal_long, signal_short = get_signals(ell1, 5.0, th_pct2=0.02)
backtest_sica(signal_long, signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb2].c, df_map[sb2].c, df_map[sb2].ts, (20 * 60) ÷ interval_sec, symbol=sb2, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb4].c, df_map[sb4].c, df_map[sb4].ts, (20 * 60) ÷ interval_sec, symbol=sb4, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb5].c, df_map[sb5].c, df_map[sb5].ts, (20 * 60) ÷ interval_sec, symbol=sb5, is_display=true)







si = [i for i in 1:2000:size(df_map[sb1], 1)]

plot(df_map[sb1].c[si])
plot(df_map[sb2].c[si])
plot(df_map[sb4].c[si])
plot(df_map[sb5].c[si])


plot(ell1[si], df_map[sb1].ret_bp[si], seriestype=:scatter, markersize=2)
plot(ell1[si], df_map[sb2].ret_bp[si], seriestype=:scatter, markersize=2)
plot(ell1[si], df_map[sb4].ret_bp[si], seriestype=:scatter, markersize=2)
plot(ell1[si], df_map[sb5].ret_bp[si], seriestype=:scatter, markersize=2)






se1 = rolling_ema(df_map[sb1].ud, 12*10)
le1 = rolling_ema(df_map[sb1].ud, 12 * 60 * 4)
# histogram(se1)
# histogram(le1)
ll1 = se1 .- le1
histogram(ll1)

simple_view_feature_power(ll1, df_map[sb1].ret_bp)
simple_view_feature_power(ll1, df_map[sb2].ret_bp)
simple_view_feature_power(ll1, df_map[sb4].ret_bp)
simple_view_feature_power(ll1, df_map[sb5].ret_bp)





se1 = rolling_ema(df_map[sb1].ud, 12 * 5)
le1 = rolling_ema(df_map[sb1].ud, 12 * 60 * 4)
std1 = rolling_ema_std(df_map[sb1].ud, 12 * 60 * 4)
ll1_norm = (se1 .- le1) ./ std1
histogram(ll1_norm)

simple_view_feature_power(ll1_norm, df_map[sb1].ret_bp)
simple_view_feature_power(ll1_norm, df_map[sb2].ret_bp)
simple_view_feature_power(ll1_norm, df_map[sb4].ret_bp)
simple_view_feature_power(ll1_norm, df_map[sb5].ret_bp)

plot(ll1_norm[si], df_map[sb1].ret_bp[si], seriestype=:scatter, markersize=2)
plot(ll1_norm[si], df_map[sb2].ret_bp[si], seriestype=:scatter, markersize=2)
plot(ll1_norm[si], df_map[sb4].ret_bp[si], seriestype=:scatter, markersize=2)
plot(ll1_norm[si], df_map[sb5].ret_bp[si], seriestype=:scatter, markersize=2)


signal_long, signal_short = get_signals(ll1_norm, 1.0, th_pct2=0.01)
backtest_sica(signal_long, signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb2].c, df_map[sb2].c, df_map[sb2].ts, (20 * 60) ÷ interval_sec, symbol=sb2, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb4].c, df_map[sb4].c, df_map[sb4].ts, (20 * 60) ÷ interval_sec, symbol=sb4, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb5].c, df_map[sb5].c, df_map[sb5].ts, (20 * 60) ÷ interval_sec, symbol=sb5, is_display=true)

backtest_sica_2(signal_long, signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, symbol=sb1, is_display=true)




signal = [i for (i, v) in enumerate(signal_long) if v]
d = diff(signal)
histogram(d, xlims=(0, 1000))

length(d)
sum(d .< 100)

fd = d[d .< 100]
histogram(fd, bins=100, xlims=(0, 10))


histogram(signal)










se1_norm = norm_by_before_n_days(rolling_ema(df_map[sb1].ud, 12 * 5), 7, interval_sec)
le1_norm = norm_by_before_n_days(rolling_ema(df_map[sb1].ud, 12 * 60 * 4), 7, interval_sec)
histogram(se1_norm, xlims=(-10, 10))
histogram(le1_norm, xlims=(-10, 10))
ll1_norm2 = se1_norm .- le1_norm
# ll1_norm2 = le1_norm .- se1_norm
histogram(ll1_norm2, xlims=(-10, 10))

simple_view_feature_power(ll1_norm2, df_map[sb1].ret_bp)
simple_view_feature_power(ll1_norm2, df_map[sb2].ret_bp)
simple_view_feature_power(ll1_norm2, df_map[sb4].ret_bp)
simple_view_feature_power(ll1_norm2, df_map[sb5].ret_bp)

# plot(ll1_norm2[si], df_map[sb1].ret_bp[si], seriestype=:scatter, markersize=2)
# plot(ll1_norm2[si], df_map[sb2].ret_bp[si], seriestype=:scatter, markersize=2)
# plot(ll1_norm2[si], df_map[sb4].ret_bp[si], seriestype=:scatter, markersize=2)
# plot(ll1_norm2[si], df_map[sb5].ret_bp[si], seriestype=:scatter, markersize=2)

signal_long, signal_short = get_signals(ll1_norm2, 1.0, th_pct2=0.0)
backtest_sica(signal_long, signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb2].c, df_map[sb2].c, df_map[sb2].ts, (20 * 60) ÷ interval_sec, symbol=sb2, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb4].c, df_map[sb4].c, df_map[sb4].ts, (20 * 60) ÷ interval_sec, symbol=sb4, is_display=true)
backtest_sica(signal_long, signal_short, df_map[sb5].c, df_map[sb5].c, df_map[sb5].ts, (20 * 60) ÷ interval_sec, symbol=sb5, is_display=true)

pl, tr_res_vec = backtest_sica_keep(signal_long, signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)
pl, tr_res_vec = backtest_sica_keep(signal_long, signal_short, df_map[sb2].c, df_map[sb2].c, df_map[sb2].ts, (20 * 60) ÷ interval_sec, symbol=sb2, is_display=true)
pl, tr_res_vec = backtest_sica_keep(signal_long, signal_short, df_map[sb4].c, df_map[sb4].c, df_map[sb4].ts, (20 * 60) ÷ interval_sec, symbol=sb4, is_display=true)
pl, tr_res_vec = backtest_sica_keep(signal_long, signal_short, df_map[sb5].c, df_map[sb5].c, df_map[sb5].ts, (20 * 60) ÷ interval_sec, symbol=sb5, is_display=true)

# pl, tr_res_vec = backtest_sica_keep(signal_long, fill(false, length(signal_long)), df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)
# backtest_sica(signal_long, fill(false, length(signal_long)), df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)

# pl, tr_res_vec = backtest_sica_keep(fill(false, length(signal_long)), signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)
# backtest_sica(fill(false, length(signal_long)), signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, (20 * 60) ÷ interval_sec, symbol=sb1, is_display=true)


lp = [res[end] for res in tr_res_vec if res[2] == 1]
sp = [res[end] for res in tr_res_vec if res[2] == -1]




function get_ll_norm(df)
    se_norm = norm_by_before_n_days(rolling_ema(df.ud, 12 * 3), 7, interval_sec)
    le_norm = norm_by_before_n_days(rolling_ema(df.ud, 12 * 30 * 4), 7, interval_sec)
    ll_norm = se_norm .- le_norm
    return ll_norm
end



for sb in [sb1, sb2, sb3, sb4, sb5]
    if sb in keys(df_map)
        println("\n$sb")
        ll_norm = get_ll_norm(df_map[sb])
        signal_long, signal_short = get_signals(ll_norm, 0.50, th_pct2=0.1)
        backtest_sica_keep(signal_long, signal_short, df_map[sb].c, df_map[sb].c, df_map[sb].ts, (20 * 60) ÷ interval_sec, symbol=sb, is_display=true)
    end
end



Int(df_map[sb].ts[end])




















signal_long, signal_short = get_signals(ll1_norm, 1.0)

sum(signal_long) / size(df_map[sb1], 1)


sb = sb1
backtest_sica_2(signal_long, signal_short, df_map[sb1].c, df_map[sb1].c, df_map[sb1].ts, symbol=sb1, is_display=true)
backtest_sica_2(signal_long, signal_short, df_map[sb2].c, df_map[sb2].c, df_map[sb2].ts, symbol=sb2, is_display=true)
backtest_sica_2(signal_long, signal_short, df_map[sb4].c, df_map[sb4].c, df_map[sb4].ts, symbol=sb4, is_display=true)
backtest_sica_2(signal_long, signal_short, df_map[sb5].c, df_map[sb5].c, df_map[sb5].ts, symbol=sb5, is_display=true)









