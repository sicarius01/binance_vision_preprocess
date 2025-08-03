
using CSV
using DataFrames
using Dates
using ZipFile
using Plots
using Parquet2
using StatsBase
using Base.Threads
using SkipNan
using CodecZlib
using CodecZstd
using Serialization
using Dates
using RollingFunctions
using Memoization
using MLJ
using MLJLinearModels
using LinearAlgebra
using LinearAlgebra.BLAS
using BenchmarkTools
using DataStructures





mutable struct RowState
    o::Float64
    h::Float64
    l::Float64
    c::Float64
    v::Float64
    total_cnt::Int64
    buy_flow_cnt::Int64
    sell_flow_cnt::Int64
    vwap::Float64

    function RowState()::RowState
        return new(NaN, NaN, NaN, NaN, 0.0, 0, 0, 0, NaN)
    end
end


function update_row_state(rs::RowState, df, i)
    p, q, id_s, id_e, ts, ibm = df[i, 2:end]
    
    (rs.o === NaN) && (rs.o = p)
    rs.h = rs.h === NaN ? p : (rs.h > p ? rs.h : p)
    rs.l = rs.l === NaN ? p : (rs.l < p ? rs.l : p)
    rs.c = p
    rs.v += q

    id_cnt = 1 + id_e - id_s
    rs.total_cnt += id_cnt
    ibm ? (rs.sell_flow_cnt += id_cnt) : (rs.buy_flow_cnt += id_cnt)

    rs.vwap = rs.vwap === NaN ? p : (rs.vwap * (rs.v - q) + p * q) / rs.v
end


function simple_view_feature_power(X, target)
    px = [0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 25, 50, 75, 90, 95, 99]
    py_l = []
    py_h = []
    for pct in px
        ml, mh = get_pct_mean(X, pct, target)
        ml, mh = round(ml, digits=4), round(mh, digits=4)
        println("pct: $(pct) - ml: $ml\tmh: $mh")
        push!(py_l, ml)
        push!(py_h, mh)
    end
    plot(px, py_l, seriestype=:scatter, label="low")
    plot!(px, py_h, seriestype=:scatter, label="high")
end


function simple_view_feature_power_return(
    X, target;
    title::Union{Nothing, String}=nothing,
    is_display=true,
    )
    println("")
    px = [1, 2, 3, 5, 10, 25, 50, 75, 90, 95, 99]
    py_l = []
    py_h = []
    for pct in px
        ml, mh = get_pct_mean(X, pct, target)
        ml, mh = round(ml, digits=4), round(mh, digits=4)
        println("pct: $(pct) - ml: $ml\tmh: $mh")
        push!(py_l, ml)
        push!(py_h, mh)
    end
    p = plot(px, py_l, seriestype=:scatter, label="low")
    plot!(px, py_h, seriestype=:scatter, label="high")
    if title !== nothing
        title!(title)
    end
    is_display && display(p)
    return py_l, py_h
end


function rolling_ema(data_vec::AbstractVector, span::Int;padding_value=0.0)
    n = length(data_vec)
    if n == 0
        return Float64[]
    end
    if span < 1
        error("span은 1 이상의 양의 정수여야 합니다.")
    end

    if n < span
        return fill(padding_value, n)
    end

    result_vector = fill(padding_value, n)
    alpha = 2.0 / (span + 1.0)
    current_ema = Float64(data_vec[1])

    for i in 2:n
        current_ema = (Float64(data_vec[i]) * alpha) + (current_ema * (1.0 - alpha))
        if i >= span
            result_vector[i] = current_ema
        end
    end

    return result_vector
end

function rolling_ema_std(data_vec::AbstractVector, span::Int; padding_value=0.0)
    n = length(data_vec)
    
    if span < 1
        error("span은 1 이상의 양의 정수여야 합니다.")
    end

    if n < span
        return fill(padding_value, n)
    end

    result_vector = fill(padding_value, n)
    alpha = 2.0 / (span + 1.0)

    ema_x = Float64(data_vec[1])
    ema_x2 = Float64(data_vec[1]^2)

    for i in 2:n
        current_val = Float64(data_vec[i])
        ema_x = (current_val * alpha) + (ema_x * (1.0 - alpha))
        ema_x2 = ((current_val^2) * alpha) + (ema_x2 * (1.0 - alpha))

        if i >= span
            variance = ema_x2 - ema_x^2
            result_vector[i] = sqrt(max(0.0, variance))
        end
    end

    return result_vector
end

function ema_norm(feature, l1, l2)
    feature[isnan.(feature)] .= 0.0
    ema_short = rolling_ema(feature, l1, padding_value=0.0)
    ema_long = rolling_ema(feature, l2, padding_value=0.0)
    ema_std = rolling_ema_std(feature, l2, padding_value=0.0)
    res = (ema_short - ema_long) ./ ema_std
    return res
end

function get_pct_mean(X, pct, target)
    x = X[.!isnan.(X)]
    th_low, th_high = quantile(x, pct/100), quantile(x, (100-pct)/100)
    # println("th_low: $(th_low), th_high: $(th_high)")
    mask_down = X .<= th_low
    mask_up = X .>= th_high
    return mean(skipnan(target[mask_down])), mean(skipnan(target[mask_up]))
end


function get_df_res(df; interval_sec=5)
    tsv = df.transact_time
    n_days = Int(ceil((tsv[end] - tsv[1]) / (24 * 3600 * 1000)))
    total_rows = (n_days * 3600 * 24) ÷ interval_sec
    cols = [:ts, :o, :h, :l, :c, :v, :total_id_cnt, :buy_flow_id_cnt, :sell_flow_id_cnt, :vwap]
    df_res = DataFrame()
    for col in cols
        df_res[!, col] = fill(NaN, total_rows)
    end

    idx_res = 1
    interval_ms = 1000 * interval_sec
    ts_s = 60_000 * (df.transact_time[1] ÷ 60_000)
    ts_e = ts_s + interval_ms
    rs = RowState()

    for (i, ts) in enumerate(df[!, end-1])
        if ts > ts_e
            ts_at_time = interval_ms * (ts ÷ interval_ms)
            df_res[idx_res, :] = [ts_at_time, rs.o, rs.h, rs.l, rs.c, rs.v, rs.total_cnt, rs.buy_flow_cnt, rs.sell_flow_cnt, rs.vwap]
            idx_res += 1
            ts_e += interval_ms
            ts_s += interval_ms
            rs = RowState()
        end
        update_row_state(rs, df, i)
    end
    ts_at_time = df_res[idx_res-1, 1] + interval_ms
    df_res[idx_res, :] = [ts_at_time, rs.o, rs.h, rs.l, rs.c, rs.v, rs.total_cnt, rs.buy_flow_cnt, rs.sell_flow_cnt, rs.vwap]
    return df_res
end




function read_zip_to_df(zip_path::String; col_names=nothing)
    zip_reader = ZipFile.Reader(zip_path)
    csv_file = filter(f -> endswith(f.name, ".csv"), zip_reader.files)[1]
    data = read(csv_file)
    io = IOBuffer(data)
    df = CSV.read(io, DataFrame; delim = ',', header = true, missingstring = "null")
    close(zip_reader)
    col_names !== nothing && rename!(df, col_names)
    return df
end


function is_no_alphabet(rows)
    for row in rows
        if lowercase(row) in ["true", "false"] continue end
        for value in row
            if occursin(r"[a-zA-Z]", string(value))
                return false
            end
        end
    end
    return true
end


function get_colname_by_daily(um_path, symbol, dtype)
    daily_path = joinpath(um_path, "daily", dtype, symbol)
    file_list = [fp for fp in readdir(daily_path) if endswith(fp, ".zip")]
    ffn = joinpath(daily_path, file_list[1])
    df_temp = read_zip_to_df(ffn)
    colnames = names(df_temp)
    return colnames
end


function get_d_m_path(um_path, symbol, dtype)
    d, m = "", ""
    if dtype == "premiumIndexKlines"
        d = joinpath(um_path, "daily", dtype, symbol, "1m")
        m = joinpath(um_path, "monthly", dtype, symbol, "1m")
    else
        d = joinpath(um_path, "daily", dtype, symbol)
        m = joinpath(um_path, "monthly", dtype, symbol)
    end
    return d, m
end

function get_file_list(um_path, symbol, dtype)
    d, m = get_d_m_path(um_path, symbol, dtype)
    daily_files = [joinpath(d, fn) for fn in readdir(d) if endswith(fn, ".zip")]
    monthly_files = [joinpath(m, fn) for fn in readdir(m) if endswith(fn, ".zip")]
    sort!(daily_files)
    sort!(monthly_files)
    fl = [monthly_files; daily_files]
    return fl
end



function process_one_month(file_name, symbol, save_dir, interval_sec)
    s = time()
    data_type = "aggTrades"
    month = split(split(file_name, ".")[1], "\\$(symbol)-$(data_type)-")[end]
    save_path = joinpath(save_dir, "bar_$(interval_sec)s", symbol, "$(month).parquet")
    if isfile(save_path)
        println("Already Exist : $(symbol)-$(month)")
        return
    end

    df_aggTr = read_zip_to_df(file_name, col_names=col_names_aggTr)
    df_res = get_df_res(df_aggTr, interval_sec=interval_sec)

    dn = dirname(save_path)
    isdir(dn) || mkpath(dn)
    Parquet2.writefile(save_path, df_res)
    e = time()
    println("Complete : $(symbol)-$(month) [$(round(e-s, digits=2)) sec]")
end


function get_symbol_list(um_path)
    aggTr_path = joinpath(um_path, "monthly", "aggTrades")
    symbol_list = readdir(aggTr_path)
    return symbol_list
end



function get_df_map(bar_dir, ym_start, ym_end, symbols)
    df_map = Dict()
    for symbol in symbols
        # println("\n$symbol")
        df_sb_list = []
        sb_path = joinpath(bar_dir, symbol)
        files = readdir(sb_path)
        for fn in files
            # println(fn)
            if occursin(".parquet", fn)
                ym_str = split(fn, ".parquet")[1]
                y, m = split(ym_str, "-")
                ym_num = parse(Int64, y) * 100 + parse(Int64, m)
                if ym_start <= ym_num <= ym_end
                    ffn = joinpath(sb_path, fn)
                    df_ym = Parquet2.readfile(ffn) |> DataFrame
                    push!(df_sb_list, df_ym)
                end
            end
        end
        df_sb = vcat(df_sb_list...)
        df_map[symbol] = df_sb
    end
    return df_map
end


function norm_by_before_n_days(feature, n_days, interval_sec)
    one_day = (3600 * 24) ÷ interval_sec
    n = one_day * n_days
    ma_vec = fill(NaN, length(feature))
    std_vec = fill(NaN, length(feature))
    for i in n+1:one_day:length(feature)
        mean_val = sum(skipnan(feature[i-one_day : i])) / one_day
        std_val = std(skipnan(feature[i-one_day : i]))
        ma_vec[i : i+one_day-1] .= mean_val
        std_vec[i : i+one_day-1] .= std_val
        # println("i: $i, mean: $mean_val, std: $std_val")
        # break
    end
    normed_vec = (feature .- ma_vec) ./ std_vec
    return normed_vec
end


function backtest_sica(
    signal_long, signal_short, entry_price_vec, exit_price_vec, timestamp_vec, keep_position_idx; 
    symbol="", name="backtest result", is_display=false,
    resampling_interval=1800,
    )        
    tr_res_vec::Vector{Tuple{Int64, Int64, Float64, Float64, Float64}} = []
    curr_pos = 0
    idx_close_pos = -1
    entry_price = 0.0

    total_len = length(signal_long)
    for idx in 1:total_len
        if isnan(entry_price_vec[idx]) || isnan(exit_price_vec[idx]) continue end
        if curr_pos !== 0
            if idx >= idx_close_pos
                d = curr_pos
                exit_price = exit_price_vec[idx]
                profit = d > 0 ? exit_price - entry_price : entry_price - exit_price
                tr_res = (idx, d, entry_price, exit_price, profit)
                push!(tr_res_vec, tr_res)
                curr_pos = 0
            end
        else
            if signal_long[idx]
                curr_pos = 1
                entry_price = entry_price_vec[idx]
                idx_close_pos = idx + keep_position_idx
            elseif signal_short[idx]
                curr_pos = -1
                entry_price = entry_price_vec[idx]
                idx_close_pos = idx + keep_position_idx
            end
        end
    end


    T = [tr_res[1] for tr_res in tr_res_vec]
    PnL = [tr_res[5] for tr_res in tr_res_vec]
    cumPnL = cumsum(PnL)
    p = plot(T, cumPnL)
    # display(p)

    full_PnL = fill(0.0, total_len)
    for (i, t) in enumerate(T)
        full_PnL[t] = PnL[i]
    end
    full_cumPnL = cumsum(full_PnL)

    win_cnt, lose_cnt = 0, 0
    win_profit_bp_vec, lose_profit_bp_vec = [], []
    win_long_profit_bp_vec, win_short_profit_bp_vec = [], []
    lose_long_profit_bp_vec, lose_short_profit_bp_vec = [], []
    for tr_res in tr_res_vec
        profit = tr_res[5]
        profit_bp = 10000 * profit / tr_res[3]
        if profit > 0
            win_cnt += 1
            push!(win_profit_bp_vec, profit_bp)
            if tr_res[2] > 0
                push!(win_long_profit_bp_vec, profit_bp)
            elseif tr_res[2] < 0
                push!(win_short_profit_bp_vec, profit_bp)
            end
        elseif profit < 0
            lose_cnt += 1
            push!(lose_profit_bp_vec, profit_bp)
            if tr_res[2] > 0
                push!(lose_long_profit_bp_vec, profit_bp)
            elseif tr_res[2] < 0
                push!(lose_short_profit_bp_vec, profit_bp)
            end
        end
    end
    win_rate, lose_rate = win_cnt / length(tr_res_vec), lose_cnt / length(tr_res_vec)
    total_profit_bp_vec = [win_profit_bp_vec; lose_profit_bp_vec]    
    avg_win_short_profit_bp = length(win_short_profit_bp_vec) == 0 ? 0.0 : round(mean(win_short_profit_bp_vec), digits=4)
    avg_lose_short_profit_bp = length(lose_short_profit_bp_vec) == 0 ? 0.0 : round(mean(lose_short_profit_bp_vec), digits=4)

    println("Win Rate: $(round(win_rate * 100, digits=2))%, Lose Rate: $(round(lose_rate * 100, digits=2))%")
    println("Mean Profig(bp) : $(round(mean(total_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) - Win: $(round(mean(win_profit_bp_vec), digits=4))\tLose: $(round(mean(lose_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) Win - Long: $(round(mean(win_long_profit_bp_vec), digits=4))\tShort: $(avg_win_short_profit_bp)")
    println("Mean Profit(bp) Lose - Long: $(round(mean(lose_long_profit_bp_vec), digits=4))\tShort: $(avg_lose_short_profit_bp)")

    show_idx = [idx for idx in 1:resampling_interval:total_len]
    date_vec = Dates.unix2datetime.([Int64(ts ÷ 1_000) for ts in timestamp_vec[show_idx]])  # 날짜만 추출
    p1 = plot(date_vec, exit_price_vec[show_idx], label="Benchmark", color=:gray)
    p2 = twinx()
    plot!(p2, date_vec, full_cumPnL[show_idx], label="CumPnL", color=:blue)
    title!("$(name)-$(symbol)")
    
    stats_text = """
        --- Backtest Summary --------------
        Equity Resample Interval: $(resampling_interval)sec
        -----------------------------------
        Total Trades:         $(length(tr_res_vec))
        Total Return (bp):    $(round(sum(total_profit_bp_vec), digits=3))
        Avg.  Return (bp):    $(round(mean(total_profit_bp_vec), digits=3))
        -----------------------------------
        Win Rate:             $(round(win_rate * 100, digits=2))%
        Profit Factor:        ???
        Avg. Win  (bp):       $(round(mean(win_profit_bp_vec), digits=4))
        Avg. Loss (bp):       $(round(mean(lose_profit_bp_vec), digits=4))
        ----------------------
        Avg. Win  Long  (bp): $(round(mean(win_long_profit_bp_vec), digits=4))
        Avg. Win  Short (bp): $(avg_win_short_profit_bp)
        Avg. Loss Long  (bp): $(round(mean(lose_long_profit_bp_vec), digits=4))
        Avg. Loss Short (bp): $(avg_lose_short_profit_bp)
        ------------------------------------
        Max Drawdown:         ???
        Sharpe Ratio:         ???
        Sortino Ratio:        ???
        ------------------------------------
        """
    p_info = plot(framestyle=:none, legend=false, xlims=(0, 1), ylims=(0, 1))
    annotate!(p_info, 0.05, 0.95, text(stats_text, :left, :top, 10, "Courier New"))
    
    # 최종 플롯은 p1을 기준으로 합칩니다. p1이 twin_p 정보를 포함하고 있습니다.
    final_plot = plot(p1, p_info, layout=(1, 2), widths=(0.7, 0.3), size=(1200, 600))

    if is_display
        display(final_plot)
    end

    return final_plot
end


function backtest_sica_keep(
    signal_long, signal_short, entry_price_vec, exit_price_vec, timestamp_vec, keep_position_idx; 
    symbol="", name="backtest result", is_display=false,
    resampling_interval=1800,
    )        
    tr_res_vec::Vector{Tuple{Int64, Int64, Int64, Float64, Float64, Float64}} = []
    curr_pos = 0
    idx_close_pos = -1
    idx_pos_entry = -1
    entry_price = 0.0

    total_len = length(signal_long)
    for idx in 1:total_len
        if isnan(entry_price_vec[idx]) || isnan(exit_price_vec[idx]) continue end
        if curr_pos !== 0
            if idx >= idx_close_pos
                d = curr_pos
                exit_price = exit_price_vec[idx]
                profit = d > 0 ? exit_price - entry_price : entry_price - exit_price
                tr_res = (idx_pos_entry, idx, d, entry_price, exit_price, profit)
                push!(tr_res_vec, tr_res)
                curr_pos = 0
            else
                if curr_pos == 1 && signal_long[idx]
                    idx_close_pos = idx + keep_position_idx
                elseif curr_pos == -1 && signal_short[idx]
                    idx_close_pos = idx + keep_position_idx
                end
            end
        else
            if signal_long[idx]
                curr_pos = 1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
                idx_close_pos = idx + keep_position_idx
            elseif signal_short[idx]
                curr_pos = -1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
                idx_close_pos = idx + keep_position_idx
            end
        end
    end


    T = [tr_res[2] for tr_res in tr_res_vec]
    PnL = [tr_res[6] for tr_res in tr_res_vec]
    cumPnL = cumsum(PnL)
    p = plot(T, cumPnL)
    # display(p)

    full_PnL = fill(0.0, total_len)
    for (i, t) in enumerate(T)
        full_PnL[t] = PnL[i]
    end
    full_cumPnL = cumsum(full_PnL)

    win_cnt, lose_cnt = 0, 0
    win_profit_bp_vec, lose_profit_bp_vec = [], []
    win_long_profit_bp_vec, win_short_profit_bp_vec = [], []
    lose_long_profit_bp_vec, lose_short_profit_bp_vec = [], []
    for tr_res in tr_res_vec
        profit = tr_res[6]
        profit_bp = 10000 * profit / tr_res[4]
        if profit > 0
            win_cnt += 1
            push!(win_profit_bp_vec, profit_bp)
            if tr_res[3] > 0
                push!(win_long_profit_bp_vec, profit_bp)
            elseif tr_res[3] < 0
                push!(win_short_profit_bp_vec, profit_bp)
            end
        elseif profit < 0
            lose_cnt += 1
            push!(lose_profit_bp_vec, profit_bp)
            if tr_res[3] > 0
                push!(lose_long_profit_bp_vec, profit_bp)
            elseif tr_res[3] < 0
                push!(lose_short_profit_bp_vec, profit_bp)
            end
        end
    end
    win_rate, lose_rate = win_cnt / length(tr_res_vec), lose_cnt / length(tr_res_vec)
    total_profit_bp_vec = [win_profit_bp_vec; lose_profit_bp_vec]
    avg_win_short_profit_bp = length(win_short_profit_bp_vec) == 0 ? 0.0 : round(mean(win_short_profit_bp_vec), digits=4)
    avg_lose_short_profit_bp = length(lose_short_profit_bp_vec) == 0 ? 0.0 : round(mean(lose_short_profit_bp_vec), digits=4)

    println("Win Rate: $(round(win_rate * 100, digits=2))%, Lose Rate: $(round(lose_rate * 100, digits=2))%")
    println("Mean Profig(bp) : $(round(mean(total_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) - Win: $(round(mean(win_profit_bp_vec), digits=4))\tLose: $(round(mean(lose_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) Win - Long: $(round(mean(win_long_profit_bp_vec), digits=4))\tShort: $(avg_win_short_profit_bp)")
    println("Mean Profit(bp) Lose - Long: $(round(mean(lose_long_profit_bp_vec), digits=4))\tShort: $(avg_lose_short_profit_bp)")

    show_idx = [idx for idx in 1:resampling_interval:total_len]
    date_vec = Dates.unix2datetime.([Int64(ts ÷ 1_000) for ts in timestamp_vec[show_idx]])  # 날짜만 추출
    p1 = plot(date_vec, exit_price_vec[show_idx], label="Benchmark", color=:gray)
    p2 = twinx()
    plot!(p2, date_vec, full_cumPnL[show_idx], label="CumPnL", color=:blue)
    title!("$(name)-$(symbol)")
    
    stats_text = """
        --- Backtest Summary --------------
        Equity Resample Interval: $(resampling_interval)sec
        -----------------------------------
        Total Trades:         $(length(tr_res_vec))
        Total Return (bp):    $(round(sum(total_profit_bp_vec), digits=3))
        Avg.  Return (bp):    $(round(mean(total_profit_bp_vec), digits=3))
        -----------------------------------
        Win Rate:             $(round(win_rate * 100, digits=2))%
        Profit Factor:        ???
        Avg. Win  (bp):       $(round(mean(win_profit_bp_vec), digits=4))
        Avg. Loss (bp):       $(round(mean(lose_profit_bp_vec), digits=4))
        ----------------------
        Avg. Win  Long  (bp): $(round(mean(win_long_profit_bp_vec), digits=4))
        Avg. Win  Short (bp): $(avg_win_short_profit_bp)
        Avg. Loss Long  (bp): $(round(mean(lose_long_profit_bp_vec), digits=4))
        Avg. Loss Short (bp): $(avg_lose_short_profit_bp)
        ------------------------------------
        Max Drawdown:         ???
        Sharpe Ratio:         ???
        Sortino Ratio:        ???
        ------------------------------------
        """
    p_info = plot(framestyle=:none, legend=false, xlims=(0, 1), ylims=(0, 1))
    annotate!(p_info, 0.05, 0.95, text(stats_text, :left, :top, 10, "Courier New"))
    
    # 최종 플롯은 p1을 기준으로 합칩니다. p1이 twin_p 정보를 포함하고 있습니다.
    final_plot = plot(p1, p_info, layout=(1, 2), widths=(0.7, 0.3), size=(1200, 600))

    if is_display
        display(final_plot)
    end

    return final_plot, tr_res_vec
end


function backtest_sica_2(
    signal_long, signal_short, entry_price_vec, exit_price_vec, timestamp_vec; 
    symbol="", name="backtest result", is_display=false,
    resampling_interval=1800, max_duration=Inf, is_print=true
    )        
    # signal_long = aq_rb .>= quantile(aq_rb[.!isnan.(aq_rb)], 0.9)
    # signal_short = aq_rb .<= quantile(aq_rb[.!isnan.(aq_rb)], 0.1)
    # entry_price_vec = df.WAP_Lag_300ms
    # exit_price_vec = df.WAP_Lag_0ms

    tr_res_vec::Vector{Tuple{Int64, Int64, Int64, Float64, Float64, Float64}} = []
    curr_pos = 0
    idx_pos_entry = -1
    entry_price = 0.0

    total_len = length(signal_long)
    for idx in 1:total_len
        if isnan(entry_price_vec[idx]) || isnan(exit_price_vec[idx]) || (idx - idx_pos_entry >= max_duration) continue end
        if curr_pos !== 0
            if (curr_pos > 0 && signal_short[idx]) || (curr_pos < 0 && signal_long[idx])
                d = curr_pos
                exit_price = exit_price_vec[idx]
                profit = d > 0 ? exit_price - entry_price : entry_price - exit_price
                tr_res = (idx_pos_entry, idx, d, entry_price, exit_price, profit)
                push!(tr_res_vec, tr_res)
                curr_pos = 0
            end
        else
            if signal_long[idx]
                curr_pos = 1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
            elseif signal_short[idx]
                curr_pos = -1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
            end
        end
    end


    T = [tr_res[2] for tr_res in tr_res_vec]
    PnL = [tr_res[6] for tr_res in tr_res_vec]
    cumPnL = cumsum(PnL)
    p = plot(T, cumPnL)
    # display(p)

    full_PnL = fill(0.0, total_len)
    for (i, t) in enumerate(T)
        full_PnL[t] = PnL[i]
    end
    full_cumPnL = cumsum(full_PnL)

    win_cnt, lose_cnt = 0, 0
    total_profit_bp_vec = []
    win_profit_bp_vec, lose_profit_bp_vec = [], []
    win_long_profit_bp_vec, win_short_profit_bp_vec = [], []
    lose_long_profit_bp_vec, lose_short_profit_bp_vec = [], []
    for tr_res in tr_res_vec
        profit = tr_res[6]
        profit_bp = 10000 * profit / tr_res[4]
        if profit > 0
            win_cnt += 1
            push!(win_profit_bp_vec, profit_bp)
            push!(total_profit_bp_vec, profit_bp)
            if tr_res[3] > 0
                push!(win_long_profit_bp_vec, profit_bp)
            elseif tr_res[3] < 0
                push!(win_short_profit_bp_vec, profit_bp)
            end
        elseif profit < 0
            lose_cnt += 1
            push!(lose_profit_bp_vec, profit_bp)
            push!(total_profit_bp_vec, profit_bp)
            if tr_res[3] > 0
                push!(lose_long_profit_bp_vec, profit_bp)
            elseif tr_res[3] < 0
                push!(lose_short_profit_bp_vec, profit_bp)
            end
        end
    end
    win_rate, lose_rate = win_cnt / length(tr_res_vec), lose_cnt / length(tr_res_vec)
    avg_win_long = length(win_long_profit_bp_vec) == 0 ? 0.0 : round(mean(win_long_profit_bp_vec), digits=4)
    avg_win_short = length(win_short_profit_bp_vec) == 0 ? 0.0 : round(mean(win_short_profit_bp_vec), digits=4)
    avg_lose_long = length(lose_long_profit_bp_vec) == 0 ? 0.0 : round(mean(lose_long_profit_bp_vec), digits=4)
    avg_lose_short = length(lose_short_profit_bp_vec) == 0 ? 0.0 : round(mean(lose_short_profit_bp_vec), digits=4)
    if is_print
        println("Win Rate: $(round(win_rate * 100, digits=2))%, Lose Rate: $(round(lose_rate * 100, digits=2))%")
        println("Mean Profit(bp) : $(round(mean(total_profit_bp_vec), digits=4))")
        println("Mean Profit(bp) - Win: $(round(mean(win_profit_bp_vec), digits=4))\tLose: $(round(mean(lose_profit_bp_vec), digits=4))")
        println("Mean Profit(bp) Win - Long: $(avg_win_long)\tShort: $(avg_win_short)")
        println("Mean Profit(bp) Lose - Long: $(avg_lose_long)\tShort: $(avg_lose_short)")
    end

    show_idx = [idx for idx in 1:resampling_interval:total_len]
    date_vec = Dates.unix2datetime.([Int64(ts ÷ 1_000) for ts in timestamp_vec[show_idx]])  # 날짜만 추출
    p1 = plot(date_vec, exit_price_vec[show_idx], label="Benchmark", color=:gray)
    p2 = twinx()
    plot!(p2, date_vec, full_cumPnL[show_idx], label="CumPnL", color=:blue)
    title!("$(name)-$(symbol)")

    total_profit_bp_vec = [win_profit_bp_vec; lose_profit_bp_vec]
    # running_max = accumulate(max, cumPnL)
    # drawdowns = (running_max .- cumPnL) ./ running_max
    # max_drawdown = maximum(drawdowns)
    
    stats_text = """
        --- Backtest Summary --------------
        Equity Resample Interval: $(resampling_interval)sec
        -----------------------------------
        Total Trades:         $(length(tr_res_vec))
        Total Return (bp):    $(round(sum(total_profit_bp_vec), digits=4))
        Avg.  Return (bp):    $(round(mean(total_profit_bp_vec), digits=4))
        -----------------------------------
        Win Rate:             $(round(win_rate * 100, digits=2))%
        Profit Factor:        ???
        Avg. Win  (bp):       $(round(mean(win_profit_bp_vec), digits=4))
        Avg. Loss (bp):       $(round(mean(lose_profit_bp_vec), digits=4))
        ----------------------
        Avg. Win  Long  (bp): $(avg_win_long)
        Avg. Win  Short (bp): $(avg_win_short)
        Avg. Loss Long  (bp): $(avg_lose_long)
        Avg. Loss Short (bp): $(avg_lose_short)
        ------------------------------------
        Max Drawdown:         ???
        Sharpe Ratio:         ???
        Sortino Ratio:        ???
        ------------------------------------
        """
    p_info = plot(framestyle=:none, legend=false, xlims=(0, 1), ylims=(0, 1))
    annotate!(p_info, 0.05, 0.95, text(stats_text, :left, :top, 10, "Courier New"))
    
    # 최종 플롯은 p1을 기준으로 합칩니다. p1이 twin_p 정보를 포함하고 있습니다.
    final_plot = plot(p1, p_info, layout=(1, 2), widths=(0.7, 0.3), size=(1200, 600))

    if is_display
        display(final_plot)
    end

    return final_plot, tr_res_vec
end



function backtest_sica_3(
    signal_long_entry, signal_long_exit, signal_short_entry, signal_short_exit, entry_price_vec, exit_price_vec, timestamp_vec; 
    symbol="", name="backtest result", is_display=false,
    resampling_interval=1800, 
    # max_duration=Inf, min_duration=0,
    )

    tr_res_vec::Vector{Tuple{Int64, Int64, Int64, Float64, Float64, Float64}} = []
    curr_pos = 0
    idx_pos_entry = -1
    entry_price = 0.0

    total_len = length(signal_long_entry)
    for idx in 1:total_len
        if isnan(entry_price_vec[idx]) || isnan(exit_price_vec[idx]) continue end
        if curr_pos !== 0
            if (curr_pos < 0 && signal_short_exit[idx]) || (curr_pos > 0 && signal_long_exit[idx])
                d = curr_pos
                exit_price = exit_price_vec[idx]
                profit = d > 0 ? exit_price - entry_price : entry_price - exit_price
                tr_res = (idx_pos_entry, idx, d, entry_price, exit_price, profit)
                push!(tr_res_vec, tr_res)
                curr_pos = 0
            end
        else
            if signal_long_entry[idx]
                curr_pos = 1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
            elseif signal_short_entry[idx]
                curr_pos = -1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
            end
        end
    end


    T = [tr_res[2] for tr_res in tr_res_vec]
    PnL = [tr_res[6] for tr_res in tr_res_vec]
    cumPnL = cumsum(PnL)
    p = plot(T, cumPnL)
    # display(p)

    full_PnL = fill(0.0, total_len)
    for (i, t) in enumerate(T)
        full_PnL[t] = PnL[i]
    end
    full_cumPnL = cumsum(full_PnL)

    win_cnt, lose_cnt = 0, 0
    total_profit_bp_vec = []
    win_profit_bp_vec, lose_profit_bp_vec = [], []
    win_long_profit_bp_vec, win_short_profit_bp_vec = [], []
    lose_long_profit_bp_vec, lose_short_profit_bp_vec = [], []
    for tr_res in tr_res_vec
        profit = tr_res[6]
        profit_bp = 10000 * profit / tr_res[4]
        if profit > 0
            win_cnt += 1
            push!(win_profit_bp_vec, profit_bp)
            push!(total_profit_bp_vec, profit_bp)
            if tr_res[3] > 0
                push!(win_long_profit_bp_vec, profit_bp)
            elseif tr_res[3] < 0
                push!(win_short_profit_bp_vec, profit_bp)
            end
        elseif profit < 0
            lose_cnt += 1
            push!(lose_profit_bp_vec, profit_bp)
            push!(total_profit_bp_vec, profit_bp)
            if tr_res[3] > 0
                push!(lose_long_profit_bp_vec, profit_bp)
            elseif tr_res[3] < 0
                push!(lose_short_profit_bp_vec, profit_bp)
            end
        end
    end
    win_rate, lose_rate = win_cnt / length(tr_res_vec), lose_cnt / length(tr_res_vec)
    avg_win_long = length(win_long_profit_bp_vec) == 0 ? 0.0 : round(mean(win_long_profit_bp_vec), digits=4)
    avg_win_short = length(win_short_profit_bp_vec) == 0 ? 0.0 : round(mean(win_short_profit_bp_vec), digits=4)
    avg_lose_long = length(lose_long_profit_bp_vec) == 0 ? 0.0 : round(mean(lose_long_profit_bp_vec), digits=4)
    avg_lose_short = length(lose_short_profit_bp_vec) == 0 ? 0.0 : round(mean(lose_short_profit_bp_vec), digits=4)

    println("Win Rate: $(round(win_rate * 100, digits=2))%, Lose Rate: $(round(lose_rate * 100, digits=2))%")
    println("Mean Profig(bp) : $(round(mean(total_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) - Win: $(round(mean(win_profit_bp_vec), digits=4))\tLose: $(round(mean(lose_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) Win - Long: $(avg_win_long)\tShort: $(avg_win_short)")
    println("Mean Profit(bp) Lose - Long: $(avg_lose_long)\tShort: $(avg_lose_short)")

    show_idx = [idx for idx in 1:resampling_interval:total_len]
    date_vec = Dates.unix2datetime.([Int64(ts ÷ 1_000) for ts in timestamp_vec[show_idx]])  # 날짜만 추출
    p1 = plot(date_vec, exit_price_vec[show_idx], label="Benchmark", color=:gray)
    p2 = twinx()
    plot!(p2, date_vec, full_cumPnL[show_idx], label="CumPnL", color=:blue)
    title!("$(name)-$(symbol)")

    total_profit_bp_vec = [win_profit_bp_vec; lose_profit_bp_vec]
    # running_max = accumulate(max, cumPnL)
    # drawdowns = (running_max .- cumPnL) ./ running_max
    # max_drawdown = maximum(drawdowns)
    
    stats_text = """
        --- Backtest Summary --------------
        Equity Resample Interval: $(resampling_interval)sec
        -----------------------------------
        Total Trades:         $(length(tr_res_vec))
        Total Return (bp):    $(round(sum(total_profit_bp_vec), digits=4))
        Avg.  Return (bp):    $(round(mean(total_profit_bp_vec), digits=4))
        -----------------------------------
        Win Rate:             $(round(win_rate * 100, digits=2))%
        Profit Factor:        ???
        Avg. Win  (bp):       $(round(mean(win_profit_bp_vec), digits=4))
        Avg. Loss (bp):       $(round(mean(lose_profit_bp_vec), digits=4))
        ----------------------
        Avg. Win  Long  (bp): $(avg_win_long)
        Avg. Win  Short (bp): $(avg_win_short)
        Avg. Loss Long  (bp): $(avg_lose_long)
        Avg. Loss Short (bp): $(avg_lose_short)
        ------------------------------------
        Max Drawdown:         ???
        Sharpe Ratio:         ???
        Sortino Ratio:        ???
        ------------------------------------
        """
    p_info = plot(framestyle=:none, legend=false, xlims=(0, 1), ylims=(0, 1))
    annotate!(p_info, 0.05, 0.95, text(stats_text, :left, :top, 10, "Courier New"))
    
    # 최종 플롯은 p1을 기준으로 합칩니다. p1이 twin_p 정보를 포함하고 있습니다.
    final_plot = plot(p1, p_info, layout=(1, 2), widths=(0.7, 0.3), size=(1200, 600))

    if is_display
        display(final_plot)
    end

    return final_plot, tr_res_vec
end


function get_signals(f, th_pct1; th_pct2=0.0)
    valid_mask = .!isnan.(f) .& isfinite.(f)
    th_long1 = quantile(f[valid_mask], (100 - th_pct1) / 100)
    th_long2 = quantile(f[valid_mask], (100 - th_pct2) / 100)
    th_short1 = quantile(f[valid_mask], th_pct1 / 100)  
    th_short2 = quantile(f[valid_mask], th_pct2 / 100)
    signal_long = th_long1 .<= f .<= th_long2
    signal_short = th_short2 .<= f .<= th_short1
    return signal_long, signal_short
end


function read_zstd_file_one(path)
    df = open(path, "r") do f
        io = ZstdDecompressorStream(f)
        try 
            deserialize(io) 
        finally 
            close(io) 
        end
    end
    return df
end


function data_injection(df, df_source, cols)
    tsv = Int64.(df.timestamp .* 1000)
    idx_wap, idx_source = 1, 1

    while size(df, 1) >= idx_wap && size(df_source, 1) >= idx_source
        t_wap = tsv[idx_wap]
        t_source = df_source.local_timestamp[idx_source]

        if t_wap < t_source
            if idx_source > 1
                idx_copy = idx_source - 1
                for col in cols
                    df[idx_wap, col] = ismissing(df_source[idx_copy, col]) ? NaN : df_source[idx_copy, col]
                end
            end
            idx_wap += 1
        else
            idx_source += 1
        end
    end
end


function set_ret_bp(df, ret_interval)
    fut_wap = [df.WAP_Lag_0ms[ret_interval+1 : end]; fill(NaN, ret_interval)]
    ret_bp = 10_000 .* (fut_wap .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms
    df.ret_bp = ret_bp
end


function get_df_oneday_full(tardis_dir, s7_dir, symbol, date, features)
    # println("$(symbol) - $(date)")
    deri_dir = joinpath(tardis_dir, "binance-futures", "derivative_ticker", symbol)
    liqu_dir = joinpath(tardis_dir, "binance-futures", "liquidations", symbol)

    date_str = "$(date[1:4])-$(date[5:6])-$(date[7:8])"
    deri_path = joinpath(deri_dir, "$(date_str)_$(symbol).csv.gz")
    df_deri = CSV.read(GzipDecompressorStream(open(deri_path)), DataFrame)

    liqu_path = joinpath(liqu_dir, "$(date_str)_$(symbol).csv.gz")
    df_liqu = CSV.read(GzipDecompressorStream(open(liqu_path)), DataFrame)

    deri_cols = [:open_interest, :index_price, :mark_price]
    liqu_cols = [:side, :price, :amount]
    
    df_features = []
    for (feature, cols) in features
        feature_path = joinpath(s7_dir, "binance-futures", symbol, date, "$(feature).df.zst")
        df_feature_all = read_zstd_file_one(feature_path)
        df_feature = df_feature_all[!, cols]
        push!(df_features, df_feature)
    end

    df = deepcopy(df_features[1])
    for df_feature in df_features
        for col in names(df_feature)
            df[!, col] = df_feature[!, col]
        end
    end

    for col in [deri_cols; liqu_cols]
        if col == :side
            df[!, col] .= ""
        else
            df[!, col] .= NaN
        end
    end

    data_injection(df, df_deri, deri_cols)
    data_injection(df, df_liqu, liqu_cols)
    return df
end



function get_df_oneday_with_deri_liqu(tardis_dir, s7_dir, symbol, date)
    deri_dir = joinpath(tardis_dir, "binance-futures", "derivative_ticker", symbol)
    liqu_dir = joinpath(tardis_dir, "binance-futures", "liquidations", symbol)

    date_str = "$(date[1:4])-$(date[5:6])-$(date[7:8])"
    deri_path = joinpath(deri_dir, "$(date_str)_$(symbol).csv.gz")
    df_deri = CSV.read(GzipDecompressorStream(open(deri_path)), DataFrame)

    liqu_path = joinpath(liqu_dir, "$(date_str)_$(symbol).csv.gz")
    df_liqu = CSV.read(GzipDecompressorStream(open(liqu_path)), DataFrame)

    wap_0ms_path = joinpath(s7_dir, "binance-futures", symbol, date, "WAP_Lag_0ms.df.zst")
    wap_200ms_path = joinpath(s7_dir, "binance-futures", symbol, date, "WAP_Lag_200ms.df.zst")

    df_0ms = read_zstd_file_one(wap_0ms_path)
    df_200ms = read_zstd_file_one(wap_200ms_path)

    deri_cols = [:open_interest, :index_price, :mark_price]
    liqu_cols = [:side, :price, :amount]

    df = deepcopy(df_0ms)
    df.WAP_Lag_200ms = df_200ms.WAP_Lag_200ms
    fut_wap = [df.WAP_Lag_0ms[15 + 1 : end]; fill(NaN, 15)]
    df.ret_bp = 10_000 .* (fut_wap .- df.WAP_Lag_200ms) ./ df.WAP_Lag_200ms
    # df.ret_bp = 10_000 .* (fut_wap .- df.WAP_Lag_0ms) ./ df.WAP_Lag_0ms

    for col in [deri_cols; liqu_cols]
        if col == :side
            df[!, col] .= ""
        else
            df[!, col] .= NaN
        end
    end

    data_injection(df, df_deri, deri_cols)
    data_injection(df, df_liqu, liqu_cols)
    return df
end


function get_last_non_nan_idx(data)
    idx_end = length(data)
    while idx_end > 0
        if data[idx_end] !== NaN return idx_end end
        idx_end -= 1
    end
    return idx_end
end

function get_norm_ret_bp_path(tr_res_vec, wap)
    directions = []
    wap_path = []
    for tr_res in tr_res_vec
        i0, i1, direction = tr_res[1:3]
        wp = wap[i0:i1]
        push!(wap_path, wp)
        push!(directions, direction > 0)
    end

    norm_paths = [fill(NaN, 100) for i in 1:length(wap_path)]
    for (i, wp) in enumerate(wap_path)
        is_long = directions[i]
        l = length(wp)
        ii = l >= 100 ? (1 : l÷99: l) : (1 : l)
        for (i2, ii2)in enumerate(ii)
            if i2 > 100 break end
            diff = is_long ? wp[ii2] - wp[1] : wp[1] - wp[ii2]
            norm_paths[i][i2] = 10_000 * diff / wp[1]
        end
        diff = is_long ? wp[end] - wp[1] : wp[1] - wp[end]
        idx_end = get_last_non_nan_idx(norm_paths[i])
        norm_paths[i][idx_end] = 10_000 * diff / wp[1]
    end

    avg_norm_paths = fill(NaN, 100)
    median_norm_paths = fill(NaN, 100)
    for i in 1:100
        avg_norm_paths[i] = mean(skipnan(norm_paths[i2][i] for i2 in 1:length(norm_paths)))
        median_norm_paths[i] = median(skipnan(norm_paths[i2][i] for i2 in 1:length(norm_paths)))
    end
    return avg_norm_paths, median_norm_paths
end


function get_norm_feature_path(tr_res_vec, feature)
    wap_path = []
    for tr_res in tr_res_vec
        i0, i1 = tr_res[1:2]
        wp = feature[i0:i1]
        push!(wap_path, wp)
    end

    norm_paths = [fill(NaN, 100) for i in 1:length(wap_path)]
    for (i, wp) in enumerate(wap_path)
        l = length(wp)
        ii = l >= 100 ? (1 : l÷99: l) : (1 : l)
        for (i2, ii2)in enumerate(ii)
            if i2 > 100 break end
            diff = wp[ii2] - wp[1]
            norm_paths[i][i2] = diff
        end
        diff = wp[end] - wp[1]
        idx_end = get_last_non_nan_idx(norm_paths[i])
        norm_paths[i][idx_end] = diff
    end

    avg_norm_paths = fill(NaN, 100)
    median_norm_paths = fill(NaN, 100)
    for i in 1:100
        avg_norm_paths[i] = mean(skipnan(norm_paths[i2][i] for i2 in 1:length(norm_paths)))
        median_norm_paths[i] = median(skipnan(norm_paths[i2][i] for i2 in 1:length(norm_paths)))
    end
    return avg_norm_paths, median_norm_paths
end


function get_full_ret_bp_path(tr_res_vec, wap)
    directions = []
    wap_path = []
    for tr_res in tr_res_vec
        i0, i1, direction = tr_res[1:3]
        wp = wap[i0:i1]
        push!(wap_path, wp)
        push!(directions, direction > 0)
    end

    max_l = maximum([length(wp) for wp in wap_path])
    pr_paths = [fill(NaN, max_l) for i in 1:length(wap_path)]
    for (i, wp) in enumerate(wap_path)
        is_long = directions[i]
        for (ii2, wpi) in enumerate(wp)
            diff = is_long ? wpi - wp[1] : wp[1] - wpi
            pr_paths[i][ii2] = 10_000 * diff / wp[1]
        end
    end

    avg_pr_paths = fill(NaN, max_l)
    median_pr_paths = fill(NaN, max_l)
    remainer = fill(0, max_l)
    for i in 1:max_l
        try
            avg_pr_paths[i] = mean(skipnan(pr_paths[i2][i] for i2 in 1:length(pr_paths)))
            median_pr_paths[i] = median(skipnan(pr_paths[i2][i] for i2 in 1:length(pr_paths)))
            remainer[i] = sum(.!isnan.([pr_paths[i2][i] for i2 in 1:length(pr_paths)]))
        catch
            println(i)
        end
    end
    return avg_pr_paths, median_pr_paths, remainer
end


function get_first_idx(data, target)
    for (i, v) in enumerate(data)
        if v == target return i end
    end
    return -1
end


function get_all_targeted_idx(data, target)
    res = []
    for (i, v) in enumerate(data)
        if v == target push!(res, i) end
    end
    return res
end


function get_at_time_profile(ft, wap, win, idx_at_list)
    fvs2d = [fill(NaN, 2*win+1) for i in 1:length(idx_at_list)]
    waps2d = [fill(NaN, 2*win+1) for i in 1:length(idx_at_list)]
    for (i, idx_at) in enumerate(idx_at_list)
        i0 = idx_at - win
        i1 = idx_at + win
        if i0 < 1 || i1 > length(wap) continue end
        fvs2d[i] .= (ft[i0 : i1] .- ft[idx_at])
        waps2d[i] .= 10_000 .* (wap[i0 : i1] .- wap[idx_at]) ./ wap[idx_at]
    end

    avg_fvs, avg_waps = fill(NaN, 2*win+1), fill(NaN, 2*win+1)
    median_fvs, median_waps = fill(NaN, 2*win+1), fill(NaN, 2*win+1)
    for i in 1 : 2*win+1
        avg_fvs[i] = mean(skipnan(fvs2d[i2][i] for i2 in 1:length(idx_at_list)))
        avg_waps[i] = mean(skipnan(waps2d[i2][i] for i2 in 1:length(idx_at_list)))
        median_fvs[i] = median(skipnan(fvs2d[i2][i] for i2 in 1:length(idx_at_list)))
        median_waps[i] = median(skipnan(waps2d[i2][i] for i2 in 1:length(idx_at_list)))
    end
    return avg_fvs, avg_waps, median_fvs, median_waps
end


function get_at_time_profile2(ft, wap, win, idx_at_list)
    fvs2d = [fill(NaN, length(idx_at_list)) for i in 1:2*win+1]
    waps2d = [fill(NaN, length(idx_at_list)) for i in 1:2*win+1]
    for (i, idx_at) in enumerate(idx_at_list)
        i0 = idx_at - win
        i1 = idx_at + win
        if i0 < 1 || i1 > length(wap) continue end
        for ii2 in i0:i1
            i2 = 1 + ii2 - i0
            fvs2d[i2][i] = (ft[ii2] - ft[idx_at])
            waps2d[i2][i] = 10_000 * (wap[ii2] - wap[idx_at]) / wap[idx_at]
        end
    end

    avg_fvs, avg_waps = fill(NaN, 2*win+1), fill(NaN, 2*win+1)
    median_fvs, median_waps = fill(NaN, 2*win+1), fill(NaN, 2*win+1)
    for i in 1 : 2*win+1
        avg_fvs[i] = mean(skipnan(fvs2d[i][i2] for i2 in 1:length(idx_at_list)))
        avg_waps[i] = mean(skipnan(waps2d[i][i2] for i2 in 1:length(idx_at_list)))
        median_fvs[i] = median(skipnan(fvs2d[i][i2] for i2 in 1:length(idx_at_list)))
        median_waps[i] = median(skipnan(waps2d[i][i2] for i2 in 1:length(idx_at_list)))
    end
    return avg_fvs, avg_waps, median_fvs, median_waps
end


@memoize function get_quantile_value(sorted_x, th)
    # @assert 0.0 <= th <= 100
    idx = 1 + Int64(((length(sorted_x) - 1) * (th / 100.0)) ÷ 1)
    # @assert idx <= length(sorted_x)
    sub_idx = ((length(sorted_x) - 1) * (th / 100.0)) % 1.0
    # @assert 0.0 <= sub_idx
    quantile_value = if idx == length(sorted_x)
        sorted_x[idx]
    else
        (1.0 - sub_idx) * sorted_x[idx] + sub_idx * sorted_x[idx+1]
    end
    return quantile_value
end


function get_ev_info(X, ret_bp, th_vec)
    mask = .!isnan.(X) .& .!isnan.(ret_bp)
    sorted_x = sort(X[mask])
    qvv = [get_quantile_value(sorted_x, th) for th in th_vec]
    tb_info = []
    for (qv1, qv2) in zip(qvv[1:end-1], qvv[2:end])
        msk = qv1 .<= X .<= qv2
        avg = mean(skipnan(ret_bp[msk]))
        push!(tb_info, (qv1, qv2, avg))
    end
    return tb_info
end



"""
바이너리 서치를 이용해 tb_info에서 tbi[1] <= xi < tbi[2]를 만족하는 인덱스를 찾는 함수
tb_info: [(start, end, value), ...] 형태의 배열 (오름차순 정렬됨)
xi: 찾고자 하는 값
반환값: 조건을 만족하는 인덱스 (1-based), 
        xi < tb_info[1][1]이면 1, 
        xi >= tb_info[end][2]이면 length(tb_info)
"""
function find_interval_index(tb_info, xi)
    # 경계 조건 체크
    if xi < tb_info[1][1]
        return 1
    end
    if xi >= tb_info[end][2]
        return length(tb_info)
    end
    
    left = 1
    right = length(tb_info)
    
    while left <= right
        mid = (left + right) ÷ 2
        tbi = tb_info[mid]
        
        if tbi[1] <= xi < tbi[2]
            return mid
        elseif xi < tbi[1]
            right = mid - 1
        else  # xi >= tbi[2]
            left = mid + 1
        end
    end
    
    # 이 지점에 도달하면 안됨 (이미 경계 조건을 체크했으므로)
    error("Unexpected case in find_interval_index")
end


function get_wsv(tb_info, xi)
    if isnan(xi) return NaN end
    ii = find_interval_index(tb_info, xi)
    i1, i2 = if ii == 1
        (1, 2)
    elseif ii == length(tb_info)
        (length(tb_info) - 1, length(tb_info))
    else
        mid = (tb_info[ii][1] + tb_info[ii][2]) / 2.0
        if xi < mid
            (ii - 1, ii)
        else
            (ii, ii + 1)
        end
    end

    dist1 = abs((tb_info[i1][1] + tb_info[i1][2]) / 2.0 - xi)
    dist2 = abs((tb_info[i2][1] + tb_info[i2][2]) / 2.0 - xi)
    wsv = (dist2 * tb_info[i1][3] + dist1 * tb_info[i2][3]) / (dist1 + dist2)
    return wsv
end

"""
벡터화된 get_wsv 함수 - 대폭 성능 개선
"""
function get_wsv_vectorized(tb_info, X_vec)
    n = length(X_vec)
    result = Vector{Float64}(undef, n)
    
    # tb_info를 배열로 미리 변환 (접근 속도 향상)
    n_bins = length(tb_info)
    starts = [tb_info[i][1] for i in 1:n_bins]
    ends = [tb_info[i][2] for i in 1:n_bins]
    values = [tb_info[i][3] for i in 1:n_bins]
    mids = (starts .+ ends) ./ 2.0
    
    for i in 1:n
        xi = X_vec[i]
        if isnan(xi)
            result[i] = NaN
            continue
        end
        
        # 바이너리 서치로 구간 찾기
        ii = if xi < starts[1]
            1
        elseif xi >= ends[end]
            n_bins
        else
            # 벡터화된 searchsortedfirst 사용
            searchsortedfirst(starts, xi) - 1
            if searchsortedfirst(starts, xi) > n_bins
                n_bins
            else
                idx = searchsortedfirst(starts, xi)
                if idx > 1 && xi < ends[idx-1]
                    idx - 1
                else
                    idx > n_bins ? n_bins : idx
                end
            end
        end
        
        # 보간할 두 인덱스 결정
        i1, i2 = if ii == 1
            (1, 2)
        elseif ii == n_bins
            (n_bins - 1, n_bins)
        else
            xi < mids[ii] ? (ii - 1, ii) : (ii, ii + 1)
        end
        
        # 거리 기반 가중 평균
        dist1 = abs(mids[i1] - xi)
        dist2 = abs(mids[i2] - xi)
        
        if dist1 + dist2 == 0.0
            result[i] = values[i1]
        else
            result[i] = (dist2 * values[i1] + dist1 * values[i2]) / (dist1 + dist2)
        end
    end
    
    return result
end

"""
더 빠른 벡터화 버전 - searchsorted 활용
"""
function get_wsv_fast(tb_info, X_vec)
    n = length(X_vec)
    result = Vector{Float64}(undef, n)
    
    # tb_info를 배열로 변환
    n_bins = length(tb_info)
    starts = [tb_info[i][1] for i in 1:n_bins]
    ends = [tb_info[i][2] for i in 1:n_bins]
    values = [tb_info[i][3] for i in 1:n_bins]
    mids = (starts .+ ends) ./ 2.0
    
    @inbounds for i in 1:n
        xi = X_vec[i]
        if isnan(xi)
            result[i] = NaN
            continue
        end
        
        # 효율적인 구간 찾기
        if xi < starts[1]
            ii = 1
        elseif xi >= ends[end]
            ii = n_bins
        else
            # 시작점들 중에서 xi보다 작거나 같은 마지막 인덱스
            ii = searchsortedlast(starts, xi)
            if ii == 0 || xi >= ends[ii]
                ii = min(ii + 1, n_bins)
            end
        end
        
        # 보간
        i1, i2 = if ii == 1
            (1, min(2, n_bins))
        elseif ii == n_bins
            (max(n_bins - 1, 1), n_bins)
        else
            xi < mids[ii] ? (max(ii - 1, 1), ii) : (ii, min(ii + 1, n_bins))
        end
        
        dist1 = abs(mids[i1] - xi)
        dist2 = abs(mids[i2] - xi)
        
        if dist1 + dist2 == 0.0
            result[i] = values[i1]
        else
            result[i] = (dist2 * values[i1] + dist1 * values[i2]) / (dist1 + dist2)
        end
    end
    
    return result
end

"""
멀티쓰레드 병렬 처리 버전
"""
function get_wsv_parallel(tb_info, X_vec; chunk_size=10000)
    n = length(X_vec)
    result = Vector{Float64}(undef, n)
    
    # tb_info를 배열로 변환 (모든 쓰레드에서 공유)
    n_bins = length(tb_info)
    starts = [tb_info[i][1] for i in 1:n_bins]
    ends = [tb_info[i][2] for i in 1:n_bins]
    values = [tb_info[i][3] for i in 1:n_bins]
    mids = (starts .+ ends) ./ 2.0
    
    # 청크 단위로 분할하여 병렬 처리
    @threads for chunk_start in 1:chunk_size:n
        chunk_end = min(chunk_start + chunk_size - 1, n)
        
        @inbounds for i in chunk_start:chunk_end
            xi = X_vec[i]
            if isnan(xi)
                result[i] = NaN
                continue
            end
            
            # 효율적인 구간 찾기
            if xi < starts[1]
                ii = 1
            elseif xi >= ends[end]
                ii = n_bins
            else
                ii = searchsortedlast(starts, xi)
                if ii == 0 || xi >= ends[ii]
                    ii = min(ii + 1, n_bins)
                end
            end
            
            # 보간
            i1, i2 = if ii == 1
                (1, min(2, n_bins))
            elseif ii == n_bins
                (max(n_bins - 1, 1), n_bins)
            else
                xi < mids[ii] ? (max(ii - 1, 1), ii) : (ii, min(ii + 1, n_bins))
            end
            
            dist1 = abs(mids[i1] - xi)
            dist2 = abs(mids[i2] - xi)
            
            if dist1 + dist2 == 0.0
                result[i] = values[i1]
            else
                result[i] = (dist2 * values[i1] + dist1 * values[i2]) / (dist1 + dist2)
            end
        end
    end
    
    return result
end

"""
더 세밀한 병렬 처리 버전 - pmap 활용
"""
function get_wsv_distributed(tb_info, X_vec; n_chunks=nothing)
    n = length(X_vec)
    
    # 기본적으로 CPU 코어 수만큼 청크 분할
    if n_chunks === nothing
        n_chunks = min(Threads.nthreads(), 16)  # 최대 16개 청크
    end
    
    chunk_size = div(n, n_chunks)
    ranges = [i:min(i+chunk_size-1, n) for i in 1:chunk_size:n]
    
    # 각 청크를 병렬로 처리
    results = Vector{Vector{Float64}}(undef, length(ranges))
    
    @threads for (idx, range) in collect(enumerate(ranges))
        results[idx] = get_wsv_fast(tb_info, X_vec[range])
    end
    
    # 결과 합치기
    return vcat(results...)
end


function get_evv_df(ft_gen_map, ret_col_name, th_vec)
    ft_map = Dict()
    for (ft_name, ft_generator) in ft_gen_map
        # println("generate $(ft_name)")
        ft_map[ft_name] = (ft_generator(df_train), ft_generator(df_test))
    end

    tb_info_map = Dict()
    ret_bp = df_train[!, ret_col_name]
    for (ft_name, ft_tuple) in ft_map
        # println("$(ft_name)")
        tb_info_map[ft_name] = get_ev_info(ft_tuple[1], ret_bp, th_vec)
    end

    df_evv_train, df_evv_test = DataFrame(), DataFrame()
    for ft_name in keys(ft_map)
        tb_info = tb_info_map[ft_name]
        df_evv_train[!, ft_name] = get_wsv_parallel(tb_info, ft_map[ft_name][1])
        df_evv_test[!, ft_name] = get_wsv_parallel(tb_info, ft_map[ft_name][2])
    end

    # 모든 feature들의 동일가중 평균 계산
    df_evv_train[!, "equal_weight_sum"] .= sum(df_evv_train[!, ft_name] for ft_name in keys(ft_map)) ./ length(ft_map)
    df_evv_test[!, "equal_weight_sum"] .= sum(df_evv_test[!, ft_name] for ft_name in keys(ft_map)) ./ length(ft_map)

    return df_evv_train, df_evv_test
end


function get_evv_df_fast(df_train, df_test, ft_gen_map, ret_bp, th_vec)
    ft_tasks = [Threads.@spawn (ft_name, ft_generator(df_train), ft_generator(df_test)) for (ft_name, ft_generator) in ft_gen_map]
    ft_map = Dict(task_result[1] => (task_result[2], task_result[3]) for task_result in fetch.(ft_tasks))

    # ret_bp = df_train[!, ret_col_name]
    tb_info_tasks = [Threads.@spawn (ft_name, get_ev_info(ft_tuple[1], ret_bp, th_vec)) for (ft_name, ft_tuple) in ft_map]
    tb_info_map = Dict(fetch(task)[1] => fetch(task)[2] for task in tb_info_tasks)

    df_evv_train, df_evv_test = DataFrame(), DataFrame()
    for ft_name in keys(ft_map)
        tb_info = tb_info_map[ft_name]
        df_evv_train[!, ft_name] = get_wsv_parallel(tb_info, ft_map[ft_name][1])
        df_evv_test[!, ft_name] = get_wsv_parallel(tb_info, ft_map[ft_name][2])
    end

    # 모든 feature들의 동일가중 평균 계산
    df_evv_train[!, "equal_weight_sum"] .= sum(df_evv_train[!, ft_name] for ft_name in keys(ft_map)) ./ length(ft_map)
    df_evv_test[!, "equal_weight_sum"] .= sum(df_evv_test[!, ft_name] for ft_name in keys(ft_map)) ./ length(ft_map)

    return df_evv_train, df_evv_test
end


function get_evv_df_fast2(df_train, df_test, ft_gen_map, ret_bp, th_vec; is_calc_lr=true)
    # ret_bp = df_train[!, ret_col_name]
    # df_evv_train, df_evv_test = DataFrame(), DataFrame()
    # @threads for ft_name in collect(keys(ft_gen_map))
    #     ft_train = ft_gen_map[ft_name](df_train)
    #     ft_test = ft_gen_map[ft_name](df_test)
    #     tb_info = get_ev_info(ft_train, ret_bp, th_vec)
    #     df_evv_train[!, ft_name] = get_wsv_parallel(tb_info, ft_train)
    #     df_evv_test[!, ft_name] = get_wsv_parallel(tb_info, ft_test)
    # end

    ft_names = collect(keys(ft_gen_map))
    
    # 병렬로 계산만 수행
    results = Vector{Tuple{String, Vector{Float64}, Vector{Float64}}}(undef, length(ft_names))
    @threads for i in 1:length(ft_names)
        ft_name = ft_names[i]
        ft_train = ft_gen_map[ft_name](df_train)
        ft_test = ft_gen_map[ft_name](df_test)
        tb_info = get_ev_info(ft_train, ret_bp, th_vec)
        evv_train = get_wsv_parallel(tb_info, ft_train)
        evv_test = get_wsv_parallel(tb_info, ft_test)
        results[i] = (ft_name, evv_train, evv_test)
    end
    
    # DataFrame에 순차적으로 추가
    df_evv_train, df_evv_test = DataFrame(), DataFrame()
    for (ft_name, evv_train, evv_test) in results
        df_evv_train[!, ft_name] = evv_train
        df_evv_test[!, ft_name] = evv_test
    end

    # 모든 feature들의 동일가중 평균 계산
    df_evv_train[!, "equal_weight_sum"] .= sum(df_evv_train[!, ft_name] for ft_name in keys(ft_gen_map)) ./ length(ft_gen_map)
    df_evv_test[!, "equal_weight_sum"] .= sum(df_evv_test[!, ft_name] for ft_name in keys(ft_gen_map)) ./ length(ft_gen_map)
    
    is_calc_lr && calc_weighted_sum(df_train, df_evv_train, df_evv_test)

    return df_evv_train, df_evv_test
end


function calc_weighted_sum(df_train, df_evv_train, df_evv_test; res_col_name="lin_reg_weight_sum")
    coef_lr_train = get_lr_coef(df_evv_train, df_train.ret_bp)
    weights = Dict(cf[1] => cf[2] for cf in coef_lr_train.coefs)
    df_evv_train[!, res_col_name] = calc_linear_prediction(df_evv_train, weights, coef_lr_train.intercept)
    df_evv_test[!, res_col_name] = calc_linear_prediction(df_evv_test, weights, coef_lr_train.intercept)
end


function calc_weighted_sum_return(df_train, df_evv_train, df_evv_test)
    coef_lr_train = get_lr_coef(df_evv_train, df_train.ret_bp)
    weights = Dict(cf[1] => cf[2] for cf in coef_lr_train.coefs)
    lin_reg_weight_sum_train = calc_linear_prediction(df_evv_train, weights, coef_lr_train.intercept)
    lin_reg_weight_sum_test = calc_linear_prediction(df_evv_test, weights, coef_lr_train.intercept)
    return lin_reg_weight_sum_train, lin_reg_weight_sum_test
end


function calc_linear_prediction(df_evv, coef_dict, intercept)
    prediction = fill(intercept, size(df_evv, 1))
    
    for (ft_name, coef) in coef_dict
        prediction .+= df_evv[!, ft_name] .* coef
    end
    
    return prediction
end

# 레짐별 가중평균 계산 함수 
function calc_regime_weighted_sum(df_evv, regime_vec, regime_weights_dict, ft_names)
    n_samples = size(df_evv, 1)
    all_regimes = unique(regime_vec)
    total_weight_map = Dict()
    for regime in all_regimes
        total_weight_map[regime] = sum(values(regime_weights_dict[regime]))
    end

    regime_weight_map = Dict()
    for ft_name in ft_names
        regime_weight_map[ft_name] = fill(0.0, n_samples)
        for regime in all_regimes
            regime_weight_map[ft_name][regime_vec .== regime] .= regime_weights_dict[regime][ft_name] / total_weight_map[regime]
        end
    end
    weighted_sum = sum(df_evv[!, ft_name] .* regime_weight_map[ft_name] for ft_name in ft_names)
    return weighted_sum
end


function summary_evv_result(df_evv_train, df_evv_test, ft_names, th)
    er_bps_train, er_bps_test = [], []
    for ft_name in ft_names
        sl_train, ss_train = get_signals(df_evv_train[!, ft_name], th)
        plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_print=false)
        er_bp_train = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_train])
        push!(er_bps_train, er_bp_train)
    
        sl_test, ss_test = get_signals(df_evv_test[!, ft_name], th)
        plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_print=false)
        er_bp_test = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_test])
        push!(er_bps_test, er_bp_test)
    end
    
    sl_train, ss_train = get_signals(df_evv_train.equal_weight_sum, th)
    plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_print=false)
    er_bp_train = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_train])
    push!(er_bps_train, er_bp_train)
    
    sl_test, ss_test = get_signals(df_evv_test.equal_weight_sum, th)
    plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_print=false)
    er_bp_test = mean([10_000 * tr[6] / tr[4] for tr in tr_res_vec_test])
    push!(er_bps_test, er_bp_test)
        
    train_bps_str = join(string.(round.(er_bps_train[1 : end-1], digits=2)), ", ")
    test_bps_str = join(string.(round.(er_bps_test[1 : end-1], digits=2)), ", ")
    println("th: $th  $(join(ft_names, "  "))")
    println("Train : [$(train_bps_str)] -> $(round(er_bps_train[end], digits=2)) ($(length(tr_res_vec_train)) tr)")
    println("Test  : [$(test_bps_str)] -> $(round(er_bps_test[end], digits=2)) ($(length(tr_res_vec_test)) tr)")
end


function view_norm_ret_path_by_df_evv(th, df_evv_train, df_evv_test, df_train, df_test; er_name="equal_weight_sum")
    sl_train, ss_train = get_signals(df_evv_train[!, er_name], th)
    plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_print=false)
    er_bp_train = mean(10_000 * tr[6] / tr[4] for tr in tr_res_vec_train)
    
    sl_test, ss_test = get_signals(df_evv_test[!, er_name], th)
    plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_print=false)
    er_bp_test = mean(10_000 * tr[6] / tr[4] for tr in tr_res_vec_test)

    wap_train = df_train.WAP_Lag_0ms
    avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec_train, wap_train)
    plt_train = plot(avg_norm_paths, label="avg")
    plot!(plt_train, median_norm_paths, label="median")
    title!(plt_train, "Train - er: $(round(er_bp_train, digits=3))bp")
    # display(plt_train)

    wap_test = df_test.WAP_Lag_0ms
    avg_norm_paths, median_norm_paths = get_norm_ret_bp_path(tr_res_vec_test, wap_test)
    plt_test = plot(avg_norm_paths, label="avg")
    plot!(plt_test, median_norm_paths, label="median")
    title!(plt_test, "Test - er: $(round(er_bp_test, digits=3))bp")
    # display(plt_test)

    plt_total = plot(plt_train, plt_test, layout=(2, 1), plot_title="Norm Ret(bp) th: $(th)")
    display(plt_total)
    return plt_total
end


function get_tr_res_vecs_by_dfs(th, df_train, df_evv_train, df_test, df_evv_test, er_name="equal_weight_sum")
    sl_train, ss_train = get_signals(df_evv_train[!, er_name], th)
    _plt_train, tr_res_vec_train = backtest_sica_2(sl_train, ss_train, df_train.WAP_Lag_200ms, df_train.WAP_Lag_0ms, df_train.timestamp, is_print=false)
    
    sl_test, ss_test = get_signals(df_evv_test[!, er_name], th)
    _plt_test, tr_res_vec_test = backtest_sica_2(sl_test, ss_test, df_test.WAP_Lag_200ms, df_test.WAP_Lag_0ms, df_test.timestamp, is_print=false)

    return tr_res_vec_train, tr_res_vec_test
end


function get_plt_total_by_tr_res_vec(tr_res_vec, category, feature, wap, win)
    ial_in_l = [tr[1] for tr in tr_res_vec if tr[3] > 0]
    ial_out_l = [tr[2] for tr in tr_res_vec if tr[3] > 0]
    ial_in_s = [tr[1] for tr in tr_res_vec if tr[3] < 0]
    ial_out_s = [tr[2] for tr in tr_res_vec if tr[3] < 0]

    profiles_in_l = get_at_time_profile2(feature, wap, win, ial_in_l)
    profiles_out_l = get_at_time_profile2(feature, wap, win, ial_out_l)
    profiles_in_s = get_at_time_profile2(feature, wap, win, ial_in_s)
    profiles_out_s = get_at_time_profile2(feature, wap, win, ial_out_s)

    p_in_l = plot(-win:win, profiles_in_l[2], label="avg wap")
    scatter!(p_in_l, [0], [0])
    plot!(twinx(p_in_l), -win:win, profiles_in_l[1], label="avg fv", color=:gray, linestyle=:dot)
    title!(p_in_l, "In - Long")
    
    p_in_s = plot(-win:win, profiles_in_s[2], label="avg wap")
    scatter!(p_in_s, [0], [0])
    plot!(twinx(p_in_s), -win:win, profiles_in_s[1], label="avg fv", color=:gray, linestyle=:dot)
    title!(p_in_s, "In - Short")
    
    p_out_l = plot(-win:win, profiles_out_l[2], label="avg wap")
    scatter!(p_out_l, [0], [0])
    plot!(twinx(p_out_l), -win:win, profiles_out_l[1], label="avg fv", color=:gray, linestyle=:dot)
    title!(p_out_l, "Out - Long")
    
    p_out_s = plot(-win:win, profiles_out_s[2], label="avg wap")
    scatter!(p_out_s, [0], [0])
    plot!(twinx(p_out_s), -win:win, profiles_out_s[1], label="avg fv", color=:gray, linestyle=:dot)
    title!(p_out_s, "Out - Short")

    plt_total = plot(p_in_l, p_out_l, p_in_s, p_out_s, layout=(2, 2), legend=false, plot_title=category)
    return plt_total
end


function get_lr_coef(df_evv, ret_bp)
    matrix_reg = hcat(Matrix(df_evv), ret_bp)
    keep_rows = .!any(isnan.(matrix_reg), dims=2)[:]
    lr = @load LinearRegressor pkg=MLJLinearModels
    lrm = lr()
    mach_lr = machine(lrm, df_evv[keep_rows, :], ret_bp[keep_rows])
    fit!(mach_lr)
    coef_lr = fitted_params(mach_lr)
    return coef_lr
end


function calc_er_vec_by_coef(df_evv, coef)
    ft_name_vec::Vector{Symbol} = []
    coef_vec::Vector{Float64} = []
    for (ft_name, cf_value) in coef.coefs
        push!(ft_name_vec, ft_name)
        push!(coef_vec, cf_value)
    end
    x_matrix = Matrix(df_evv[!, ft_name_vec])
    return x_matrix * coef_vec .+ coef.intercept
end


"""
feature_for_regime_divide 로 레짐을 나누고 그 벡터를 반환한다
regime 벡터를 1, 2, 3, ...으로 채운다
quantile th 는 0 ~ 1.0 값을 받고, 하위 비율로 계산
quantile th 값이 0.4 라면, 하위 40%, 상위 60%를 나눔 
"""
function get_regime_vec(feature_for_regime_divide, regime_quantile_th_vec) # 이거 test에서는 못씀
    regime_vec = fill(1, length(feature_for_regime_divide))
    for (regime_num, regime_th) in enumerate(regime_quantile_th_vec)
        mask = feature_for_regime_divide .>= quantile(skipnan(feature_for_regime_divide), regime_th)
        regime_vec[mask] .= regime_num + 1
    end
    return regime_vec
end


function get_regime_th_values_by_quantile_th(feature_for_regime_divide, regime_quantile_th_vec)
    res = []
    for regime_th in regime_quantile_th_vec
        push!(res, quantile(skipnan(feature_for_regime_divide), regime_th))
    end
    return res
end


function get_regime_vec_by_th_values(feature_for_regime_divide, regime_th_values)
    regime_vec = fill(1, length(feature_for_regime_divide))
    for (regime_num, regime_th_value) in enumerate(regime_th_values)
        mask = feature_for_regime_divide .>= regime_th_value
        regime_vec[mask] .= regime_num + 1
    end
    return regime_vec
end


function get_all_combinations(unique_vals)
    if isempty(unique_vals)
        return []
    elseif length(unique_vals) == 1
        return [(val,) for val in unique_vals[1]]
    else
        combinations = []
        for combo in get_all_combinations(unique_vals[2:end])
            for val in unique_vals[1]
                push!(combinations, (val, combo...))
            end
        end
        return combinations
    end
end


"""
특정 regime 조합에 해당하는 데이터의 마스크를 구하는 함수
"""
function get_regime_mask(rv_train, regime_combination)
    n_rows = length(rv_train[1])
    mask = fill(true, n_rows)
    
    for (i, regime_val) in enumerate(regime_combination)
        mask .&= (rv_train[i] .== regime_val)
    end
    
    return mask
end

"""
모든 regime 조합에 대한 마스크를 한번에 구하는 함수
"""
function get_all_regime_masks(rv_train, all_combinations)
    regime_masks = Dict()
    
    for combo in all_combinations
        mask = get_regime_mask(rv_train, combo)
        regime_masks[combo] = mask
    end
    
    return regime_masks
end


function get_regime_coef_map(df_evv_train, ret_bp_train, rg_mask_map_train; min_samples=100, is_print=false)
    rg_coef_map = Dict()
    feature_names = names(df_evv_train)
    equal_weight = 1.0 / length(feature_names)
    
    for (regime_combo, mask) in rg_mask_map_train
        n_samples = sum(mask)
        
        if n_samples < min_samples
            # Equal weight 설정
            equal_coefs = [(Symbol(fname), equal_weight) for fname in feature_names]
            rg_coef_map[regime_combo] = (coefs = equal_coefs, intercept = 0.0)
            is_print && println("Regime $regime_combo: Equal weight ($(n_samples) samples)")
            
        else
            # 선형회귀
            df_filtered = df_evv_train[mask, :]
            ret_filtered = ret_bp_train[mask]
            rg_coef_map[regime_combo] = get_lr_coef(df_filtered, ret_filtered)
            is_print && println("Regime $regime_combo: Regression ($(n_samples) samples)")
        end
    end
    
    return rg_coef_map
end


function calc_regime_based_er_vec(rg_info_map, df_evv, mask_symbol; default_value=NaN)
    n_rows = size(df_evv, 1)
    er_vec = fill(default_value, n_rows)
    
    # 각 regime별로 계수 적용
    for (regime_idx, rg_info) in rg_info_map
        mask = rg_info[Symbol(mask_symbol)]
        coef = rg_info.coef
        
        if sum(mask) == 0
            continue  # 해당 regime에 데이터가 없으면 건너뛰기
        end
        
        # 해당 regime 데이터에 대해 prediction 계산
        prediction = fill(coef.intercept, sum(mask))
        df_regime = df_evv[mask, :]
        
        for (ft_name, coef_val) in coef.coefs
            prediction .+= df_regime[!, ft_name] .* coef_val
        end
        
        # 결과 벡터에 할당
        er_vec[mask] = prediction
    end
    
    return er_vec
end


function calc_multi_regime_based_er_vec(rg_info_map, df_evv, mask_symbol; default_value=NaN)
    n_rows = size(df_evv, 1)
    er_vec = fill(default_value, n_rows)

    # 각 regime별로 계수 적용
    for (regime_idx, rg_info) in rg_info_map
        mask = rg_info[Symbol(mask_symbol)]
        coef = rg_info.coef
        
        if sum(mask) == 0
            continue  # 해당 regime에 데이터가 없으면 건너뛰기
        end
        
        # 해당 regime 데이터에 대해 prediction 계산
        prediction = fill(coef.intercept, sum(mask))
        df_regime = df_evv[mask, :]
        
        for (ft_name, coef_val) in coef.coefs
            if occursin("_weight_sum", string(ft_name)) || string(ft_name) == "timestamp" continue end
            for sb in symbols
                # prediction .+= df_regime[!, "$(sb)__$(ft_name)"] .* coef_val
                prediction .+= df_regime[!, ft_name] .* coef_val
            end
        end
        
        # 결과 벡터에 할당
        er_vec[mask] = prediction
    end
    return er_vec
end


function get_rg_info_map(regime_ft_gen_map, df_train, df_test, df_evv_train)
    ft_name_vec::Vector{String} = []
    rv_train::Vector{Vector{Int64}} = []
    rv_test::Vector{Vector{Int64}} = []
    for (ft_name, (generator, rg_th_vec)) in regime_ft_gen_map
        println(ft_name, "  ", generator, "  ", rg_th_vec)
        ft_train = generator(df_train)
        ft_test = generator(df_test)
        regime_th_values = get_regime_th_values_by_quantile_th(ft_train, rg_th_vec)
        regime_vec_train = get_regime_vec_by_th_values(ft_train, regime_th_values)
        regime_vec_test = get_regime_vec_by_th_values(ft_test, regime_th_values)
        push!(ft_name_vec, ft_name)
        push!(rv_train, regime_vec_train)
        push!(rv_test, regime_vec_test)
    end

    unique_values = [sort(unique(rv)) for rv in rv_train]
    all_combined_regime = get_all_combinations(unique_values)
    rg_mask_map_train = get_all_regime_masks(rv_train, all_combined_regime)
    rg_mask_map_test = get_all_regime_masks(rv_test, all_combined_regime)

    rg_coef_map = get_regime_coef_map(df_evv_train, df_train.ret_bp, rg_mask_map_train; min_samples=100)

    rg_info_map = SortedDict()
    for (regime_idx, comb_regime) in enumerate(all_combined_regime)
        mask_train = rg_mask_map_train[comb_regime]
        mask_test = rg_mask_map_test[comb_regime]
        coef = rg_coef_map[comb_regime]
        rg_info_map[regime_idx] = (; mask_train, mask_test, coef, comb_regime)
    end
    return rg_info_map
end



function get_multi_rg_info_map(regime_ft_gen_map, df_train, df_test, df_evv_train)
    ft_name_vec::Vector{String} = []
    rv_train::Vector{Vector{Int64}} = []
    rv_test::Vector{Vector{Int64}} = []
    for (ft_name, (generator, rg_th_vec)) in regime_ft_gen_map
        println(ft_name, "  ", generator, "  ", rg_th_vec)
        ft_train = generator(df_train)
        ft_test = generator(df_test)
        regime_th_values = get_regime_th_values_by_quantile_th(ft_train, rg_th_vec)
        regime_vec_train = get_regime_vec_by_th_values(ft_train, regime_th_values)
        regime_vec_test = get_regime_vec_by_th_values(ft_test, regime_th_values)
        push!(ft_name_vec, ft_name)
        push!(rv_train, regime_vec_train)
        push!(rv_test, regime_vec_test)
    end

    unique_values = [sort(unique(rv)) for rv in rv_train]
    all_combined_regime = get_all_combinations(unique_values)
    rg_mask_map_train = get_all_regime_masks(rv_train, all_combined_regime)
    rg_mask_map_test = get_all_regime_masks(rv_test, all_combined_regime)

    rg_coef_map = get_regime_coef_map(df_evv_train, df_train.ret_bp, rg_mask_map_train; min_samples=100)

    rg_info_map = SortedDict()
    for (regime_idx, comb_regime) in enumerate(all_combined_regime)
        mask_train = rg_mask_map_train[comb_regime]
        mask_test = rg_mask_map_test[comb_regime]
        coef = rg_coef_map[comb_regime]
        rg_info_map[regime_idx] = (; mask_train, mask_test, coef, comb_regime)
    end
    return rg_info_map
end


function calc_avg_ret_bp(tr_res_vec)
    profit_bp_vec = [10_000 * tr[6] / tr[4] for tr in tr_res_vec]
    return round(mean(profit_bp_vec), digits=4)
end


function get_multi_equal_weight_sum(df_evv_multi_sb)
    multi_equal_weight_sum = fill(0.0, nrow(df_evv_multi_sb))
    cols = [ft_name for ft_name in names(df_evv_multi_sb) if !occursin("weight_sum", ft_name) && ft_name !== "timestamp"]
    ew = 1 / length(cols)
    for ft_name in cols
        multi_equal_weight_sum .+= (df_evv_multi_sb[!, ft_name]) ./ ew
    end
    return multi_equal_weight_sum
end




