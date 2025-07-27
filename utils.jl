
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
    display(p)
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

    return final_plot, tr_res_vec
end


function backtest_sica_2(
    signal_long, signal_short, entry_price_vec, exit_price_vec, timestamp_vec; 
    symbol="", name="backtest result", is_display=false,
    resampling_interval=1800,
    )        
    # signal_long = aq_rb .>= quantile(aq_rb[.!isnan.(aq_rb)], 0.9)
    # signal_short = aq_rb .<= quantile(aq_rb[.!isnan.(aq_rb)], 0.1)
    # entry_price_vec = df.WAP_Lag_300ms
    # exit_price_vec = df.WAP_Lag_0ms

    tr_res_vec::Vector{Tuple{Int64, Int64, Float64, Float64, Float64}} = []
    curr_pos = 0
    idx_pos_entry = -1
    entry_price = 0.0

    total_len = length(signal_long)
    for idx in 1:total_len
        if isnan(entry_price_vec[idx]) || isnan(exit_price_vec[idx]) continue end
        if curr_pos !== 0
            if (curr_pos > 0 && signal_short[idx]) || (curr_pos < 0 && signal_long[idx])
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
                idx_pos_entry = idx
            elseif signal_short[idx]
                curr_pos = -1
                entry_price = entry_price_vec[idx]
                idx_pos_entry = idx
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
    total_profit_bp_vec = []
    win_profit_bp_vec, lose_profit_bp_vec = [], []
    win_long_profit_bp_vec, win_short_profit_bp_vec = [], []
    lose_long_profit_bp_vec, lose_short_profit_bp_vec = [], []
    for tr_res in tr_res_vec
        profit = tr_res[5]
        profit_bp = 10000 * profit / tr_res[3]
        if profit > 0
            win_cnt += 1
            push!(win_profit_bp_vec, profit_bp)
            push!(total_profit_bp_vec, profit_bp)
            if tr_res[2] > 0
                push!(win_long_profit_bp_vec, profit_bp)
            elseif tr_res[2] < 0
                push!(win_short_profit_bp_vec, profit_bp)
            end
        elseif profit < 0
            lose_cnt += 1
            push!(lose_profit_bp_vec, profit_bp)
            push!(total_profit_bp_vec, profit_bp)
            if tr_res[2] > 0
                push!(lose_long_profit_bp_vec, profit_bp)
            elseif tr_res[2] < 0
                push!(lose_short_profit_bp_vec, profit_bp)
            end
        end
    end
    win_rate, lose_rate = win_cnt / length(tr_res_vec), lose_cnt / length(tr_res_vec)
    println("Win Rate: $(round(win_rate * 100, digits=2))%, Lose Rate: $(round(lose_rate * 100, digits=2))%")
    println("Mean Profig(bp) : $(round(mean(total_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) - Win: $(round(mean(win_profit_bp_vec), digits=4))\tLose: $(round(mean(lose_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) Win - Long: $(round(mean(win_long_profit_bp_vec), digits=4))\tShort: $(round(mean(win_short_profit_bp_vec), digits=4))")
    println("Mean Profit(bp) Lose - Long: $(round(mean(lose_long_profit_bp_vec), digits=4))\tShort: $(round(mean(lose_short_profit_bp_vec), digits=4))")

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
        Avg. Win  Long  (bp): $(round(mean(win_long_profit_bp_vec), digits=4))
        Avg. Win  Short (bp): $(round(mean(win_short_profit_bp_vec), digits=4))
        Avg. Loss Long  (bp): $(round(mean(lose_long_profit_bp_vec), digits=4))
        Avg. Loss Short (bp): $(round(mean(lose_short_profit_bp_vec), digits=4))
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
        # t_source = df_source.timestamp[idx_source]

        if t_wap < t_source
            if idx_source > 1
                idx_copy = idx_source - 1
                for col in cols
                    df[idx_wap, col] = ismissing(df_source[idx_copy, col]) ? NaN : df_source[idx_copy, col]
                end
            # else
            #     println("can't write cause idx_deri == 1")
            end
            idx_wap += 1
        else
            idx_source += 1
        end
    end
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



