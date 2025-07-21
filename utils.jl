
using CSV
using DataFrames
using Dates
using ZipFile
using Plots
using Parquet2
using StatsBase
using Base.Threads




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

function get_pct_mean(X, pct, target)
    x = X[.!isnan.(X)]
    th_low, th_high = quantile(x, pct/100), quantile(x, (100-pct)/100)
    # println("th_low: $(th_low), th_high: $(th_high)")
    mask_down = X .<= th_low
    mask_up = X .>= th_high
    return mean(target[mask_down]), mean(target[mask_up])
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

