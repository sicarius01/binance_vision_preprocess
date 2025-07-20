


using CSV
using DataFrames
using Dates
using ZipFile
using Plots




#######################################################################




function read_zip_to_df(zip_path::String)
    zip_reader = ZipFile.Reader(zip_path)
    csv_file = filter(f -> endswith(f.name, ".csv"), zip_reader.files)[1]
    data = read(csv_file)
    io = IOBuffer(data)
    df = CSV.read(io, DataFrame; delim = ',', header = true, missingstring = "null")
    close(zip_reader)
    return df
end


function is_no_alphabet(rows)
    for row in rows
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
    fl = [monthly_files; daily_files]
    sort!(fl)
    return fl
end


um_path = raw"C:\Users\haeso\Documents\project\Binance_Futures_Download\binance-vision-data\futures\um"
symbol = "BTCUSDT"
dtype = "premiumIndexKlines"


file_list_pi = get_file_list(um_path, symbol, dtype)
df_pi = read_zip_to_df(file_list_pi[1])



dtype = "aggTrades"
file_list_aggTr = get_file_list(um_path, symbol, dtype)
df_aggTr = read_zip_to_df(file_list_aggTr[1])



dtype = "fundingRate"
file_list_fdr = get_file_list(um_path, symbol, dtype)
df_fdr = read_zip_to_df(file_list_fdr[1])



dts = diff(df_aggTr.transact_time)
histogram(dts[dts .> 0], xlims=(-30, 1000))
minimum(dts[dts .> 0])




# timestamp는 ms 단위
# aggTrades는 모이는대로 찍히고, premiumIndexKlines는 분 단위, fundingRate는 8시간 단위
# aggTrades만 전처리 하자. 1초봉을 만들자. ohlcv 그리고 total_id_cnt, buy_flow_id_cnt, sell_flow_id_cnt, vwap까지 만들면 될듯 



df_aggTr.price

df_aggTr




